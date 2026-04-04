#!/usr/bin/env python3
"""
train.py — QLoRA fine-tuning of a causal LLM using PEFT + TRL SFTTrainer.

Supports:
  - mistralai/Mistral-7B-v0.1  (default)
  - meta-llama/Meta-Llama-3-8B

Works on:
  - Google Colab T4 (16 GB VRAM) — use default settings
  - Local GPU with ≥ 8 GB VRAM  — reduce --max_seq_length 256 if needed

Usage:
    python train.py
    python train.py --model_name mistralai/Mistral-7B-v0.1 --num_train_epochs 3 --use_wandb
"""

import argparse
import logging
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template — must match the format used in evaluate.py
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning script")

    # Model & data
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model from HuggingFace hub",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/dataset.jsonl",
        help="Path to JSONL training data produced by prepare_data.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/lora-adapter",
        help="Directory where the LoRA adapter will be saved",
    )

    # LoRA hyper-parameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Transformer modules to apply LoRA to",
    )

    # Training hyper-parameters
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)

    # Logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires `wandb login`)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prompt formatter
# ---------------------------------------------------------------------------
def build_prompt(sample: dict) -> str:
    """Combine instruction + response into a single training string."""
    return PROMPT_TEMPLATE.format(
        instruction=sample["instruction"],
        response=sample["response"],
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name: str):
    """
    Load the base model in 4-bit NF4 (QLoRA) and its tokenizer.

    Key decisions:
    - bnb_4bit_compute_dtype=torch.float16  → T4 does NOT support bf16;
      using bf16 here causes "_amp_foreach_non_finite_check_and_unscale_cuda
      not implemented for BFloat16" at the first optimizer step.
    - prepare_model_for_kbit_training       → freezes base weights, upcasts
      LayerNorm to fp32 for numerical stability.
    - gradient_checkpointing is enabled later via SFTConfig to save VRAM.
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right-padding required by SFTTrainer packing / collation
    tokenizer.padding_side = "right"

    # Explicitly fp16 compute dtype — never bf16 on T4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # ← must be float16, not bfloat16
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading model in 4-bit fp16: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Allow tf32 for matmul — safe on Ampere+ and speeds up training slightly
    torch.backends.cuda.matmul.allow_tf32 = True

    # Freeze base weights, upcast norms to fp32
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,    # activations recomputed → saves VRAM
    )
    # KV cache is incompatible with gradient checkpointing
    model.config.use_cache = False

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
def apply_lora(model, args: argparse.Namespace):
    """Wrap the base model with a trainable LoRA adapter."""
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Adapter verification
# ---------------------------------------------------------------------------
def verify_adapter(output_dir: str) -> None:
    """Confirm that the two critical adapter files exist after saving."""
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not os.path.exists(os.path.join(output_dir, f))]
    if missing:
        # adapter_model.bin is the legacy name — accept either
        if "adapter_model.safetensors" in missing:
            bin_path = os.path.join(output_dir, "adapter_model.bin")
            if os.path.exists(bin_path):
                missing.remove("adapter_model.safetensors")
    if missing:
        raise RuntimeError(
            f"Adapter save incomplete — missing files: {missing}\n"
            f"Check {output_dir} and re-run training."
        )
    logger.info(f"Adapter verified: {output_dir}")
    for f in os.listdir(output_dir):
        logger.info(f"  {f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── WandB / console logging ───────────────────────────────────────────────
    if args.use_wandb:
        import wandb  # noqa: F401
        os.environ.setdefault("WANDB_PROJECT", "qlora-finetuning")
        report_to = "wandb"
        logger.info("WandB logging enabled.")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.data_path}")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    logger.info(f"Training samples: {len(dataset)}")

    # ── Model & tokenizer ─────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model = apply_lora(model, args)

    # ── SFTConfig ─────────────────────────────────────────────────────────────
    # fp16=True  / bf16=False  → mandatory for T4 (Turing architecture)
    # gradient_checkpointing   → trades compute for VRAM
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,                          # ← float16 AMP — required for T4
        bf16=False,                         # ← explicitly off — T4 has no bf16
        gradient_checkpointing=True,        # ← recompute activations to save VRAM
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_steps=args.warmup_steps,
        report_to=report_to,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        group_by_length=True,
        dataloader_pin_memory=False,
        max_length=args.max_seq_length,     # TRL 1.0 API
    )

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,         # TRL 1.0: processing_class replaces tokenizer
        train_dataset=dataset,
        formatting_func=build_prompt,
        args=training_args,
    )

    logger.info("Starting training...")
    trainer.train()

    # ── Save ONLY the LoRA adapter (~30 MB, not the full 7B base) ─────────────
    logger.info(f"Saving LoRA adapter → {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Confirm critical files landed on disk
    verify_adapter(args.output_dir)
    logger.info("Training finished successfully.")


if __name__ == "__main__":
    main()
