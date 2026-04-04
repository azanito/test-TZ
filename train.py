#!/usr/bin/env python3
"""
train.py — QLoRA fine-tuning of a causal LLM using PEFT + TRL SFTTrainer.

Supports:
  - mistralai/Mistral-7B-v0.1  (default)
  - meta-llama/Meta-Llama-3-8B

Works on:
  - Google Colab T4 (16 GB VRAM) — use default settings
  - Local GPU with ≥ 8 GB VRAM  — reduce batch_size / seq_length if needed

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
    `prepare_model_for_kbit_training` freezes base weights and casts
    LayerNorm layers to float32 for stable training.
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # pad_token must exist for batched training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right-padding is required by SFTTrainer
    tokenizer.padding_side = "right"

    # 4-bit quantisation config (doubles quantisation saves ~25% VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4 is optimal for normally-distributed weights
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading model in 4-bit: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # automatically shard across available GPUs / CPU
        trust_remote_code=True,
    )

    # Prepare for LoRA training: freeze base weights, upcast norms to fp32
    model = prepare_model_for_kbit_training(model)
    # Disable the KV cache — not needed during training, wastes memory
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
        target_modules=args.target_modules,  # only Q and V projections by default
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # shows how many params are actually trained
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── WandB / console logging ───────────────────────────────────────────────
    if args.use_wandb:
        import wandb  # noqa: F401  (checked at runtime)
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

    # ── SFTConfig (replaces TrainingArguments in TRL ≥ 1.0) ──────────────────
    # max_length is the TRL 1.0 equivalent of max_seq_length
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Effective batch size = per_device_batch * gradient_accumulation_steps
        learning_rate=args.learning_rate,
        fp16=True,                      # mixed-precision training
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,             # keep only the latest checkpoint
        warmup_steps=args.warmup_steps,
        report_to=report_to,
        optim="paged_adamw_8bit",       # 8-bit paged AdamW — saves VRAM
        lr_scheduler_type="cosine",
        group_by_length=True,           # group similarly-lengthed samples → less padding
        dataloader_pin_memory=False,    # avoids OOM on some systems
        max_length=args.max_seq_length, # TRL 1.0 API: controls tokenisation truncation
    )

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,     # TRL 1.0: tokenizer → processing_class
        train_dataset=dataset,
        formatting_func=build_prompt,   # converts each sample dict → prompt string
        args=training_args,
    )

    logger.info("Starting training...")
    trainer.train()

    # ── Save ONLY the LoRA adapter (≈ a few MB, not the full 7B model) ────────
    logger.info(f"Saving LoRA adapter → {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
