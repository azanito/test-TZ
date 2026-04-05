import argparse
import logging
import os

import torch
torch.cuda.empty_cache()

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--data_path", type=str, default="data/dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="output/lora-adapter")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


def build_prompt(sample):
    return PROMPT_TEMPLATE.format(
        instruction=sample["instruction"],
        response=sample["response"],
    )


def load_model_and_tokenizer(model_name):
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading model in 4-bit fp16: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    torch.backends.cuda.matmul.allow_tf32 = True

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    return model, tokenizer


def apply_lora(model, args):
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


def verify_adapter(output_dir):
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not os.path.exists(os.path.join(output_dir, f))]
    if missing:
        if "adapter_model.safetensors" in missing:
            if os.path.exists(os.path.join(output_dir, "adapter_model.bin")):
                missing.remove("adapter_model.safetensors")
    if missing:
        raise RuntimeError(f"Adapter save incomplete, missing: {missing}")
    logger.info(f"Adapter saved to {output_dir}")
    for f in os.listdir(output_dir):
        logger.info(f"  {f}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        import wandb  # noqa
        os.environ.setdefault("WANDB_PROJECT", "qlora-finetuning")
        report_to = "wandb"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"

    logger.info(f"Loading dataset: {args.data_path}")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    logger.info(f"Training samples: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model = apply_lora(model, args)

    # cast LoRA params to fp16 — Mistral loads in bf16 by default which breaks fp16 AMP on T4
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float16)

    model.gradient_checkpointing_enable()

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_steps=args.warmup_steps,
        report_to=report_to,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        group_by_length=True,
        dataloader_pin_memory=False,
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=build_prompt,
        args=training_args,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving LoRA adapter -> {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    verify_adapter(args.output_dir)
    logger.info("Training finished successfully.")


if __name__ == "__main__":
    main()
