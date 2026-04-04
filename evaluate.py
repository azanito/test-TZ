#!/usr/bin/env python3
"""
evaluate.py — Compare base model vs. LoRA fine-tuned model using ROUGE-L.

Workflow:
  1. Load base model (4-bit)
  2. Generate responses to TEST_PROMPTS
  3. Attach LoRA adapter → fine-tuned model
  4. Generate the same prompts again
  5. Compute ROUGE-L between fine-tuned and base outputs
  6. Save full results to evaluation/results.json
  7. Print a human-readable summary

Usage:
    python evaluate.py
    python evaluate.py --model_name mistralai/Mistral-7B-v0.1 --max_new_tokens 256
"""

import argparse
import json
import logging
import os

import torch
from peft import PeftModel
from rouge_score import rouge_scorer as rs_module
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
# Evaluation prompts  (15 diverse instructions)
# ---------------------------------------------------------------------------
TEST_PROMPTS = [
    "Explain the concept of photosynthesis in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python 2 and Python 3?",
    "Describe the water cycle step by step.",
    "How do you make a basic pasta dish? Provide step-by-step instructions.",
    "What is machine learning and how does it work?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "List five practical tips for improving productivity at work.",
    "Explain how vaccines work to protect the human body against disease.",
    "What is the difference between RAM and ROM in a computer?",
    "How do airplanes generate lift?",
    "Describe the process of making chocolate from raw cacao beans.",
    "What causes earthquakes and where do they occur most often?",
    "Explain the concept of supply and demand in economics with an example.",
    "How does the human immune system fight bacterial infections?",
]

# Must match the template used in train.py
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base model vs. LoRA fine-tuned model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model from HuggingFace hub",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="output/lora-adapter",
        help="Path to saved LoRA adapter (produced by train.py)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation/results.json",
        help="Where to write the JSON results file",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Load in full fp16 instead of 4-bit (requires more VRAM)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-padding is standard for generation
    return tokenizer


def load_base_model(model_name: str, use_4bit: bool) -> AutoModelForCausalLM:
    logger.info(f"Loading base model: {model_name}  (4-bit={use_4bit})")
    kwargs = dict(device_map="auto", trust_remote_code=True)
    if use_4bit:
        kwargs["quantization_config"] = get_bnb_config()
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def attach_lora_adapter(base_model: AutoModelForCausalLM, adapter_path: str) -> PeftModel:
    """Load the LoRA adapter on top of the already-loaded base model."""
    logger.info(f"Attaching LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate(
    model,
    tokenizer: AutoTokenizer,
    instruction: str,
    max_new_tokens: int,
) -> str:
    """
    Run greedy decoding (deterministic) for one instruction.
    Returns only the newly generated text (not the prompt).
    """
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                        # greedy → reproducible
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Slice off the prompt tokens
    generated_ids = output_ids[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def batch_generate(
    model,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    label: str,
) -> list[str]:
    outputs = []
    for i, prompt in enumerate(prompts):
        response = generate(model, tokenizer, prompt, max_new_tokens)
        logger.info(f"[{label}] ({i+1}/{len(prompts)}) {prompt[:50]}…")
        outputs.append(response)
    return outputs


# ---------------------------------------------------------------------------
# ROUGE-L metric
# ---------------------------------------------------------------------------
def rouge_l(prediction: str, reference: str) -> float:
    """
    ROUGE-L F1 between prediction and reference.
    When a ground-truth reference is unavailable we score the fine-tuned
    output against the base model output to measure divergence.
    """
    scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return round(score["rougeL"].fmeasure, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    use_4bit = not args.no_4bit
    tokenizer = load_tokenizer(args.model_name)

    # ── 1. Base model responses ───────────────────────────────────────────────
    base_model = load_base_model(args.model_name, use_4bit)
    logger.info("Generating base model responses…")
    base_outputs = batch_generate(
        base_model, tokenizer, TEST_PROMPTS, args.max_new_tokens, "BASE"
    )

    # ── 2. Fine-tuned model responses ─────────────────────────────────────────
    # We reuse the already-loaded base_model weights and attach the adapter on
    # top — no need to reload 7B parameters from disk.
    finetuned_model = attach_lora_adapter(base_model, args.adapter_path)
    logger.info("Generating fine-tuned model responses…")
    ft_outputs = batch_generate(
        finetuned_model, tokenizer, TEST_PROMPTS, args.max_new_tokens, "FT"
    )

    # ── 3. Score & collect results ────────────────────────────────────────────
    results: list[dict] = []
    scores: list[float] = []

    for prompt, base_out, ft_out in zip(TEST_PROMPTS, base_outputs, ft_outputs):
        # ROUGE-L of fine-tuned vs. base output.
        # A lower score means the FT model diverges (personalises) more;
        # a higher score means similar wording.  Both are informative.
        score = rouge_l(prediction=ft_out, reference=base_out)
        scores.append(score)
        results.append(
            {
                "prompt": prompt,
                "base_output": base_out,
                "finetuned_output": ft_out,
                "rouge_l_vs_base": score,
            }
        )

    avg_rouge = round(sum(scores) / len(scores), 4)

    summary = {
        "model": args.model_name,
        "adapter": args.adapter_path,
        "metric": "ROUGE-L (fine-tuned vs. base output)",
        "num_prompts": len(TEST_PROMPTS),
        "average_rouge_l": avg_rouge,
        "results": results,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved → {args.output_path}")

    # ── 4. Print human-readable summary ──────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print("EVALUATION SUMMARY")
    print(sep)
    print(f"Base model   : {args.model_name}")
    print(f"Adapter      : {args.adapter_path}")
    print(f"Prompts      : {len(TEST_PROMPTS)}")
    print(f"Avg ROUGE-L  : {avg_rouge}  (FT vs. base)")
    print(sep)
    print("Sample comparison — first 3 prompts:")
    for r in results[:3]:
        print(f"\n  Prompt  : {r['prompt']}")
        print(f"  Base    : {r['base_output'][:120]}")
        print(f"  FT      : {r['finetuned_output'][:120]}")
        print(f"  Score   : {r['rouge_l_vs_base']}")
    print(f"\n{sep}")
    print(
        "NOTE: ROUGE-L here measures output divergence from the base model.\n"
        "  A lower value = the fine-tuned model responds differently (expected).\n"
        "  For absolute quality, compare against ground-truth references or use\n"
        "  a human / LLM-as-a-judge evaluation."
    )
    print(sep)


if __name__ == "__main__":
    main()
