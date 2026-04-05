import argparse
import json
import logging
import os

import torch
from peft import PeftModel
from rouge_score import rouge_scorer as rs_module
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

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

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--adapter_path", type=str, default="output/lora-adapter")
    parser.add_argument("--output_path", type=str, default="evaluation/results.json")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


def check_adapter_exists(adapter_path):
    if not os.path.isdir(adapter_path):
        raise ValueError(f"Adapter not found: {adapter_path}\nRun train.py first.")
    if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        raise ValueError(f"adapter_config.json missing in {adapter_path}\nRun train.py first.")
    logger.info(f"Adapter found: {adapter_path}")


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(model_name, use_4bit):
    logger.info(f"Loading base model: {model_name}")
    kwargs = dict(device_map="auto", trust_remote_code=True)
    if use_4bit:
        kwargs["quantization_config"] = get_bnb_config()
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def attach_lora_adapter(base_model, adapter_path):
    abs_path = os.path.abspath(adapter_path)
    logger.info(f"Loading LoRA adapter from: {abs_path}")
    model = PeftModel.from_pretrained(base_model, abs_path, local_files_only=True)
    model.eval()
    logger.info("Adapter loaded.")
    return model


@torch.inference_mode()
def generate(model, tokenizer, instruction, max_new_tokens):
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def batch_generate(model, tokenizer, prompts, max_new_tokens, label):
    outputs = []
    for i, prompt in enumerate(prompts):
        response = generate(model, tokenizer, prompt, max_new_tokens)
        logger.info(f"[{label}] ({i+1}/{len(prompts)}) {prompt[:60]}")
        outputs.append(response)
    return outputs


def rouge_l(prediction, reference):
    scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return round(score["rougeL"].fmeasure, 4)


def main():
    args = parse_args()

    check_adapter_exists(args.adapter_path)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    tokenizer = load_tokenizer(args.model_name)
    base_model = load_base_model(args.model_name, not args.no_4bit)

    logger.info("Generating base model responses...")
    base_outputs = batch_generate(base_model, tokenizer, TEST_PROMPTS, args.max_new_tokens, "BASE")

    finetuned_model = attach_lora_adapter(base_model, args.adapter_path)
    logger.info("Generating fine-tuned model responses...")
    ft_outputs = batch_generate(finetuned_model, tokenizer, TEST_PROMPTS, args.max_new_tokens, "FT")

    results = []
    scores = []
    for prompt, base_out, ft_out in zip(TEST_PROMPTS, base_outputs, ft_outputs):
        score = rouge_l(ft_out, base_out)
        scores.append(score)
        results.append({
            "prompt": prompt,
            "base_output": base_out,
            "finetuned_output": ft_out,
            "rouge_l_vs_base": score,
        })

    avg_rouge = round(sum(scores) / len(scores), 4)

    summary = {
        "model": args.model_name,
        "adapter": os.path.abspath(args.adapter_path),
        "metric": "ROUGE-L (fine-tuned vs. base output)",
        "num_prompts": len(TEST_PROMPTS),
        "average_rouge_l": avg_rouge,
        "results": results,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved -> {args.output_path}")

    sep = "=" * 60
    print(f"\n{sep}")
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print(sep)
    print(f"Модель    : {args.model_name}")
    print(f"Адаптер   : {os.path.abspath(args.adapter_path)}")
    print(f"Промптов  : {len(TEST_PROMPTS)}")
    print(f"ROUGE-L   : {avg_rouge}")
    print(sep)
    print("Примеры (первые 3):")
    for r in results[:3]:
        print(f"\n  Вопрос : {r['prompt']}")
        print(f"  Base   : {r['base_output'][:120]}")
        print(f"  FT     : {r['finetuned_output'][:120]}")
        print(f"  Score  : {r['rouge_l_vs_base']}")
    print(f"\n{sep}")


if __name__ == "__main__":
    main()
