import argparse
import json
import logging
import os

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca",
                        choices=["tatsu-lab/alpaca", "databricks/databricks-dolly-15k"])
    parser.add_argument("--output_path", type=str, default="data/dataset.jsonl")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--min_instruction_len", type=int, default=10)
    parser.add_argument("--min_response_len", type=int, default=20)
    return parser.parse_args()


def load_and_normalise(dataset_name):
    logger.info(f"Downloading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    logger.info(f"Raw rows loaded: {len(ds)}")

    samples = []

    if dataset_name == "tatsu-lab/alpaca":
        for row in ds:
            instruction = (row.get("instruction") or "").strip()
            extra_input = (row.get("input") or "").strip()
            if extra_input:
                instruction = f"{instruction}\n\n{extra_input}"
            response = (row.get("output") or "").strip()
            samples.append({"instruction": instruction, "response": response})

    elif dataset_name == "databricks/databricks-dolly-15k":
        for row in ds:
            instruction = (row.get("instruction") or "").strip()
            context = (row.get("context") or "").strip()
            if context:
                instruction = f"{instruction}\n\nContext: {context}"
            response = (row.get("response") or "").strip()
            samples.append({"instruction": instruction, "response": response})

    return samples


def filter_samples(samples, min_instruction_len, min_response_len):
    before = len(samples)
    logger.info(f"Samples before filtering: {before}")

    for s in samples:
        s["instruction"] = s["instruction"].strip()
        s["response"] = s["response"].strip()

    samples = [
        s for s in samples
        if len(s["instruction"]) >= min_instruction_len
        and len(s["response"]) >= min_response_len
    ]
    logger.info(f"After length filter: {len(samples)} (removed {before - len(samples)})")

    seen = set()
    unique = []
    for s in samples:
        key = s["instruction"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    logger.info(f"After deduplication: {len(unique)} (removed {len(samples) - len(unique)} duplicates)")
    return unique


def save_jsonl(samples, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples)} samples -> {output_path}")


def main():
    args = parse_args()

    samples = load_and_normalise(args.dataset)
    samples = filter_samples(samples, args.min_instruction_len, args.min_response_len)

    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[:args.max_samples]
        logger.info(f"Capped to {args.max_samples} samples")

    save_jsonl(samples, args.output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
