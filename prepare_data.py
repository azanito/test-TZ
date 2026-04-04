#!/usr/bin/env python3
"""
prepare_data.py — Download, filter, and save an instruction-response dataset.

Supports:
  - tatsu-lab/alpaca
  - databricks/databricks-dolly-15k

Usage:
    python prepare_data.py --dataset tatsu-lab/alpaca --max_samples 500
"""

import argparse
import json
import logging
import os

from datasets import load_dataset

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
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare instruction-response dataset for fine-tuning"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        choices=["tatsu-lab/alpaca", "databricks/databricks-dolly-15k"],
        help="HuggingFace dataset to use (default: tatsu-lab/alpaca)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/dataset.jsonl",
        help="Output path for processed JSONL dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of samples to keep after filtering (0 = no limit)",
    )
    parser.add_argument(
        "--min_instruction_len",
        type=int,
        default=10,
        help="Minimum number of characters for instruction field",
    )
    parser.add_argument(
        "--min_response_len",
        type=int,
        default=20,
        help="Minimum number of characters for response field",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading & normalisation
# ---------------------------------------------------------------------------
def load_and_normalise(dataset_name: str) -> list[dict]:
    """
    Download a HuggingFace dataset and normalise every row to:
        {"instruction": str, "response": str}
    """
    logger.info(f"Downloading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    logger.info(f"Raw rows loaded: {len(ds)}")

    samples: list[dict] = []

    if dataset_name == "tatsu-lab/alpaca":
        for row in ds:
            instruction = (row.get("instruction") or "").strip()
            # Alpaca has an optional 'input' that contextualises the instruction
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


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def filter_samples(
    samples: list[dict],
    min_instruction_len: int,
    min_response_len: int,
) -> list[dict]:
    """
    Apply quality filters:
      1. Trim whitespace
      2. Remove samples with instruction or response that is too short
      3. Deduplicate by normalised instruction text
    """
    before = len(samples)
    logger.info(f"Samples before filtering: {before}")

    # Step 1 — trim whitespace
    for s in samples:
        s["instruction"] = s["instruction"].strip()
        s["response"] = s["response"].strip()

    # Step 2 — length filter
    samples = [
        s
        for s in samples
        if len(s["instruction"]) >= min_instruction_len
        and len(s["response"]) >= min_response_len
    ]
    after_length = len(samples)
    logger.info(
        f"After length filter : {after_length} "
        f"(removed {before - after_length} short samples)"
    )

    # Step 3 — deduplication (case-insensitive on instruction)
    seen: set[str] = set()
    unique: list[dict] = []
    for s in samples:
        key = s["instruction"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    logger.info(
        f"After deduplication : {len(unique)} "
        f"(removed {after_length - len(unique)} duplicates)"
    )
    return unique


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_jsonl(samples: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples)} samples → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    samples = load_and_normalise(args.dataset)
    samples = filter_samples(samples, args.min_instruction_len, args.min_response_len)

    # Optionally cap the dataset size
    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]
        logger.info(f"Capped to {args.max_samples} samples")

    save_jsonl(samples, args.output_path)
    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
