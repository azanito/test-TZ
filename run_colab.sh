#!/bin/bash
# run_colab.sh — Full pipeline runner for Google Colab T4
#
# Usage in Colab cell:
#   !bash run_colab.sh
#
# Or step-by-step (recommended for monitoring progress):
#   !python prepare_data.py
#   !python train.py --num_train_epochs 1
#   !python evaluate.py

set -e  # exit immediately on any error

echo "=========================================="
echo " QLoRA Fine-Tuning Pipeline"
echo "=========================================="

# ── Step 0: Install dependencies ──────────────────────────────────────────
echo ""
echo "[0/3] Installing dependencies..."
pip install -r requirements.txt -q

# ── Step 1: Prepare dataset ───────────────────────────────────────────────
echo ""
echo "[1/3] Preparing dataset..."
python prepare_data.py \
    --dataset tatsu-lab/alpaca \
    --max_samples 500 \
    --output_path data/dataset.jsonl

# ── Step 2: Fine-tune ─────────────────────────────────────────────────────
echo ""
echo "[2/3] Starting QLoRA fine-tuning (1 epoch, ~30-45 min on T4)..."
python train.py \
    --model_name mistralai/Mistral-7B-v0.1 \
    --data_path data/dataset.jsonl \
    --output_dir output/lora-adapter \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 512

# ── Step 3: Evaluate ──────────────────────────────────────────────────────
echo ""
echo "[3/3] Evaluating base vs. fine-tuned model..."
python evaluate.py \
    --model_name mistralai/Mistral-7B-v0.1 \
    --adapter_path output/lora-adapter \
    --output_path evaluation/results.json \
    --max_new_tokens 150

echo ""
echo "=========================================="
echo " Pipeline complete!"
echo " Results: evaluation/results.json"
echo "=========================================="
