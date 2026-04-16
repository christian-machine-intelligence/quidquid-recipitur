#!/bin/bash
# ICMI-008 Replication: Parable of the Sower (VirtueBench V2)
# ============================================================
# Tests whether Qwen2.5-72B-Instruct responds to psalm injection
# while Qwen2.5-32B-Instruct does not.
#
# 2 models × 2 conditions × 5 runs × 100 scenarios × 4 virtues
# = 8,000 evaluations total
#
# With batch_size=8:
#   72B (4-bit, 6×4090): ~2-3x faster → ~6-8h/condition → ~14h
#   32B (4-bit, 6×4090): ~2-3x faster → ~3-5h/condition → ~8h
#   Grand total: ~1 day
#
# Usage:
#   nohup bash run_study.sh > /tmp/psalm_scale_v2.log 2>&1 &

set -e

cd /home/tiny/psalm-scale-v2
source ~/ml-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--runner hf-local --variant ratio --runs 5 --temperature 0.7 --seed 42 --detailed --concurrency 1 --batch-size 8 --limit 100"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "ICMI-008 Replication — Parable of the Sower (V2)"
echo "Started: $(date)"
echo "=========================================="

# --- Phase 1: Qwen2.5-72B-Instruct (4-bit NF4) ---

MODEL_72B="Qwen/Qwen2.5-72B-Instruct"
QUANT_72B="--hf-quantize 4bit"

echo ""
echo "[1/4] 72B Vanilla — $(date)"
virtue-bench run --model "$MODEL_72B" $QUANT_72B $COMMON \
    --output qwen72b_vanilla
echo "[1/4] DONE — $(date)"

echo ""
echo "[2/4] 72B Psalm (random_baseline) — $(date)"
virtue-bench run --model "$MODEL_72B" $QUANT_72B $COMMON \
    --psalm-set random_baseline \
    --output qwen72b_psalm
echo "[2/4] DONE — $(date)"

# --- Phase 2: Qwen2.5-32B-Instruct (4-bit NF4) ---

MODEL_32B="Qwen/Qwen2.5-32B-Instruct"
QUANT_32B="--hf-quantize 4bit"

echo ""
echo "[3/4] 32B Vanilla — $(date)"
virtue-bench run --model "$MODEL_32B" $QUANT_32B $COMMON \
    --output qwen32b_vanilla
echo "[3/4] DONE — $(date)"

echo ""
echo "[4/4] 32B Psalm (random_baseline) — $(date)"
virtue-bench run --model "$MODEL_32B" $QUANT_32B $COMMON \
    --psalm-set random_baseline \
    --output qwen32b_psalm
echo "[4/4] DONE — $(date)"

echo ""
echo "=========================================="
echo "ALL CONDITIONS COMPLETE — $(date)"
echo "=========================================="
echo "Results in: $(pwd)/results/"
ls -la results/
