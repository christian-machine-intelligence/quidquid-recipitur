#!/bin/bash
# ICMI-008 Replication: Wikipedia Control for 32B and 72B
# ========================================================
# Adds the length-matched Wikipedia prose control condition
# that was omitted from the main study. This allows us to
# distinguish psalm-specific effects from generic
# context-length effects at the scales where the psalm
# effect is statistically significant (32B and 72B).
#
# 2 models × 1 condition × 5 runs × 100 scenarios × 4 virtues
# = 4,000 evaluations total
#
# Estimated runtime: ~6 hours (72B: ~4h, 32B: ~2h, sequential)
#
# Usage (on Marx):
#   nohup bash run_controls.sh > /tmp/psalm_scale_v2_controls.log 2>&1 &

set -e

cd /home/tiny/psalm-scale-v2
source ~/ml-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--runner hf-local --variant ratio --runs 5 --temperature 0.7 --seed 42 --detailed --concurrency 1 --batch-size 8 --limit 100 --hf-quantize 4bit"

mkdir -p results

echo "=========================================="
echo "Wikipedia Control Runs — 32B and 72B"
echo "Started: $(date)"
echo "=========================================="

echo ""
echo "[1/2] 72B Wikipedia Control — $(date)"
virtue-bench run --model Qwen/Qwen2.5-72B-Instruct $COMMON \
    --inject data/wikipedia_control.txt \
    --output qwen72b_control
echo "[1/2] DONE — $(date)"

echo ""
echo "[2/2] 32B Wikipedia Control — $(date)"
virtue-bench run --model Qwen/Qwen2.5-32B-Instruct $COMMON \
    --inject data/wikipedia_control.txt \
    --output qwen32b_control
echo "[2/2] DONE — $(date)"

echo ""
echo "=========================================="
echo "CONTROL RUNS COMPLETE — $(date)"
echo "=========================================="
ls -la /home/tiny/virtue-bench-2/results/qwen{32,72}b_control*.json
