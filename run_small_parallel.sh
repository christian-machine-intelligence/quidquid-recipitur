#!/bin/bash
# ICMI-008 Scaling Curve: Parallel across GPUs
# =============================================
# 5 models on 5 GPUs simultaneously, each running vanilla then psalm.
#
# GPU 0: Qwen2.5-0.5B-Instruct (bf16, ~1GB)
# GPU 1: Qwen2.5-1.5B-Instruct (bf16, ~3GB)
# GPU 2: Qwen2.5-3B-Instruct   (bf16, ~6GB)
# GPU 3: Qwen2.5-7B-Instruct   (bf16, ~14GB)
# GPU 4: Qwen2.5-14B-Instruct  (4bit, ~10GB)
# GPU 5: (free)
#
# Each GPU runs both conditions sequentially (vanilla then psalm).
# All 5 GPUs run in parallel. Estimated ~2-4 hours total.
#
# Usage:
#   nohup bash run_small_parallel.sh > /tmp/psalm_scale_v2_small.log 2>&1 &

set -e

cd /home/tiny/psalm-scale-v2
source ~/ml-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--runner hf-local --variant ratio --runs 5 --temperature 0.7 --seed 42 --detailed --concurrency 1 --batch-size 8 --limit 100"

mkdir -p results

echo "=========================================="
echo "ICMI-008 Scaling Curve — Parallel GPU Launch"
echo "Started: $(date)"
echo "=========================================="

# GPU 0: 0.5B
(
  export CUDA_VISIBLE_DEVICES=0
  echo "[GPU 0] 0.5B Vanilla — $(date)"
  virtue-bench run --model Qwen/Qwen2.5-0.5B-Instruct $COMMON --output qwen05b_vanilla
  echo "[GPU 0] 0.5B Vanilla DONE — $(date)"
  echo "[GPU 0] 0.5B Psalm — $(date)"
  virtue-bench run --model Qwen/Qwen2.5-0.5B-Instruct $COMMON --psalm-set random_baseline --output qwen05b_psalm
  echo "[GPU 0] 0.5B Psalm DONE — $(date)"
) > /tmp/psalm_gpu0.log 2>&1 &
PID0=$!

# GPU 1: 1.5B
(
  export CUDA_VISIBLE_DEVICES=1
  echo "[GPU 1] 1.5B Vanilla — $(date)"
  virtue-bench run --model Qwen/Qwen2.5-1.5B-Instruct $COMMON --output qwen15b_vanilla
  echo "[GPU 1] 1.5B Vanilla DONE — $(date)"
  echo "[GPU 1] 1.5B Psalm — $(date)"
  virtue-bench run --model Qwen/Qwen2.5-1.5B-Instruct $COMMON --psalm-set random_baseline --output qwen15b_psalm
  echo "[GPU 1] 1.5B Psalm DONE — $(date)"
) > /tmp/psalm_gpu1.log 2>&1 &
PID1=$!

# GPU 2: 3B
(
  export CUDA_VISIBLE_DEVICES=2
  echo "[GPU 2] 3B Vanilla — $(date)"
  virtue-bench run --model Qwen/Qwen2.5-3B-Instruct $COMMON --output qwen3b_vanilla
  echo "[GPU 2] 3B Vanilla DONE — $(date)"
  echo "[GPU 2] 3B Psalm — $(date)"
  virtue-bench run --model Qwen/Qwen2.5-3B-Instruct $COMMON --psalm-set random_baseline --output qwen3b_psalm
  echo "[GPU 2] 3B Psalm DONE — $(date)"
) > /tmp/psalm_gpu2.log 2>&1 &
PID2=$!

# GPU 3: 7B
(
  export CUDA_VISIBLE_DEVICES=3
  echo "[GPU 3] 7B Vanilla — $(date)"
  virtue-bench run --model Qwen/Qwen2.5-7B-Instruct $COMMON --output qwen7b_vanilla
  echo "[GPU 3] 7B Vanilla DONE — $(date)"
  echo "[GPU 3] 7B Psalm — $(date)"
  virtue-bench run --model Qwen/Qwen2.5-7B-Instruct $COMMON --psalm-set random_baseline --output qwen7b_psalm
  echo "[GPU 3] 7B Psalm DONE — $(date)"
) > /tmp/psalm_gpu3.log 2>&1 &
PID3=$!

# GPU 4: 14B (4bit quantization — 14B bf16 won't fit on single 4090)
(
  export CUDA_VISIBLE_DEVICES=4
  echo "[GPU 4] 14B Vanilla — $(date)"
  virtue-bench run --model Qwen/Qwen2.5-14B-Instruct --hf-quantize 4bit $COMMON --output qwen14b_vanilla
  echo "[GPU 4] 14B Vanilla DONE — $(date)"
  echo "[GPU 4] 14B Psalm — $(date)"
  virtue-bench run --model Qwen/Qwen2.5-14B-Instruct --hf-quantize 4bit $COMMON --psalm-set random_baseline --output qwen14b_psalm
  echo "[GPU 4] 14B Psalm DONE — $(date)"
) > /tmp/psalm_gpu4.log 2>&1 &
PID4=$!

echo "Launched 5 parallel GPU jobs:"
echo "  GPU 0 (0.5B):  PID $PID0"
echo "  GPU 1 (1.5B):  PID $PID1"
echo "  GPU 2 (3B):    PID $PID2"
echo "  GPU 3 (7B):    PID $PID3"
echo "  GPU 4 (14B):   PID $PID4"
echo ""
echo "Monitor with: tail -5 /tmp/psalm_gpu{0,1,2,3,4}.log"
echo ""

# Wait for all to finish
wait $PID0 $PID1 $PID2 $PID3 $PID4

echo ""
echo "=========================================="
echo "ALL SMALL MODELS COMPLETE — $(date)"
echo "=========================================="
echo "Results:"
ls -la results/qwen*b_*.json 2>/dev/null || echo "(no results yet)"
