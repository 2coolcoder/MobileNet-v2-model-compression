#!/usr/bin/env bash
# Full reproduction of every number in REPORT.md, start to finish.
# Total wall-clock on one RTX 6000 Ada: about 2 hours.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "== [1/5] Train the FP32 baseline (Q1) -- ~30 min"
python train.py --epochs 200 --batch_size 128 --lr 0.1 --seed 42 --wandb \
                --run_name baseline-mobilenetv2-200ep

echo "== [2/5] Verify the compression pipeline (Q2)"
python scripts/check_commands.py
python analyze.py --self_test
python analyze.py --ablation

echo "== [3/5] Per-layer pruning sensitivity (Q2a) -- ~5 min"
python analyze.py --sensitivity --target_sparsity 0.6 --tolerance 1.0

echo "== [4/5] Compression sweep + W&B parallel coordinates (Q3) -- ~7 min + upload"
python sweep.py
python upload_sweep.py

echo "== [5/5] Final operating point with QAT (Q4) -- ~10 min"
python analyze.py --final --wandb \
    --weight_bits 4 --act_bits 8 --group_size 0 --scale_bits 8 \
    --prune_ratio 0.65 --qat_epochs 12 --sparsity_map none --layer_table 20

python export.py
python analyze.py --figures
echo "== done: see results/ and results/figures/"
