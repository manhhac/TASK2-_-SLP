#!/usr/bin/env bash
set -e
# Chạy tất cả các phiên bản experiment

echo "Running all versions..."

# Baseline
echo "=== Running baseline ==="
python -m src.train.run_version --version baseline

# V1: Augment
echo "=== Running V1 (augment) ==="
python -m src.train.run_version --version v1_augment

# V2: LoRA + beam
echo "=== Running V2 (LoRA + beam) ==="
python -m src.train.run_version --version v2_lora

# V3: Noise-robust MT
echo "=== Running V3 (noise-robust MT) ==="
python -m src.train.run_version --version v3_noise_robust_mt

# V4: End-to-end
echo "=== Running V4 (end-to-end) ==="
python -m src.train.run_version --version v4_e2e

echo "All versions completed!"
