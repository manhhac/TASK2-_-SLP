#!/usr/bin/env bash
set -e
# Giai đoạn 1: dữ liệu
python -m src.data.prepare_data --config configs/baseline.yaml

# Giai đoạn 2: baseline (3 seeds)
for s in 42 123 2024; do
  python -m src.train.train_asr --config configs/baseline.yaml --seed $s
done

# Đánh giá baseline
python -m src.eval.evaluate \
  --asr_dir outputs/baseline/seed_42/best \
  --manifest data/manifests/test.jsonl \
  --out outputs/baseline/seed_42/test_metrics.json

# Error analysis
python -m src.eval.error_analysis \
  --pred outputs/baseline/seed_42/test_predictions.jsonl \
  --out outputs/baseline/seed_42/error_analysis.json
