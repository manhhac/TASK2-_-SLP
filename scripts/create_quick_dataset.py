#!/usr/bin/env python
"""Tạo tập con nhỏ từ VIVOS với EN translations từ MT model."""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.mt_model import build_mt_model, translate_vi_en
import torch

def create_quick_dataset(n_samples=100):
    """Tạo tập con nhỏ với EN translations."""
    # Load manifests
    train_path = Path("data/manifests/train.jsonl")
    dev_path = Path("data/manifests/dev.jsonl")
    test_path = Path("data/manifests/test.jsonl")

    with open(train_path, "r", encoding="utf-8") as f:
        train = [json.loads(l) for l in f]
    with open(dev_path, "r", encoding="utf-8") as f:
        dev = [json.loads(l) for l in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test = [json.loads(l) for l in f]

    # Lấy n_samples từ mỗi split
    train_subset = train[:n_samples]
    dev_subset = dev[:min(n_samples // 5, len(dev))]
    test_subset = test[:min(n_samples // 5, len(test))]

    # Load MT model
    print("Loading MT model...")
    mt_model, tokenizer = build_mt_model("Helsinki-NLP/opus-mt-vi-en")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mt_model = mt_model.to(device).eval()

    # Dịch text_vi sang text_en
    def translate_batch(samples):
        vi_texts = [s["text_vi"] for s in samples]
        en_texts = translate_vi_en(vi_texts, mt_model, tokenizer, device=device, num_beams=4)
        for s, en in zip(samples, en_texts):
            s["text_en"] = en
        return samples

    print("Translating train...")
    train_subset = translate_batch(train_subset)
    print("Translating dev...")
    dev_subset = translate_batch(dev_subset)
    print("Translating test...")
    test_subset = translate_batch(test_subset)

    # Save quick manifests
    quick_dir = Path("data/manifests")
    for split, samples in [("train", train_subset), ("dev", dev_subset), ("test", test_subset)]:
        path = quick_dir / f"{split}_quick.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Saved {len(samples)} samples to {path}")

    print(f"\nTotal samples: {len(train_subset) + len(dev_subset) + len(test_subset)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=100)
    args = ap.parse_args()
    create_quick_dataset(args.n_samples)
