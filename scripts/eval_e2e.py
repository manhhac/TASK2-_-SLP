#!/usr/bin/env python
"""Đánh giá Whisper E2E Translation (task=translate).
   python scripts/eval_e2e.py --model_dir outputs/v4_quick/seed_42/best --manifest data/manifests/test_quick.jsonl --out outputs/v4_quick/seed_42/test_metrics_e2e.json
"""
import argparse
import json
import time
import sys
from pathlib import Path

import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import compute_mt_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to fine-tuned Whisper model")
    ap.add_argument("--manifest", required=True, help="Path to test manifest")
    ap.add_argument("--out", required=True, help="Output JSON file")
    ap.add_argument("--limit", type=int, default=-1, help="Limit samples for quick test")
    args = ap.parse_args()

    print(f"Loading model from: {args.model_dir}")

    # Load fine-tuned Whisper
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)

    # Set for translation task
    model.generation_config.language = "en"
    model.generation_config.task = "translate"
    model.generation_config.forced_decoder_ids = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    # Load manifest
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    if args.limit > 0:
        rows = rows[:args.limit]
        print(f"Running on {len(rows)} samples (limited)")

    print(f"\nEvaluating on {len(rows)} samples...\n")

    wavs, ref_en = [], []
    for r in rows:
        y, _ = librosa.load(r["audio_path"], sr=16000, mono=True)
        wavs.append(y)
        ref_en.append(r["text_en"])

    # E2E translation inference
    t0 = time.time()
    hyp_en = []
    for i, wav in enumerate(wavs):
        inputs = processor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(inputs, max_new_tokens=440)
        translation = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        hyp_en.append(translation)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(wavs)} samples")

    total_time = time.time() - t0

    # Compute metrics
    mt_metrics = compute_mt_metrics(hyp_en, ref_en)

    total_audio = sum(len(w) for w in wavs) / 16000

    result = {
        "n": len(rows),
        "mt_e2e": mt_metrics,
        "rtf": total_time / max(total_audio, 1e-6),
        "total_seconds": total_time,
        "audio_seconds": total_audio
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "="*60)
    print("E2E TRANSLATION RESULTS")
    print("="*60)
    print(f"BLEU: {mt_metrics['bleu']:.4f}")
    print(f"chrF++: {mt_metrics['chrf++']:.4f}")
    print(f"RTF: {result['rtf']:.4f}")
    print("="*60)
    print(f"\nFull results saved to: {args.out}")


if __name__ == "__main__":
    main()
