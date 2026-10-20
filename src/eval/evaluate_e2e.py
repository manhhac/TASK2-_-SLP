#!/usr/bin/env python
"""Đánh giá V4: Whisper end-to-end translation (VI→EN trực tiếp)."""
import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
import librosa

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.cascade import CascadeASRMT
from src.eval.metrics import compute_asr_metrics, compute_mt_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]

    wavs, ref_vi, ref_en = [], [], []
    for r in rows:
        y, _ = librosa.load(r["audio_path"], sr=16000, mono=True)
        wavs.append(y)
        ref_vi.append(r["text_vi"])
        ref_en.append(r["text_en"])

    pipe = CascadeASRMT(
        args.asr_dir,
        mt_name="Helsinki-NLP/opus-mt-vi-en",  # Won't be used for E2E
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    t0 = time.time()
    # V4: end-to-end translation directly
    hyp_en = pipe.transcribe(wavs, batch_size=8, num_beams=1, task="translate", language="en")
    total_time = time.time() - t0

    # Compute metrics
    mt_e2e = compute_mt_metrics(hyp_en, ref_en)

    total_audio = sum(len(w) for w in wavs) / 16000
    result = {
        "n": len(rows),
        "mt_e2e": mt_e2e,
        "rtf": total_time / max(total_audio, 1e-6),
        "total_seconds": total_time,
        "audio_seconds": total_audio,
    }

    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
