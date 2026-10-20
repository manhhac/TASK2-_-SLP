#!/usr/bin/env python
"""Đánh giá đầy đủ: ASR (WER/CER) + MT (BLEU/chrF++) + RTF.
   python -m src.eval.evaluate --asr_dir outputs/baseline/seed_42/best --manifest data/manifests/test.jsonl
"""
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
    ap.add_argument("--mt_model", default="Helsinki-NLP/opus-mt-vi-en")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    if args.limit > 0:
        rows = rows[:args.limit]

    wavs, ref_vi, ref_en = [], [], []
    for r in rows:
        y, _ = librosa.load(r["audio_path"], sr=16000, mono=True)
        wavs.append(y)
        ref_vi.append(r["text_vi"])
        ref_en.append(r["text_en"])

    pipe = CascadeASRMT(
        args.asr_dir,
        mt_name=args.mt_model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    t0 = time.time()
    hyp_vi, hyp_en = pipe.forward(wavs, asr_beams=1, mt_beams=4)
    total_time = time.time() - t0

    asr = compute_asr_metrics(hyp_vi, ref_vi)
    mt_cascade = compute_mt_metrics(hyp_en, ref_en)

    # MT oracle: dịch từ transcript chuẩn (để đo lỗi lan truyền)
    hyp_en_oracle = pipe.translate(ref_vi, num_beams=4)
    mt_oracle = compute_mt_metrics(hyp_en_oracle, ref_en)

    total_audio = sum(len(w) for w in wavs) / 16000
    result = {
        "n": len(rows),
        "asr": asr,
        "mt_cascade": mt_cascade,
        "mt_oracle_from_ref": mt_oracle,
        "rtf": total_time / max(total_audio, 1e-6),
        "total_seconds": total_time,
        "audio_seconds": total_audio,
    }

    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
