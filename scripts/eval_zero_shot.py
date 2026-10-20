#!/usr/bin/env python
"""Đánh giá Zero-shot Whisper Pretrained + MarianMT Cascade (không cần train).
   Cách nhanh nhất để có số liệu baseline cho báo cáo.

   python scripts/eval_zero_shot.py --manifest data/manifests/test.jsonl --out outputs/zero_shot_baseline.json
"""
import argparse
import json
import time
import sys
from pathlib import Path

import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import MarianTokenizer, MarianMTModel

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import compute_asr_metrics, compute_mt_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to test manifest")
    ap.add_argument("--out", required=True, help="Output JSON file")
    ap.add_argument("--limit", type=int, default=-1, help="Limit samples for quick test")
    ap.add_argument("--whisper_model", default="openai/whisper-small", help="Whisper model name")
    ap.add_argument("--mt_model", default="Helsinki-NLP/opus-mt-vi-en", help="MT model name")
    args = ap.parse_args()

    print(f"Loading Whisper: {args.whisper_model}")
    print(f"Loading MT: {args.mt_model}")

    # Load Whisper
    processor = WhisperProcessor.from_pretrained(args.whisper_model, language="vi", task="transcribe")
    asr_model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model)
    asr_model.generation_config.language = "vi"
    asr_model.generation_config.task = "transcribe"
    asr_model.generation_config.forced_decoder_ids = None

    # Load MarianMT
    mt_tokenizer = MarianTokenizer.from_pretrained(args.mt_model)
    mt_model = MarianMTModel.from_pretrained(args.mt_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = asr_model.to(device)
    mt_model = mt_model.to(device)
    print(f"Using device: {device}")

    # Load manifest
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    if args.limit > 0:
        rows = rows[:args.limit]
        print(f"Running on {len(rows)} samples (limited)")

    print(f"\nEvaluating on {len(rows)} samples...\n")

    wavs, ref_vi, ref_en = [], [], []
    for r in rows:
        y, _ = librosa.load(r["audio_path"], sr=16000, mono=True)
        wavs.append(y)
        ref_vi.append(r["text_vi"])
        ref_en.append(r["text_en"])

    # ASR inference
    t0 = time.time()
    hyp_vi = []
    for i, wav in enumerate(wavs):
        inputs = processor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(device)
        with torch.no_grad():
            predicted_ids = asr_model.generate(inputs, max_new_tokens=440)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        hyp_vi.append(transcription)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(wavs)} samples")

    asr_time = time.time() - t0

    # MT inference (cascade)
    t0 = time.time()
    hyp_en = []
    for txt in hyp_vi:
        inputs = mt_tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = mt_model.generate(**inputs, max_length=512, num_beams=4)
        hyp_en.append(mt_tokenizer.decode(outputs[0], skip_special_tokens=True))
    mt_time = time.time() - t0

    # MT oracle (dịch từ reference)
    t0 = time.time()
    hyp_en_oracle = []
    for txt in ref_vi:
        inputs = mt_tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = mt_model.generate(**inputs, max_length=512, num_beams=4)
        hyp_en_oracle.append(mt_tokenizer.decode(outputs[0], skip_special_tokens=True))
    mt_oracle_time = time.time() - t0

    # Compute metrics
    asr_metrics = compute_asr_metrics(hyp_vi, ref_vi)
    mt_cascade = compute_mt_metrics(hyp_en, ref_en)
    mt_oracle = compute_mt_metrics(hyp_en_oracle, ref_en)

    total_audio = sum(len(w) for w in wavs) / 16000

    result = {
        "setup": {
            "whisper_model": args.whisper_model,
            "mt_model": args.mt_model,
            "device": device,
            "n_samples": len(rows),
            "zero_shot": True,
            "description": "Whisper pretrained (no fine-tune) + MarianMT cascade"
        },
        "metrics": {
            "asr": asr_metrics,
            "mt_cascade": mt_cascade,
            "mt_oracle_from_ref": mt_oracle,
        },
        "timing": {
            "asr_seconds": asr_time,
            "mt_seconds": mt_time,
            "mt_oracle_seconds": mt_oracle_time,
            "total_seconds": asr_time + mt_time,
            "audio_seconds": total_audio,
            "rtf": (asr_time + mt_time) / max(total_audio, 1e-6),
            "asr_rtf": asr_time / max(total_audio, 1e-6),
            "mt_rtf": mt_time / max(total_audio, 1e-6),
        }
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "="*60)
    print("ZERO-SHOT BASELINE RESULTS")
    print("="*60)
    print(f"ASR (WER): {asr_metrics['wer']:.4f}")
    print(f"ASR (CER): {asr_metrics['cer']:.4f}")
    print(f"MT Cascade (BLEU): {mt_cascade['bleu']:.4f}")
    print(f"MT Cascade (chrF++): {mt_cascade['chrf++']:.4f}")
    print(f"MT Oracle (BLEU): {mt_oracle['bleu']:.4f}")
    print(f"RTF: {result['timing']['rtf']:.4f}")
    print(f"\nError propagation: {mt_oracle['bleu'] - mt_cascade['bleu']:.4f} BLEU")
    print("="*60)
    print(f"\nFull results saved to: {args.out}")


if __name__ == "__main__":
    main()
