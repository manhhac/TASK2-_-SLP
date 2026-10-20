#!/usr/bin/env python
"""Huấn luyện Whisper ASR. Ví dụ:
   python -m src.train.train_asr --config configs/baseline.yaml --seed 42
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            {"input_features": [f["input_features"] for f in features]},
            return_tensors="pt",
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": [f["labels"] for f in features]},
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"]
        # Replace padding token id with -100 to ignore loss
        labels = labels.masked_fill(labels.eq(self.processor.tokenizer.pad_token_id), -100)
        batch["labels"] = labels
        return batch
import evaluate

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.models.asr_whisper import build_whisper_asr
from src.data.augment import augment_waveform


class ManifestASRDataset(Dataset):
    def __init__(self, manifest, processor, cfg, train=False, label_field="text_vi"):
        self.rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
        self.proc = processor
        self.cfg = cfg
        self.train = train
        self.label_field = label_field

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        y, _ = librosa.load(r["audio_path"], sr=16000, mono=True)
        if self.train and self.cfg.train.use_augment:
            y = augment_waveform(torch.from_numpy(y), 16000).numpy()
        feat = self.proc.feature_extractor(y, sampling_rate=16000).input_features[0]
        labels = self.proc.tokenizer(r[self.label_field]).input_ids
        return {"input_features": feat, "labels": labels}


def make_compute_metrics(processor, task="transcribe"):
    if task == "translate":
        bleu_metric = evaluate.load("sacrebleu")
        chrf_metric = evaluate.load("chrf")

        def compute(pred):
            pred_ids = pred.predictions
            if isinstance(pred_ids, tuple):
                pred_ids = pred_ids[0]
            pred_ids = np.where(pred_ids == -100, processor.tokenizer.pad_token_id, pred_ids)
            label_ids = pred.label_ids
            label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
            preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            labels = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            bleu = bleu_metric.compute(predictions=preds, references=[[l] for l in labels])
            chrf = chrf_metric.compute(predictions=preds, references=[[l] for l in labels], word_order=2)
            return {
                "bleu": bleu["score"],
                "chrf++": chrf["score"],
            }
        return compute
    else:
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")

        def compute(pred):
            pred_ids = pred.predictions
            if isinstance(pred_ids, tuple):
                pred_ids = pred_ids[0]
            pred_ids = np.where(pred_ids == -100, processor.tokenizer.pad_token_id, pred_ids)
            label_ids = pred.label_ids
            label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
            preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            labels = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            return {
                "wer": 100 * wer_metric.compute(predictions=preds, references=labels),
                "cer": 100 * cer_metric.compute(predictions=preds, references=labels),
            }
        return compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--asr_only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    out = Path(cfg.train.output_dir) / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    # Determine task from config (default to transcribe)
    task = getattr(cfg.model, "task", "transcribe")
    language = getattr(cfg.model, "language", "vi")

    model, processor = build_whisper_asr(
        cfg.model.asr_model,
        language=language,
        task=task,
        use_lora=cfg.model.use_lora,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        lora_target_modules=cfg.model.lora_target_modules,
    )

    # Determine label field based on task
    label_field = "text_en" if task == "translate" else "text_vi"

    # Use quick dataset if specified
    use_quick = getattr(cfg.train, "use_quick_dataset", False)
    train_manifest = f"{cfg.data.manifest_dir}/train_quick.jsonl" if use_quick else f"{cfg.data.manifest_dir}/train.jsonl"
    dev_manifest = f"{cfg.data.manifest_dir}/dev_quick.jsonl" if use_quick else f"{cfg.data.manifest_dir}/dev.jsonl"

    train_ds = ManifestASRDataset(
        train_manifest,
        processor,
        cfg,
        train=True,
        label_field=label_field
    )
    dev_ds = ManifestASRDataset(
        dev_manifest,
        processor,
        cfg,
        train=False,
        label_field=label_field
    )
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Calculate warmup steps
    total_steps = len(train_ds) // cfg.train.batch_size * cfg.train.num_epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)

    # Set metric based on task
    metric_for_best_model = "bleu" if task == "translate" else "wer"
    greater_is_better = True if task == "translate" else False

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr,
        warmup_steps=warmup_steps,
        weight_decay=cfg.train.weight_decay,
        num_train_epochs=cfg.train.num_epochs,
        fp16=cfg.train.fp16,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_steps=cfg.train.save_steps,
        logging_steps=cfg.train.logging_steps,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        report_to=["none"],
        seed=args.seed,
        data_seed=args.seed,
    )

    # Override generation config for evaluation
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor, task),
        processing_class=processor.feature_extractor,
    )
    trainer.train()
    trainer.save_model(str(out / "best"))
    processor.save_pretrained(str(out / "best"))


if __name__ == "__main__":
    main()
