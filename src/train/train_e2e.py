#!/usr/bin/env python
"""Huấn luyện Whisper E2E translation (VI -> EN)."""
import argparse
import json
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
)
import evaluate

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.models.asr_whisper import build_whisper_asr
from src.data.augment import augment_waveform


class ManifestE2EDataset(Dataset):
    def __init__(self, manifest, processor, cfg, train=False):
        self.rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
        self.proc = processor
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        y, _ = librosa.load(r["audio_path"], sr=16000, mono=True)
        if self.train and self.cfg.train.use_augment:
            y = augment_waveform(torch.from_numpy(y), 16000).numpy()
        feat = self.proc.feature_extractor(y, sampling_rate=16000).input_features[0]
        # Use English text for translation task
        labels = self.proc.tokenizer(r["text_en"]).input_ids
        return {"input_features": feat, "labels": labels}


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
        labels = labels.masked_fill(labels.eq(self.processor.tokenizer.pad_token_id), -100)
        batch["labels"] = labels
        return batch


def make_compute_metrics(processor):
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
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    out = Path(cfg.train.output_dir) / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    # Build model with task=translate for E2E
    model, processor = build_whisper_asr(
        cfg.model.asr_model,
        language="vi",
        task="translate",  # E2E translation
        use_lora=cfg.model.use_lora,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        lora_target_modules=cfg.model.lora_target_modules,
    )

    train_ds = ManifestE2EDataset(
        f"{cfg.data.manifest_dir}/train.jsonl",
        processor,
        cfg,
        train=True
    )
    dev_ds = ManifestE2EDataset(
        f"{cfg.data.manifest_dir}/dev.jsonl",
        processor,
        cfg,
        train=False
    )
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    total_steps = len(train_ds) // cfg.train.batch_size * cfg.train.num_epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)

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
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["none"],
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )
    trainer.train()
    trainer.save_model(str(out / "best"))
    processor.save_pretrained(str(out / "best"))


if __name__ == "__main__":
    main()
