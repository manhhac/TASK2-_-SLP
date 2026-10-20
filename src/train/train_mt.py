#!/usr/bin/env python
"""Huấn luyện MarianMT cho VI→EN. Ví dụ:
   python -m src.train.train_mt --config configs/v3_noise_robust_mt.yaml --seed 42

   V3: fine-tune MT trên ASR transcript nhiễu (noise-robust MT).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import evaluate

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.models.mt_model import build_mt_model


class MTDataset(Dataset):
    """Dataset cho MT với pair (vi, en)."""

    def __init__(self, manifest, tokenizer, use_noisy_source=False):
        """
        Args:
            manifest: File manifest chứa text_vi, text_en
            tokenizer: MarianTokenizer
            use_noisy_source: Nếu True, dùng hyp_vi (ASR output) thay vì ref_vi
        """
        self.rows = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.rows.append(row)
        self.tok = tokenizer
        self.use_noisy_source = use_noisy_source

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # Source: tiếng Việt (ref hoặc noisy)
        if self.use_noisy_source and "hyp_vi" in r:
            src = r["hyp_vi"]
        else:
            src = r["text_vi"]
        # Target: tiếng Anh
        tgt = r["text_en"]

        model_inputs = self.tok(
            src,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        labels = self.tok(
            tgt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        model_inputs["labels"] = labels["input_ids"]
        return {k: v.squeeze(0) for k, v in model_inputs.items()}


def make_compute_metrics(tokenizer):
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    def compute(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids = np.where(pred_ids == -100, tokenizer.pad_token_id, pred_ids)
        label_ids = np.where(pred.label_ids == -100, tokenizer.pad_token_id, pred.label_ids)

        preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        bleu = bleu_metric.compute(predictions=preds, references=[[l] for l in labels])
        chrf = chrf_metric.compute(predictions=preds, references=[[l] for l in labels], word_order=2)

        return {"bleu": bleu["score"], "chrf++": chrf["score"]}
    return compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--asr_dir", default=None,
                    help="Đường dẫn ASR model để tạo noisy source cho V3")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    out = Path(cfg.train.output_dir) / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_mt_model(cfg.model.mt_model)

    # V3: Nếu có asr_dir, tạo noisy source trước
    use_noisy_source = args.asr_dir is not None
    if use_noisy_source:
        print(f"V3 mode: training with noisy ASR transcripts from {args.asr_dir}")
        # TODO: Run ASR trên train/dev để tạo hyp_vi
        # Đây là bước chuẩn bị cho V3
        print("Note: Run ASR inference first to generate hyp_vi in manifest")

    train_ds = MTDataset(
        f"{cfg.data.manifest_dir}/train.jsonl",
        tokenizer,
        use_noisy_source=use_noisy_source
    )
    dev_ds = MTDataset(
        f"{cfg.data.manifest_dir}/dev.jsonl",
        tokenizer,
        use_noisy_source=False  # Dev luôn dùng clean source
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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
        generation_max_length=128,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
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
        compute_metrics=make_compute_metrics(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out / "best"))
    tokenizer.save_pretrained(str(out / "best"))


if __name__ == "__main__":
    main()
