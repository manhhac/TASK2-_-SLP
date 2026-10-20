#!/usr/bin/env python
"""
Pipeline dữ liệu Task 2 — chạy từ đầu đến cuối:
    python -m src.data.prepare_data --config configs/baseline.yaml

Sinh ra:
    data/manifests/{train,dev,test}.jsonl
    data/features/feature_stats.json
    data/data_card.md
    data/eda_report.json
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.text_normalize import normalize_text_vi, normalize_text_en, NORMALIZATION_RULES_DOC


# ---------- 1. Thu thập ----------
def scan_audio_files(raw_dirs):
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    files = []
    for d in raw_dirs:
        for root, _, fs in os.walk(d):
            for f in fs:
                if Path(f).suffix.lower() in exts:
                    files.append(Path(root) / f)
    return files


def infer_label_from_path(p: Path):
    """Rule-based: dựa vào tên thư mục gốc."""
    s = str(p).lower()
    if any(k in s for k in ["vietnamese", "vivos", "nhatrang", "vi_"]) or "/vi/" in s:
        return "vi"
    if any(k in s for k in ["english", "librispeech", "commonvoice", "en_", "libri"]) or "/en/" in s:
        return "en"
    return "unknown"


# ---------- 2. Làm sạch audio ----------
def audio_quality_check(path: Path, cfg):
    """Trả về (ok, info) — info gồm duration, sr, rms, clip_ratio, lý do reject."""
    try:
        info = sf.info(str(path))
        duration = info.frames / info.samplerate
        if duration < cfg.min_dur:
            return False, {"reason": "too_short", "dur": duration}
        if duration > cfg.max_dur:
            return False, {"reason": "too_long", "dur": duration}
        y, sr = librosa.load(str(path), sr=cfg.sample_rate, mono=True)
        if len(y) == 0:
            return False, {"reason": "empty"}
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms < cfg.min_rms:
            return False, {"reason": "silent", "rms": rms}
        clip_ratio = float(np.mean(np.abs(y) > cfg.clip_threshold))
        if clip_ratio > cfg.clip_ratio_max:
            return False, {"reason": "clipping", "clip_ratio": clip_ratio}
        return True, {"duration": float(duration), "sr": int(sr), "rms": rms,
                      "clip_ratio": clip_ratio, "n_samples": len(y)}
    except Exception as e:
        return False, {"reason": f"decode_error:{type(e).__name__}"}


def load_and_resample(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.95
    return y.astype(np.float32)


# ---------- 3. Kiểm tra căn chỉnh audio–text ----------
def alignment_ok(text_vi: str, duration: float, cfg) -> bool:
    if not text_vi or duration <= 0:
        return False
    cps = len(text_vi) / duration
    return cfg.min_chars_per_sec <= cps <= cfg.max_chars_per_sec


# ---------- 4. Kiểm tra bản dịch EN ----------
def translation_ok(vi: str, en: str) -> bool:
    """Lọc các cặp dịch rõ ràng sai (quá ngắn, trùng nguyên văn VI, ký tự lạ)."""
    if not en or not vi:
        return False
    if len(en.split()) < 1:
        return False
    # Tỉ lệ độ dài từ VI-EN kỳ vọng ~0.7-1.5
    ratio = len(en.split()) / max(1, len(vi.split()))
    if ratio < 0.3 or ratio > 2.5:
        return False
    # Nếu EN trùng VI > 80% token → coi như chưa dịch
    vi_tokens = set(vi.split())
    en_tokens = set(en.split())
    if vi_tokens and en_tokens:
        overlap = len(vi_tokens & en_tokens) / len(en_tokens)
        if overlap > 0.8 and len(en_tokens) > 3:
            return False
    return True


# ---------- 5. Loại trùng ----------
def audio_hash(y: np.ndarray) -> str:
    return hashlib.md5(y.tobytes()).hexdigest()


def text_hash(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def dedup(rows):
    seen_audio, seen_text = set(), set()
    out = []
    for r in rows:
        ah, th = r["audio_hash"], r["text_hash"]
        if ah in seen_audio or th in seen_text:
            continue
        seen_audio.add(ah)
        seen_text.add(th)
        out.append(r)
    return out


# ---------- 6. Chia tập theo speaker ----------
def speaker_stratified_split(rows, cfg):
    """Chia theo speaker_id, stratify theo language + source."""
    df = pd.DataFrame(rows)
    # Tạo nhóm tổ hợp để stratify
    df["strat"] = df["language"].astype(str) + "|" + df["source"].astype(str)
    # Gom theo speaker — mỗi speaker gán vào 1 fold
    speakers = df.groupby("speaker_id").agg(
        strat=("strat", lambda s: s.mode().iloc[0]),
        n=("strat", "size"),
    ).reset_index()
    sgkf = StratifiedGroupKFold(
        n_splits=int(round(1 / (cfg.dev_ratio + cfg.test_ratio))),
        shuffle=True, random_state=cfg.seed,
    )
    # Dùng chính speaker làm "group" để không lọt speaker
    X = speakers[["strat"]].values
    y = speakers["strat"].values
    folds = list(sgkf.split(X, y, groups=speakers["speaker_id"].values))
    # Lấy 1 fold làm test, 1 fold làm dev, phần còn lại train
    test_idx = folds[0][1]
    dev_idx = folds[1][1]
    train_idx = np.setdiff1d(np.arange(len(speakers)),
                             np.concatenate([test_idx, dev_idx]))
    sp_train = set(speakers.iloc[train_idx]["speaker_id"])
    sp_dev = set(speakers.iloc[dev_idx]["speaker_id"])
    sp_test = set(speakers.iloc[test_idx]["speaker_id"])
    df["split"] = df["speaker_id"].map(
        lambda s: "train" if s in sp_train else ("dev" if s in sp_dev else "test")
    )
    return df


# ---------- 7. Phân tích mất cân bằng ----------
def imbalance_stats(df):
    stats = {}
    for key in ["language", "source", "gender", "region"]:
        if key in df:
            vc = df[key].astype(str).value_counts().to_dict()
            stats[key] = vc
    # độ dài câu
    if "duration" in df:
        stats["duration_bins"] = pd.cut(df["duration"],
                                        bins=[0, 3, 5, 10, 20, 30]).value_counts().to_dict()
        stats["duration_bins"] = {str(k): int(v) for k, v in stats["duration_bins"].items()}
    # tần suất token
    all_text = " ".join(df["text_vi"].fillna("").tolist())
    token_freq = Counter(all_text.split())
    stats["top_tokens"] = token_freq.most_common(20)
    stats["n_unique_tokens"] = len(token_freq)
    return stats


def apply_imbalance_fixes(df, cfg):
    """2 kỹ thuật: (a) speaker cap; (b) oversample minority language trong train."""
    df = df.copy()
    train = df[df.split == "train"].copy()
    others = df[df.split != "train"].copy()

    # (a) speaker cap — tránh 1 speaker chiếm quá nhiều
    cap = int(train.groupby("speaker_id").size().median() * 3)
    train = (train.groupby("speaker_id", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), cap), random_state=cfg.seed)))

    # (b) oversample language minority (augment flag)
    vc = train["language"].value_counts()
    if len(vc) >= 2:
        maj, mino = vc.idxmax(), vc.idxmin()
        n_maj, n_mino = vc[maj], vc[mino]
        target = int(n_maj * 0.85)
        if n_mino < target:
            extra = train[train.language == mino].sample(
                n=target - n_mino, replace=True, random_state=cfg.seed)
            extra = extra.copy()
            extra["is_augmented"] = True
            train = pd.concat([train, extra], ignore_index=True)

    return pd.concat([train, others], ignore_index=True)


# ---------- 8. Main pipeline ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.data.seed)

    out_dir = Path(cfg.data.manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.data.feature_dir).mkdir(parents=True, exist_ok=True)

    # 1. Thu thập
    print("[1/8] Scanning audio files...")
    files = scan_audio_files(cfg.data.raw_dirs)
    print(f"  Found {len(files)} files")

    # Tải metadata sidecar nếu có (text + speaker + translation)
    # Convention: cùng thư mục, file cùng tên .txt hoặc transcripts.tsv
    # Ở đây giả định có transcripts.tsv: audio_path, text_vi, text_en, speaker_id, gender, region
    records = []
    meta_path = Path(cfg.data.raw_dirs[0]) / "transcripts.tsv"
    meta = {}
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                meta[parts[0]] = {
                    "text_vi": parts[1] if len(parts) > 1 else "",
                    "text_en": parts[2] if len(parts) > 2 else "",
                    "speaker_id": parts[3] if len(parts) > 3 else "unknown",
                    "gender": parts[4] if len(parts) > 4 else "unknown",
                    "region": parts[5] if len(parts) > 5 else "unknown",
                }

    # 2-4. Quality check + alignment
    print("[2/8] Checking audio quality + alignment...")
    kept, rejects = [], Counter()
    for p in tqdm(files):
        ok, info = audio_quality_check(p, cfg.data)
        if not ok:
            rejects[info["reason"]] += 1
            continue
        key = p.name
        m = meta.get(key, {})
        text_vi = normalize_text_vi(m.get("text_vi", ""), number_to_words=False)
        text_en = normalize_text_en(m.get("text_en", ""))
        if not text_vi:
            rejects["no_transcript"] += 1
            continue
        if not alignment_ok(text_vi, info["duration"], cfg.data):
            rejects["alignment_fail"] += 1
            continue
        # load to hash
        y = load_and_resample(p, cfg.data.sample_rate)
        records.append({
            "id": key,
            "audio_path": str(p.resolve()),
            "duration": round(info["duration"], 3),
            "sample_rate": cfg.data.sample_rate,
            "rms": round(info["rms"], 4),
            "text_vi": text_vi,
            "text_en": text_en,
            "speaker_id": m.get("speaker_id", "unknown"),
            "gender": m.get("gender", "unknown"),
            "region": m.get("region", "unknown"),
            "source": p.parts[-2] if len(p.parts) >= 2 else "unknown",
            "language": "vi" if infer_label_from_path(p) == "vi" else "en",
            "audio_hash": audio_hash(y),
            "text_hash": text_hash(text_vi),
            "is_augmented": False,
        })
    print(f"  Kept: {len(records)} | Rejected: {dict(rejects)}")

    # 5. Check EN translation
    print("[3/8] Checking EN translations...")
    before = len(records)
    records = [r for r in records if translation_ok(r["text_vi"], r["text_en"])]
    print(f"  Removed {before - len(records)} invalid translation pairs")

    # 6. Dedup
    print("[4/8] Removing duplicates...")
    before = len(records)
    records = dedup(records)
    print(f"  Removed {before - len(records)} duplicates")

    # 7. Analyze imbalance + fix
    print("[5/8] Analyzing imbalance...")
    df_raw = pd.DataFrame(records)
    stats_before = imbalance_stats(df_raw)

    # 8. Split
    print("[6/8] Splitting by speaker...")
    df = speaker_stratified_split(records, cfg.data)

    # 9. Apply balancing
    print("[7/8] Balancing train data...")
    df = apply_imbalance_fixes(df, cfg.data)
    stats_after = imbalance_stats(df)

    # 10. Write manifests
    print("[8/8] Writing manifests + data card...")
    for split in ["train", "dev", "test"]:
        sub = df[df.split == split]
        path = out_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in sub.to_dict(orient="records"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(sub)} samples -> {path}")

    # Save report
    report = {
        "n_total": int(len(df)),
        "splits": {s: int((df.split == s).sum()) for s in ["train", "dev", "test"]},
        "rejects": dict(rejects),
        "stats_before": stats_before,
        "stats_after": stats_after,
        "normalization_rules": NORMALIZATION_RULES_DOC,
    }
    (out_dir / "pipeline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    write_data_card(df, report, out_dir / "data_card.md")
    print("Phase 1 complete")


def write_data_card(df, report, path):
    lines = ["# Data Card — Task 2 ASR VI + MT EN", "",
             "## Nguồn & giấy phép", "Xem `configs/*.yaml` và `report['sources']`.", "",
             "## Quy trình xử lý", NORMALIZATION_RULES_DOC, "",
             "## Thống kê", f"- Tổng mẫu: {report['n_total']}",
             f"- Splits: {report['splits']}", "",
             "## Mất cân bằng trước xử lý", "```json",
             json.dumps(report["stats_before"], ensure_ascii=False, indent=2), "```", "",
             "## Sau xử lý", "```json",
             json.dumps(report["stats_after"], ensure_ascii=False, indent=2), "```", "",
             "## Hạn chế / thiên lệch đã biết",
             "- Domain mismatch giữa nguồn đọc sách và hội thoại",
             "- Thiếu test OOD (giọng vùng miền chưa có)",
             "- Tỉ lệ EN/VI vẫn lệch sau cân bằng (đã cap, oversample)",
             ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
