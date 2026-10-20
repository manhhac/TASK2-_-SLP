"""Phân tích và xử lý mất cân bằng dữ liệu."""
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd


def analyze_imbalance(manifest_path: str) -> Dict:
    """
    Phân tích mất cân bằng trong manifest file.

    Returns:
        Dict chứa thống kê theo các维度: language, source, gender, region, duration
    """
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    stats = {}

    # Thống kê theo categorical
    for key in ["language", "source", "gender", "region"]:
        if key in df:
            vc = df[key].astype(str).value_counts().to_dict()
            stats[key] = vc

    # Thống kê theo duration
    if "duration" in df:
        stats["duration_bins"] = pd.cut(
            df["duration"],
            bins=[0, 3, 5, 10, 20, 30]
        ).value_counts().to_dict()
        stats["duration_bins"] = {str(k): int(v) for k, v in stats["duration_bins"].items()}
        stats["duration_stats"] = {
            "mean": float(df["duration"].mean()),
            "std": float(df["duration"].std()),
            "min": float(df["duration"].min()),
            "max": float(df["duration"].max()),
        }

    # Thống kê token frequency
    if "text_vi" in df:
        all_text = " ".join(df["text_vi"].fillna("").tolist())
        token_freq = Counter(all_text.split())
        stats["top_tokens"] = token_freq.most_common(20)
        stats["n_unique_tokens"] = len(token_freq)

    return stats


def apply_speaker_cap(manifest_path: str, output_path: str, cap_multiplier: float = 3.0) -> int:
    """
    Giới hạn số mẫu mỗi speaker (speaker cap).

    Args:
        manifest_path: File manifest đầu vào
        output_path: File manifest đầu ra
        cap_multiplier: Hệ số cap so với median

    Returns:
        Số lượng mẫu bị loại bỏ
    """
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    before = len(df)

    # Tính cap
    cap = int(df.groupby("speaker_id").size().median() * cap_multiplier)

    # Giới hạn mỗi speaker
    def cap_group(g):
        n = len(g)
        if n <= cap:
            return g
        return g.sample(cap, random_state=42)

    df = df.groupby("speaker_id", group_keys=False).apply(cap_group).reset_index(drop=True)

    # Ghi ra file
    with open(output_path, "w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return before - len(df)


def oversample_minority_language(
    manifest_path: str,
    output_path: str,
    target_ratio: float = 0.85
) -> int:
    """
    Oversample ngôn ngữ ít mẫu hơn trong train set.

    Args:
        manifest_path: File manifest đầu vào
        output_path: File manifest đầu ra
        target_ratio: Tỉ lệ giữa ngôn ngữ ít và nhiều (0.85 = 85%)

    Returns:
        Số lượng mẫu được thêm vào
    """
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    before = len(df)

    vc = df["language"].value_counts()
    if len(vc) < 2:
        return 0  # Chỉ có 1 ngôn ngữ

    maj, mino = vc.idxmax(), vc.idxmin()
    n_maj, n_mino = vc[maj], vc[mino]
    target = int(n_maj * target_ratio)

    if n_mino >= target:
        return 0  # Đã cân bằng

    # Oversample
    extra = df[df.language == mino].sample(n=target - n_mino, replace=True, random_state=42)
    extra = extra.copy()
    extra["is_augmented"] = True

    df = pd.concat([df, extra], ignore_index=True)

    # Ghi ra file
    with open(output_path, "w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(df) - before


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--action", choices=["analyze", "cap", "oversample"], required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.action == "analyze":
        stats = analyze_imbalance(args.manifest)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    elif args.action == "cap":
        n_removed = apply_speaker_cap(args.manifest, args.output or args.manifest)
        print(f"Removed {n_removed} samples")
    elif args.action == "oversample":
        n_added = oversample_minority_language(args.manifest, args.output or args.manifest)
        print(f"Added {n_added} samples")
