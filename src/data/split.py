"""Chia dữ liệu thành train/dev/test theo speaker."""
import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def speaker_stratified_split(
    manifest_path: str,
    output_dir: str,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, int]:
    """
    Chia manifest thành train/dev/test theo speaker, stratify theo language + source.

    Args:
        manifest_path: File manifest đầu vào
        output_dir: Thư mục đầu ra
        train_ratio: Tỉ lệ train
        dev_ratio: Tỉ lệ dev
        test_ratio: Tỉ lệ test
        seed: Random seed

    Returns:
        Dict với số lượng mẫu mỗi split
    """
    # Load manifest
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    # Tạo nhóm tổ hợp để stratify
    df["strat"] = df["language"].astype(str) + "|" + df["source"].astype(str)

    # Gom theo speaker — mỗi speaker gán vào 1 fold
    speakers = df.groupby("speaker_id").agg(
        strat=("strat", lambda s: s.mode().iloc[0]),
        n=("strat", "size"),
    ).reset_index()

    # Số fold dựa trên tỉ lệ dev+test
    n_splits = int(round(1 / (dev_ratio + test_ratio)))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    # Dùng chính speaker làm "group" để không lọt speaker
    X = speakers[["strat"]].values
    y = speakers["strat"].values
    folds = list(sgkf.split(X, y, groups=speakers["speaker_id"].values))

    # Lấy 1 fold làm test, 1 fold làm dev, phần còn lại train
    test_idx = folds[0][1]
    dev_idx = folds[1][1]
    train_idx = np.setdiff1d(
        np.arange(len(speakers)),
        np.concatenate([test_idx, dev_idx])
    )

    sp_train = set(speakers.iloc[train_idx]["speaker_id"])
    sp_dev = set(speakers.iloc[dev_idx]["speaker_id"])
    sp_test = set(speakers.iloc[test_idx]["speaker_id"])

    df["split"] = df["speaker_id"].map(
        lambda s: "train" if s in sp_train else ("dev" if s in sp_dev else "test")
    )

    # Ghi ra file
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    split_counts = {}
    for split in ["train", "dev", "test"]:
        sub = df[df.split == split]
        path = out_path / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in sub.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_counts[split] = len(sub)
        print(f"{split}: {len(sub)} samples → {path}")

    return split_counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--dev_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    counts = speaker_stratified_split(
        args.manifest,
        args.output_dir,
        args.train_ratio,
        args.dev_ratio,
        args.test_ratio,
        args.seed
    )
    print(f"Split counts: {counts}")
