"""Loại bỏ mẫu trùng lặp dựa trên audio hash và text hash."""
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Set


def audio_hash(data: bytes) -> str:
    """Tạo hash từ dữ liệu audio bytes."""
    return hashlib.md5(data).hexdigest()


def text_hash(text: str) -> str:
    """Tạo hash từ text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def dedup_manifest(manifest_path: str, output_path: str = None) -> int:
    """
    Loại bỏ mẫu trùng lặp từ manifest file.

    Args:
        manifest_path: Đường dẫn file manifest đầu vào
        output_path: Đường dẫn file manifest đầu ra (nếu None, ghi đè input)

    Returns:
        Số lượng mẫu bị loại bỏ
    """
    seen_audio: Set[str] = set()
    seen_text: Set[str] = set()
    kept_rows: List[Dict] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ah = row.get("audio_hash", "")
            th = row.get("text_hash", "")

            if ah in seen_audio or th in seen_text:
                continue

            seen_audio.add(ah)
            seen_text.add(th)
            kept_rows.append(row)

    if output_path is None:
        output_path = manifest_path

    with open(output_path, "w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(seen_audio)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    n_kept = dedup_manifest(args.manifest, args.output)
    print(f"Kept {n_kept} unique samples")
