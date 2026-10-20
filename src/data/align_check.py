"""Kiểm tra căn chỉnh giữa audio và transcript dựa trên thời gian."""
import json
from pathlib import Path
from typing import Dict, List


def check_alignment(
    duration: float,
    text: str,
    min_chars_per_sec: float = 2.0,
    max_chars_per_sec: float = 25.0
) -> Dict:
    """
    Kiểm tra xem text có phù hợp với duration không.

    Args:
        duration: Thời lượng audio (giây)
        text: Transcript (chuỗi)
        min_chars_per_sec: Tối thiểu ký tự/giây
        max_chars_per_sec: Tối đa ký tự/giây

    Returns:
        Dict với keys: 'ok' (bool), 'chars_per_sec' (float), 'reason' (str)
    """
    if not text or duration <= 0:
        return {"ok": False, "chars_per_sec": 0, "reason": "invalid_input"}

    cps = len(text) / duration
    ok = min_chars_per_sec <= cps <= max_chars_per_sec

    reason = ""
    if not ok:
        if cps < min_chars_per_sec:
            reason = f"too_slow ({cps:.2f} chars/sec < {min_chars_per_sec})"
        else:
            reason = f"too_fast ({cps:.2f} chars/sec > {max_chars_per_sec})"

    return {"ok": ok, "chars_per_sec": cps, "reason": reason}


def check_manifest_alignment(
    manifest_path: str,
    min_chars_per_sec: float = 2.0,
    max_chars_per_sec: float = 25.0
) -> List[Dict]:
    """
    Kiểm tra alignment cho toàn bộ manifest file.

    Returns:
        List các sample không đạt alignment
    """
    bad_samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            result = check_alignment(
                row.get("duration", 0),
                row.get("text_vi", ""),
                min_chars_per_sec,
                max_chars_per_sec
            )
            if not result["ok"]:
                bad_samples.append({
                    "id": row.get("id"),
                    "audio_path": row.get("audio_path"),
                    "duration": row.get("duration"),
                    "text_vi": row.get("text_vi"),
                    **result
                })
    return bad_samples


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--min_cps", type=float, default=2.0)
    ap.add_argument("--max_cps", type=float, default=25.0)
    args = ap.parse_args()

    bad = check_manifest_alignment(args.manifest, args.min_cps, args.max_cps)
    print(f"Found {len(bad)} samples with alignment issues")
    for b in bad[:10]:  # Show first 10
        print(f"  {b['id']}: {b['reason']}")
