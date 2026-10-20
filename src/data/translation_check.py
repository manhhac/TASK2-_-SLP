"""Kiểm tra chất lượng bản dịch VI-EN."""
import json
from pathlib import Path
from typing import Dict, List


def check_translation(vi: str, en: str) -> Dict:
    """
    Kiểm tra chất lượng cặp dịch VI-EN.

    Args:
        vi: Text tiếng Việt
        en: Text tiếng Anh

    Returns:
        Dict với keys: 'ok' (bool), 'reason' (str), 'ratio' (float)
    """
    if not en or not vi:
        return {"ok": False, "reason": "missing_text", "ratio": 0}

    if len(en.split()) < 1:
        return {"ok": False, "reason": "en_too_short", "ratio": 0}

    # Tỉ lệ độ dài từ VI-EN kỳ vọng ~0.7-1.5
    ratio = len(en.split()) / max(1, len(vi.split()))

    if ratio < 0.3:
        return {"ok": False, "reason": f"en_too_short (ratio={ratio:.2f})", "ratio": ratio}
    if ratio > 2.5:
        return {"ok": False, "reason": f"en_too_long (ratio={ratio:.2f})", "ratio": ratio}

    # Nếu EN trùng VI > 80% token → coi như chưa dịch
    vi_tokens = set(vi.split())
    en_tokens = set(en.split())
    if vi_tokens and en_tokens:
        overlap = len(vi_tokens & en_tokens) / len(en_tokens)
        if overlap > 0.8 and len(en_tokens) > 3:
            return {"ok": False, "reason": f"high_overlap ({overlap:.2f})", "ratio": ratio}

    return {"ok": True, "reason": "", "ratio": ratio}


def check_manifest_translation(manifest_path: str) -> List[Dict]:
    """
    Kiểm tra translation cho toàn bộ manifest file.

    Returns:
        List các sample không đạt translation
    """
    bad_samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            result = check_translation(row.get("text_vi", ""), row.get("text_en", ""))
            if not result["ok"]:
                bad_samples.append({
                    "id": row.get("id"),
                    "audio_path": row.get("audio_path"),
                    "text_vi": row.get("text_vi"),
                    "text_en": row.get("text_en"),
                    **result
                })
    return bad_samples


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    bad = check_manifest_translation(args.manifest)
    print(f"Found {len(bad)} samples with translation issues")
    for b in bad[:10]:  # Show first 10
        print(f"  {b['id']}: {b['reason']}")
