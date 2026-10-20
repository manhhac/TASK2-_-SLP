#!/usr/bin/env python
"""Phân loại lỗi S/D/I, lỗi dấu thanh, số, tên riêng, từ mượn; WER theo nhóm."""
import argparse
import json
import re
from collections import Counter, defaultdict
import jiwer

VI_TONE = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.IGNORECASE)


def classify_errors(pred: str, ref: str):
    """
    Phân loại lỗi ASR.

    Returns:
        Dict với số lượng S/D/I + lỗi đặc thù: tone_err, num_err
    """
    out = jiwer.process_words(ref, pred)
    n_sub = n_del = n_ins = n_hit = 0
    tone_err = num_err = 0

    for ops in out.alignments[0]:
        for op in ops:
            if op.type == "substitute":
                n_sub += 1
                a, b = ref[op.ref_start:op.ref_end], pred[op.hyp_start:op.hyp_end]
                if VI_TONE.search(a) != VI_TONE.search(b):
                    tone_err += 1
                if re.search(r"\d", a) or re.search(r"\d", b):
                    num_err += 1
            elif op.type == "delete":
                n_del += 1
            elif op.type == "insert":
                n_ins += 1
            else:
                n_hit += 1

    return {
        "S": n_sub,
        "D": n_del,
        "I": n_ins,
        "H": n_hit,
        "tone_err": tone_err,
        "num_err": num_err,
        "wer": (n_sub + n_del + n_ins) / max(n_hit + n_sub + n_del, 1)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True,
                    help="jsonl {id, hyp_vi, ref_vi, speaker_id, gender, region, ...}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pred, encoding="utf-8")]
    totals = Counter()
    by_group = defaultdict(lambda: Counter())

    for r in rows:
        c = classify_errors(r["hyp_vi"], r["ref_vi"])
        for k, v in c.items():
            totals[k] += v
        for g in ["gender", "region", "source"]:
            if g in r:
                for k, v in c.items():
                    by_group[f"{g}={r[g]}"][k] += v

    n_hit = totals["H"]
    n_err = totals["S"] + totals["D"] + totals["I"]
    report = {
        "overall": {
            **totals,
            "wer": n_err / max(n_hit + n_err, 1)
        },
        "by_group": {
            g: {
                **c,
                "wer": (c["S"] + c["D"] + c["I"]) / max(c["H"] + c["S"] + c["D"], 1)
            }
            for g, c in by_group.items()
        },
    }

    open(args.out, "w").write(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
