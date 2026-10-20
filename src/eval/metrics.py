import evaluate
from typing import List

_wer = evaluate.load("wer")
_cer = evaluate.load("cer")
_bleu = evaluate.load("sacrebleu")
_chrf = evaluate.load("chrf")


def compute_asr_metrics(preds: List[str], refs: List[str]) -> dict:
    """
    Tính toán metrics cho ASR task.

    Args:
        preds: List[str] predictions
        refs: List[str] references

    Returns:
        Dict với wer, cer
    """
    return {
        "wer": 100 * _wer.compute(predictions=preds, references=refs),
        "cer": 100 * _cer.compute(predictions=preds, references=refs),
    }


def compute_mt_metrics(preds: List[str], refs: List[str]) -> dict:
    """
    Tính toán metrics cho MT task.

    Args:
        preds: List[str] predictions
        refs: List[str] references

    Returns:
        Dict với bleu, chrf++
    """
    bleu = _bleu.compute(predictions=preds, references=[[r] for r in refs])
    chrf = _chrf.compute(predictions=preds, references=[[r] for r in refs], word_order=2)
    return {"bleu": bleu["score"], "chrf++": chrf["score"]}


def compute_all_metrics(
    asr_preds: List[str],
    asr_refs: List[str],
    mt_preds: List[str],
    mt_refs: List[str]
) -> dict:
    """
    Tính toán tất cả metrics cho pipeline ASR+MT.

    Args:
        asr_preds: ASR predictions (VI)
        asr_refs: ASR references (VI)
        mt_preds: MT predictions (EN)
        mt_refs: MT references (EN)

    Returns:
        Dict với tất cả metrics
    """
    return {
        "asr": compute_asr_metrics(asr_preds, asr_refs),
        "mt": compute_mt_metrics(mt_preds, mt_refs),
    }
