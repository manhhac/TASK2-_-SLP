import torch
from transformers import MarianMTModel, MarianTokenizer


def build_mt_model(model_name="Helsinki-NLP/opus-mt-vi-en"):
    """
    Xây dựng MarianMT model cho dịch tiếng Việt sang tiếng Anh.

    Args:
        model_name: Tên model pre-trained

    Returns:
        model, tokenizer
    """
    tok = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tok


@torch.no_grad()
def translate_vi_en(
    texts,
    model,
    tok,
    batch_size=16,
    device="cuda",
    num_beams=4,
    max_length=256
):
    """
    Dịch danh sách text tiếng Việt sang tiếng Anh.

    Args:
        texts: List[str] text tiếng Việt
        model: MarianMT model
        tok: MarianTokenizer
        batch_size: Batch size
        device: Device (cuda/cpu)
        num_beams: Beam size cho beam search
        max_length: Độ dài tối đa cho output

    Returns:
        List[str] text tiếng Anh
    """
    model.eval().to(device)
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(**enc, num_beams=num_beams, max_length=max_length)
        outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return outs
