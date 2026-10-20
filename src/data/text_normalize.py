import re
import unicodedata
from num2words import num2words  # pip install num2words

# Quy tắc:
# 1. Unicode NFC
# 2. Lowercase
# 3. Xoá ký tự lạ, giữ dấu câu cơ bản
# 4. Số → chữ (tuỳ chọn)
# 5. Từ viết tắt phổ biến
# 6. Chuẩn hoá khoảng trắng

ABBREVIATIONS_VI = {
    "tp.": "thành phố", "tp": "thành phố",
    "tw.": "thị xã", "tw": "thị xã",
    "q.": "quận", "q": "quận",
    "p.": "phường", "p": "phường",
    "nxb": "nhà xuất bản",
    "vn": "việt nam",
    "hcm": "hồ chí minh",
    "hn": "hà nội",
    "đt": "điện thoại",
    "ko": "không", "k": "không",
    "dc": "được", "đc": "được",
    "ntn": "như thế nào",
    "vs": "với",
    "tv": "thành viên",
}

PUNCT_KEEP = r".,!?;:'\"()\-"


def normalize_text_vi(text: str, number_to_words: bool = False) -> str:
    if not isinstance(text, str):
        return ""
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)
    # 2. Lowercase
    text = text.lower()
    # 3. Bỏ ký tự điều khiển, ký tự lạ (giữ chữ Việt + ASCII + dấu câu)
    text = re.sub(r"[^\w\s" + re.escape(PUNCT_KEEP) + r"À-ỹ]", " ", text, flags=re.UNICODE)
    # 4. Số → chữ
    if number_to_words:
        def _num2words(m):
            try:
                return num2words(int(m.group()), lang="vi")
            except Exception:
                return m.group()
        text = re.sub(r"\d+", _num2words, text)
    else:
        text = re.sub(r"\d+", lambda m: " ".join(list(m.group())), text)
    # 5. Viết tắt
    for abbr, full in ABBREVIATIONS_VI.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    # 6. Chuẩn hoá khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    # 7. Bỏ dấu câu lặp
    text = re.sub(r"([.,!?;:])\1+", r"\1", text)
    return text


def normalize_text_en(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s" + re.escape(PUNCT_KEEP) + r"]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


NORMALIZATION_RULES_DOC = """
QUY TẮC CHUẨN HOÁ VĂN BẢN (áp dụng nhất quán cho train/dev/test):
1. Unicode NFC (quan trọng với tiếng Việt: tổ hợp dấu thanh)
2. Lowercase toàn bộ
3. Loại bỏ ký tự điều khiển, ký tự lạ không thuộc bảng chữ Việt/ASCII
4. Số: mặc định tách từng chữ số; có thể bật number_to_words để đọc thành lời
5. Mở rộng từ viết tắt phổ biến (tp., q., p., vn, hcm, ...)
6. Chuẩn hoá khoảng trắng (1 space), strip 2 đầu
7. Gộp dấu câu lặp (!!! → !, ... → .)
8. Dấu câu giữ lại: . , ! ? ; : ' " ( ) -
"""
