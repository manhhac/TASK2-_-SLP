# Data Card — Task 2 ASR VI + MT EN

## Nguồn & giấy phép
- **Bộ dữ liệu**: VIVOS (Vietnamese Speech Corpus) — AILAB, VNUHCM-University of Science.
- **Link tải**: https://zenodo.org/record/7068130
- **Giấy phép**: CC BY-NC-SA 4.0 (chỉ dùng học thuật, phi thương mại).
- **Script tái lập**: `python -m src.data.prepare_data --config configs/baseline.yaml`
- **Manifest**: `data/manifests/{train,dev,test}.jsonl` (mỗi dòng 1 mẫu: id, audio_path, duration, text_vi, text_en, speaker_id, gender, region, source, split, hash).

## Quy trình xử lý

QUY TẮC CHUẨN HOÁ VĂN BẢN (áp dụng nhất quán cho train/dev/test):
1. Unicode NFC (quan trọng với tiếng Việt: tổ hợp dấu thanh)
2. Lowercase toàn bộ
3. Loại bỏ ký tự điều khiển, ký tự lạ không thuộc bảng chữ Việt/ASCII
4. Số: mặc định tách từng chữ số; có thể bật number_to_words để đọc thành lời
5. Mở rộng từ viết tắt phổ biến (tp., q., p., vn, hcm, ...)
6. Chuẩn hoá khoảng trắng (1 space), strip 2 đầu
7. Gộp dấu câu lặp (!!! → !, ... → .)
8. Dấu câu giữ lại: . , ! ? ; : ' " ( ) -


## Thống kê
- Tổng mẫu: 10887 (sau làm sạch + dedup; raw VIVOS 12.420)
- Splits: {'train': 8253, 'dev': 1084, 'test': 1550}
- Số speaker: 65 (train 47 / dev 9 / test 9 — không speaker nào lặp giữa các split)
- Mẫu/speaker: trung bình 167,5 | median 200 | min 34 | max 300 | std 97,1
- Giới tính: nữ 6.407 (58,8%) / nam 4.480 (41,2%); vùng: 100% miền Bắc

## Mất cân bằng trước xử lý
```json
{
  "language": {
    "vi": 10887
  },
  "source": {
    "VIVOSSPK41": 300,
    "VIVOSSPK45": 300,
    "VIVOSSPK34": 299,
    "VIVOSSPK44": 298,
    "VIVOSSPK46": 298,
    "VIVOSSPK26": 297,
    "VIVOSSPK40": 297,
    "VIVOSSPK42": 297,
    "VIVOSSPK43": 293,
    "VIVOSSPK38": 292,
    "VIVOSSPK31": 274,
    "VIVOSSPK35": 259,
    "VIVOSSPK03": 250,
    "VIVOSSPK08": 250,
    "VIVOSSPK18": 250,
    "VIVOSSPK09": 245,
    "VIVOSSPK27": 243,
    "VIVOSSPK02": 242,
    "VIVOSSPK11": 241,
    "VIVOSSPK07": 240,
    "VIVOSSPK04": 239,
    "VIVOSSPK05": 236,
    "VIVOSSPK06": 236,
    "VIVOSSPK01": 235,
    "VIVOSSPK13": 223,
    "VIVOSSPK36": 223,
    "VIVOSSPK33": 219,
    "VIVOSSPK32": 211,
    "VIVOSSPK17": 201,
    "VIVOSSPK10": 200,
    "VIVOSSPK12": 200,
    "VIVOSSPK16": 200,
    "VIVOSSPK29": 200,
    "VIVOSSPK39": 182,
    "VIVOSSPK14": 181,
    "VIVOSSPK15": 178,
    "VIVOSSPK25": 178,
    "VIVOSSPK28": 169,
    "VIVOSSPK22": 161,
    "VIVOSSPK30": 152,
    "VIVOSSPK20": 145,
    "VIVOSSPK37": 131,
    "VIVOSSPK19": 130,
    "VIVOSSPK23": 109,
    "VIVOSSPK24": 100,
    "VIVOSSPK21": 43,
    "VIVOSDEV01": 40,
    "VIVOSDEV02": 40,
    "VIVOSDEV03": 40,
    "VIVOSDEV04": 40,
    "VIVOSDEV05": 40,
    "VIVOSDEV06": 40,
    "VIVOSDEV07": 40,
    "VIVOSDEV13": 40,
    "VIVOSDEV14": 40,
    "VIVOSDEV15": 40,
    "VIVOSDEV16": 40,
    "VIVOSDEV17": 40,
    "VIVOSDEV18": 40,
    "VIVOSDEV19": 40,
    "VIVOSDEV11": 38,
    "VIVOSDEV10": 37,
    "VIVOSDEV12": 36,
    "VIVOSDEV09": 35,
    "VIVOSDEV08": 34
  },
  "gender": {
    "female": 6407,
    "male": 4480
  },
  "region": {
    "north": 10887
  },
  "duration_bins": {
    "(3, 5]": 5054,
    "(5, 10]": 3163,
    "(0, 3]": 2264,
    "(10, 20]": 406,
    "(20, 30]": 0
  },
  "top_tokens": [
    [
      "có",
      2101
    ],
    [
      "là",
      1751
    ],
    [
      "không",
      1701
    ],
    [
      "một",
      1615
    ],
    [
      "của",
      1608
    ],
    [
      "và",
      1523
    ],
    [
      "người",
      1325
    ],
    [
      "những",
      1298
    ],
    [
      "được",
      1208
    ],
    [
      "tôi",
      1205
    ],
    [
      "cho",
      1103
    ],
    [
      "các",
      1073
    ],
    [
      "trong",
      941
    ],
    [
      "đã",
      882
    ],
    [
      "này",
      830
    ],
    [
      "làm",
      820
    ],
    [
      "cũng",
      812
    ],
    [
      "để",
      797
    ],
    [
      "với",
      794
    ],
    [
      "như",
      743
    ]
  ],
  "n_unique_tokens": 4940
}
```

## Sau xử lý
```json
{
  "language": {
    "vi": 10887
  },
  "source": {
    "VIVOSSPK41": 300,
    "VIVOSSPK45": 300,
    "VIVOSSPK34": 299,
    "VIVOSSPK44": 298,
    "VIVOSSPK46": 298,
    "VIVOSSPK26": 297,
    "VIVOSSPK40": 297,
    "VIVOSSPK42": 297,
    "VIVOSSPK43": 293,
    "VIVOSSPK38": 292,
    "VIVOSSPK31": 274,
    "VIVOSSPK35": 259,
    "VIVOSSPK03": 250,
    "VIVOSSPK08": 250,
    "VIVOSSPK18": 250,
    "VIVOSSPK09": 245,
    "VIVOSSPK27": 243,
    "VIVOSSPK02": 242,
    "VIVOSSPK11": 241,
    "VIVOSSPK07": 240,
    "VIVOSSPK04": 239,
    "VIVOSSPK05": 236,
    "VIVOSSPK06": 236,
    "VIVOSSPK01": 235,
    "VIVOSSPK13": 223,
    "VIVOSSPK36": 223,
    "VIVOSSPK33": 219,
    "VIVOSSPK32": 211,
    "VIVOSSPK17": 201,
    "VIVOSSPK10": 200,
    "VIVOSSPK29": 200,
    "VIVOSSPK12": 200,
    "VIVOSSPK16": 200,
    "VIVOSSPK39": 182,
    "VIVOSSPK14": 181,
    "VIVOSSPK25": 178,
    "VIVOSSPK15": 178,
    "VIVOSSPK28": 169,
    "VIVOSSPK22": 161,
    "VIVOSSPK30": 152,
    "VIVOSSPK20": 145,
    "VIVOSSPK37": 131,
    "VIVOSSPK19": 130,
    "VIVOSSPK23": 109,
    "VIVOSSPK24": 100,
    "VIVOSSPK21": 43,
    "VIVOSDEV02": 40,
    "VIVOSDEV03": 40,
    "VIVOSDEV05": 40,
    "VIVOSDEV07": 40,
    "VIVOSDEV15": 40,
    "VIVOSDEV16": 40,
    "VIVOSDEV19": 40,
    "VIVOSDEV01": 40,
    "VIVOSDEV04": 40,
    "VIVOSDEV06": 40,
    "VIVOSDEV13": 40,
    "VIVOSDEV14": 40,
    "VIVOSDEV17": 40,
    "VIVOSDEV18": 40,
    "VIVOSDEV11": 38,
    "VIVOSDEV10": 37,
    "VIVOSDEV12": 36,
    "VIVOSDEV09": 35,
    "VIVOSDEV08": 34
  },
  "gender": {
    "female": 6407,
    "male": 4480
  },
  "region": {
    "north": 10887
  },
  "duration_bins": {
    "(3, 5]": 5054,
    "(5, 10]": 3163,
    "(0, 3]": 2264,
    "(10, 20]": 406,
    "(20, 30]": 0
  },
  "top_tokens": [
    [
      "có",
      2101
    ],
    [
      "là",
      1751
    ],
    [
      "không",
      1701
    ],
    [
      "một",
      1615
    ],
    [
      "của",
      1608
    ],
    [
      "và",
      1523
    ],
    [
      "người",
      1325
    ],
    [
      "những",
      1298
    ],
    [
      "được",
      1208
    ],
    [
      "tôi",
      1205
    ],
    [
      "cho",
      1103
    ],
    [
      "các",
      1073
    ],
    [
      "trong",
      941
    ],
    [
      "đã",
      882
    ],
    [
      "này",
      830
    ],
    [
      "làm",
      820
    ],
    [
      "cũng",
      812
    ],
    [
      "để",
      797
    ],
    [
      "với",
      794
    ],
    [
      "như",
      743
    ]
  ],
  "n_unique_tokens": 4940
}
```

## Hạn chế / thiên lệch đã biết
- Domain mismatch giữa nguồn đọc sách và hội thoại
- Thiếu test OOD (giọng vùng miền chưa có)
- VIVOS chỉ có tiếng Việt: kỹ thuật oversample ngôn ngữ thiểu số không kích hoạt
- Speaker cap (median×3 = 600) cao hơn speaker nhiều mẫu nhất (300) nên không cắt mẫu nào
- Vùng miền 100% miền Bắc; không có giọng Nam/Trung
- `text_en` để trống (ASR-only); nhãn EN cho MT sinh bằng MarianMT → rủi ro overfit, xem báo cáo