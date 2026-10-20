# Data Card — Task 2 ASR VI + MT EN

## Nguồn & giấy phép

**Nguồn chính:** Synthetic sample data (TTS-generated)
- 20 samples, 3 giây mỗi sample
- 1 speaker (speaker_0)
- License: Demo/Testing only

**Nguồn phụ (đã tải, chưa dùng):**
- VIVOS Dataset: 11,660 utterances (15 giờ)
- License: CC BY-NC-SA 4.0
- Được tải nhưng chưa convert và sử dụng do hạn chế thời gian

**Lưu ý quan trọng:**
- Data hiện tại là synthetic, không phản ánh audio thật
- Whisper pretrained không hoạt động tốt với TTS audio
- Pipeline code đã đúng, nhưng kết quả số liệu không có ý nghĩa thực tế

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
- Tổng mẫu: 10
- Splits: {'train': 3, 'dev': 3, 'test': 4}

## Mất cân bằng trước xử lý
```json
{
  "language": {
    "vi": 10
  },
  "source": {
    "sample": 10
  },
  "gender": {
    "male": 5,
    "female": 5
  },
  "region": {
    "north": 10
  },
  "duration_bins": {
    "(0, 3]": 10,
    "(3, 5]": 0,
    "(5, 10]": 0,
    "(10, 20]": 0,
    "(20, 30]": 0
  },
  "top_tokens": [
    [
      "xin",
      2
    ],
    [
      "cảm",
      2
    ],
    [
      "ơn",
      2
    ],
    [
      "tôi",
      2
    ],
    [
      "là",
      2
    ],
    [
      "chào",
      1
    ],
    [
      "tên",
      1
    ],
    [
      "hôm",
      1
    ],
    [
      "nay",
      1
    ],
    [
      "thứ",
      1
    ],
    [
      "hai",
      1
    ],
    [
      "trời",
      1
    ],
    [
      "đẹp",
      1
    ],
    [
      "quá",
      1
    ],
    [
      "thích",
      1
    ],
    [
      "học",
      1
    ],
    [
      "tiếng",
      1
    ],
    [
      "anh",
      1
    ],
    [
      "lỗi",
      1
    ],
    [
      "tạm",
      1
    ]
  ],
  "n_unique_tokens": 28
}
```

## Sau xử lý
```json
{
  "language": {
    "vi": 10
  },
  "source": {
    "sample": 10
  },
  "gender": {
    "male": 5,
    "female": 5
  },
  "region": {
    "north": 10
  },
  "duration_bins": {
    "(0, 3]": 10,
    "(3, 5]": 0,
    "(5, 10]": 0,
    "(10, 20]": 0,
    "(20, 30]": 0
  },
  "top_tokens": [
    [
      "tôi",
      2
    ],
    [
      "là",
      2
    ],
    [
      "xin",
      2
    ],
    [
      "cảm",
      2
    ],
    [
      "ơn",
      2
    ],
    [
      "tên",
      1
    ],
    [
      "thích",
      1
    ],
    [
      "học",
      1
    ],
    [
      "tiếng",
      1
    ],
    [
      "anh",
      1
    ],
    [
      "chúc",
      1
    ],
    [
      "một",
      1
    ],
    [
      "ngày",
      1
    ],
    [
      "tốt",
      1
    ],
    [
      "lành",
      1
    ],
    [
      "chào",
      1
    ],
    [
      "hôm",
      1
    ],
    [
      "nay",
      1
    ],
    [
      "thứ",
      1
    ],
    [
      "hai",
      1
    ]
  ],
  "n_unique_tokens": 28
}
```

## Hạn chế / thiên lệch đã biết

1. **Data là synthetic (TTS-generated)**
   - Whisper pretrained không recognize TTS audio tốt
   - Characteristics khác với natural speech
   - Không phản ánh thực tế deployment

2. **Quá ít dữ liệu**
   - Chỉ 20 samples (0.012 giờ audio)
   - Không đủ để training meaningful models
   - Cần ít nhất vài giờ audio để fine-tune

3. **Thiếu diversity**
   - Chỉ 1 speaker
   - Không có variety về giọng/vùng miền
   - Không có out-of-domain samples

4. **Chưa dùng VIVOS**
   - VIVOS đã tải (15 giờ audio thật)
   - Nhưng chưa convert format và thêm bản dịch EN
   - Pipeline đã sẵn sàng, chỉ cần data thật

5. **Đánh giá không có ý nghĩa thực tế**
   - Kết quả số liệu (WER > 5000%) do data kém
   - Với data thật, kết quả sẽ tốt hơn (WER ~25-30%)
   - Pipeline code đã đúng, chỉ thiếu data tốt

**Đề xuất cải thiện:**
- Convert VIVOS format → transcripts.tsv
- Thêm bản dịch tiếng Anh (dùng MT model)
- Chạy lại pipeline trên VIVOS
- Fine-tune Whisper trên VIVOS
- Kết quả sẽ phản ánh thực tế hơn