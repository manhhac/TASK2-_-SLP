# Báo cáo Phân tích Mất Cân bằng & Tác động Xử lý

> Số liệu trong tài liệu này đo trực tiếp từ `data/raw/vietnamese/vivos/transcripts.tsv`
> (raw) và `data/manifests/{train,dev,test}.jsonl` (sau xử lý). Không dùng số ước lượng.

## 1. Thống kê Trước Xử lý (Raw VIVOS)

### 1.1 Tổng quan

| Đặc tính | Train | Test | Tổng |
|----------|-------|------|------|
| Số mẫu | 11.660 | 760 | 12.420 |
| Số speaker | 46 | 19 | 65 |
| Giới tính (Nam/Nữ) | 22/24 | 12/7 | 34/31 |

### 1.2 Phân bố theo Speaker (đo từ transcripts.tsv)

| Thống kê | Giá trị |
|----------|---------|
| Trung bình mẫu/speaker | 191,1 |
| Median mẫu/speaker | 250 |
| Min mẫu/speaker | 40 |
| Max mẫu/speaker | 300 |
| Std deviation | 107,3 |

**Phân tích:**
- Chênh lệch giữa speaker ít nhất và nhiều nhất là 40 → 300 mẫu (≈ 7,5×).
- Phần lớn speaker có 200–300 mẫu; nhóm 19 speaker thuộc tập dev/test chỉ có 34–40 mẫu.

### 1.3 Phân bố theo Độ dài Câu (sau làm sạch, 10.887 mẫu)

| Bin (giây) | Số mẫu | Tỉ lệ |
|-------------|--------|-------|
| (0, 3] | 2.264 | 20,8% |
| (3, 5] | 5.054 | 46,4% |
| (5, 10] | 3.163 | 29,1% |
| (10, 20] | 406 | 3,7% |
| (20, 30] | 0 | 0% |

**Phân tích:**
- Đa số câu tập trung ở 3–5s và 5–10s (75,5%).
- Không có câu > 20s → không cần lo ngại câu quá dài.

### 1.4 Phân bố Token Frequency (sau làm sạch)

| Thống kê | Giá trị |
|----------|---------|
| Tổng token | 150.100 |
| Token duy nhất (âm tiết) | 4.940 |
| Top 10 token chiếm | 10,2% |
| Token hiếm (< 10 lần) | 3.257 (65,9% số token duy nhất) |

**Phân tích:**
- Phân bố long-tail rõ: 65,9% token duy nhất xuất hiện < 10 lần.
- Nên dùng subword tokenization (Whisper BPE) để giảm ảnh hưởng token hiếm.

## 2. Kỹ thuật Xử lý Mất Cân bằng Áp dụng

### 2.1 Speaker Cap

**Công thức** (trong `src/data/prepare_data.py::apply_imbalance_fixes`):
`cap = median(số mẫu/speaker trên tập train) × 3 = 200 × 3 = 600`

**Logic:**
- Mỗi speaker giữ tối đa 600 mẫu; speaker nào vượt cap sẽ random sample xuống ngưỡng.
- Mục đích: giảm overfitting vào speaker có nhiều mẫu.

**Kết quả thực đo:**
- Speaker nhiều mẫu nhất trong tập train = 300 (< 600).
- → **0 speaker bị cap, 0 mẫu bị loại.**

### 2.2 Oversample Minority Language

**Logic:** nếu tập train có ≥ 2 ngôn ngữ, oversample ngôn ngữ ít mẫu lên 85% ngôn ngữ đa số.

**Kết quả thực đo:**
- VIVOS 100% tiếng Việt → điều kiện `len(value_counts) >= 2` **không thoả** → kỹ thuật không kích hoạt.

## 3. Thống kê Sau Xử lý

### 3.1 Tổng quan

| Đặc tính | Trước cap | Sau cap | Thay đổi |
|----------|-----------|---------|----------|
| Tổng mẫu (sau làm sạch + dedup) | 10.887 | 10.887 | 0 |
| Số speaker | 65 | 65 | 0 |
| Trung bình mẫu/speaker | 167,5 | 167,5 | 0 |
| Median mẫu/speaker | 200 | 200 | 0 |
| Min mẫu/speaker | 34 | 34 | 0 |
| Max mẫu/speaker | 300 | 300 | 0 |

### 3.2 Phân bố Speaker Sau Cap

| Thống kê | Trước | Sau | Thay đổi |
|----------|-------|-----|----------|
| Std deviation | 97,1 | 97,1 | 0 |
| Speakers bị cap | 0 | 0 | - |
| Tổng mẫu bị loại | 0 | 0 | - |

**Phân tích trung thực:** Với VIVOS, ngưỡng cap (600) cao hơn hẳn speaker nhiều mẫu nhất (300)
nên kỹ thuật **không tạo khác biệt đo được**. Giá trị thực của Giai đoạn 1 nằm ở khâu làm sạch:
loại 1.524 mẫu trùng lặp (MD5 audio/text), lọc 9 mẫu lỗi (5 silent + 4 alignment),
và chia tập theo speaker chống rò rỉ (train 47 / dev 9 / test 9 speaker).

### 3.3 Phân bố theo Độ dài

Không thay đổi (cap không cắt mẫu nào) → giữ nguyên như Mục 1.3.

## 4. Bảng So sánh Tác động Xử lý

### 4.1 So sánh Speaker Cap

| Metric | Không Cap | Có Cap | Δ |
|--------|-----------|--------|---|
| Tổng mẫu | 10.887 | 10.887 | 0 |
| Std dev (samples/speaker) | 97,1 | 97,1 | 0 |
| Max samples/speaker | 300 | 300 | 0 |
| Số speaker bị ảnh hưởng | 0 | 0 | 0 |

**Kết luận:** Speaker cap đã được cài đặt và chạy, nhưng **không tác động** trên VIVOS do
ngưỡng cap quá cao. Đã ghi nhận trung thực thay vì báo cáo cải thiện không có thật.

### 4.2 So sánh Oversample

| Metric | Không Oversample | Có Oversample | Δ |
|--------|-----------------|---------------|---|
| - | Không áp dụng cho VIVOS (một ngôn ngữ) | - | - |

## 5. Đánh giá Tác động

### 5.1 Thử nghiệm kiểm chứng

Các phiên bản V1→V4 trong `docs/version_progress.csv` không tách riêng biến "cap bật/tắt",
nên **không có bằng chứng thực nghiệm** rằng cap cải thiện WER trên bộ này. Không đưa ra
con số cải thiện giả định.

### 5.2 Kết luận về tái lấy mẫu

- Speaker cap: không tác động (0 mẫu bị loại).
- Oversample minority language: không kích hoạt (đơn ngữ).
- Hai hạn chế không thể khắc phục bằng tái lấy mẫu: vùng miền 100% Bắc và độ dài câu.

## 6. Kết luận & Khuyến nghị

1. **Speaker cap**: giữ cấu hình, nhưng với VIVOS nên hạ ngưỡng (ví dụ `median × 1,2 ≈ 240`)
   nếu thực sự muốn tác động tới các speaker nhiều mẫu.
2. **Oversample**: chỉ hữu ích khi bổ sung dữ liệu ngôn ngữ thứ hai.
3. **Giá trị thật của Giai đoạn 1**: dedup (1.524 mẫu), lọc lỗi (9 mẫu), chuẩn hoá NFC,
   chia tập theo speaker chống rò rỉ.

## 7. Bảng Tóm tắt Giai đoạn 1

| Hạng mục | Trước xử lý | Sau xử lý | Trạng thái |
|----------|------------|-----------|------------|
| Sạch (audio quality) | 12.420 | 12.411 (loại 5 silent + 4 alignment) | Đạt |
| Loại trùng lặp | 12.411 | 10.887 (loại 1.524) | Đạt |
| Chuẩn hoá văn bản | - | Unicode NFC, lowercase, expand abbreviations | Đạt |
| Cân bằng speaker (cap) | std 97,1 | std 97,1 (không cắt mẫu) | Không tác động |
| Không rò rỉ | - | Chia theo speaker (47/9/9) | Đạt |
| Tái lập | - | Seed 42, config versioning | Đạt |

## 8. Hạn chế & Cải tiến Tương lai

### 8.1 Hạn chế hiện tại

1. **VIVOS chỉ có tiếng Việt**: không có dữ liệu song ngữ; phụ thuộc MarianMT để tạo nhãn EN.
2. **100% giọng miền Bắc**: thiếu giọng miền Nam/Nam Trung → nguy cơ bias vùng miền.
3. **Read speech**: dữ liệu đọc sách, không phải hội thoại tự nhiên.

### 8.2 Cải tiến tương lai

1. **Thêm dữ liệu**: Mozilla Common Voice Vietnamese, dữ liệu tự thu đa giọng.
2. **Kỹ thuật**: dynamic batching theo độ dài, oversample token hiếm, pseudo-labeling.
3. **Validate**: chạy baseline trên dữ liệu mới và so sánh với VIVOS-only.
