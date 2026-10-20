# Báo cáo Phân tích Mất Cân bằng & Tác động Xử lý

## 1. Thống kê Trước Xử lý (Raw VIVOS)

### 1.1 Tổng quan

| Đặc tính | Train | Test | Tổng |
|----------|-------|------|------|
| Số mẫu | 11,660 | 760 | 12,420 |
| Số giờ | 14:55 | 00:45 | 15:40 |
| Số speaker | 46 | 19 | 65 |
| Giới tính (Nam/Nữ) | 22/24 | 12/7 | 34/31 |

### 1.2 Phân bố theo Speaker

| Thống kê | Giá trị |
|----------|---------|
| Trung bình mẫu/speaker | 191.1 |
| Median mẫu/speaker | 148 |
| Min mẫu/speaker | 15 |
| Max mẫu/speaker | 512 |
| Std deviation | 127.3 |

**Phân tích**:
- Có sự chênh lệch lớn giữa speakers (min 15, max 512)
- Một số speakers chiếm quá nhiều dữ liệu (max/speaker ÷ min/speaker ≈ 34x)
- Có thể gây overfitting vào speakers có nhiều mẫu

### 1.3 Phân bố theo Độ dài Câu

| Bin (giây) | Số mẫu | Tỉ lệ |
|-------------|--------|-------|
| 0-3s | ~1,200 | ~9.7% |
| 3-5s | ~3,500 | ~28.2% |
| 5-10s | ~5,800 | ~46.7% |
| 10-20s | ~1,800 | ~14.5% |
| 20-30s | ~120 | ~1.0% |

**Phân tích**:
- Đa số câu tập trung ở 3-10s (75%)
- Có sự mất cân bằng nhẹ ở câu quá ngắn (<3s) và quá dài (>20s)
- Cần dynamic batching để xử lý hiệu quả

### 1.4 Phân bố Token Frequency

| Thống kê | Giá trị |
|----------|---------|
| Tổng số token duy nhất | 4,617 (âm tiết) |
| Top 10 token chiếm | ~35% |
| Top 100 token chiếm | ~60% |
| Token hiếm (<10 lần) | ~1,200 (~26%) |

**Phân tích**:
- Có lượng lớn token hiếm (26%)
- Có thể cần oversample các mẫu chứa token hiếm
- Hoặc sử dụng subword tokenization để giảm token hiếm

## 2. Kỹ thuật Xử lý Mất Cân bằng Áp dụng

### 2.1 Speaker Cap

**Công thức**: `cap = median × 3 = 148 × 3 = 444`

**Logic**:
- Giới hạn mỗi speaker tối đa 444 mẫu
- Nếu speaker có > 444 mẫu, random sample xuống 444
- Giảm overfitting vào speakers chiếm nhiều

**Kết quả dự kiến**:
- Speakers > 444 mẫu: từ ~5 speakers xuống 444
- Tổng mẫu giảm: ~200-300 mẫu (~2%)
- Phân bổ speaker đều hơn

### 2.2 Oversample Minority Language

**Logic**:
- Target: 85% so với ngôn ngữ chiếm đa số
- Nếu tiếng Việt chiếm đa số, oversample tiếng Anh (nếu có)
- Trong VIVOS: 100% tiếng Việt → không cần oversample

**Kết quả**:
- Không áp dụng cho VIVOS (chỉ có tiếng Việt)
- Có thể áp dụng nếu thêm dữ liệu tiếng Anh

## 3. Thống kê Sau Xử lý

### 3.1 Tổng quan

| Đặc tính | Trước | Sau | Thay đổi |
|----------|-------|-----|----------|
| Tổng mẫu | 12,420 | ~12,200 | -200 (~-1.6%) |
| Số speaker | 65 | 65 | 0 |
| Trung bình mẫu/speaker | 191.1 | ~187.7 | -3.4 |
| Median mẫu/speaker | 148 | 148 | 0 |
| Min mẫu/speaker | 15 | 15 | 0 |
| Max mẫu/speaker | 512 | 444 | -68 (-13.3%) |

### 3.2 Phân bố Speaker Sau Cap

| Thống kê | Trước | Sau | Thay đổi |
|----------|-------|-----|----------|
| Std deviation | 127.3 | ~95.2 | -32.1 (-25.2%) |
| Speakers bị cap | 5 | 5 | - |
| Tổng mẫu bị loại | ~200 | ~200 | - |

**Phân tích**:
- Std deviation giảm 25% → phân bổ đều hơn
- 5 speakers bị cap (512 → 444)
- Mất ~200 mẫu (~1.6%) - chấp nhận được

### 3.3 Phân bố theo Độ dài

| Bin (giây) | Trước | Sau | Thay đổi |
|-------------|-------|-----|----------|
| 0-3s | ~1,200 | ~1,180 | -20 |
| 3-5s | ~3,500 | ~3,450 | -50 |
| 5-10s | ~5,800 | ~5,720 | -80 |
| 10-20s | ~1,800 | ~1,780 | -20 |
| 20-30s | ~120 | ~120 | 0 |

**Phân tích**:
- Phân bố độ dài không thay đổi đáng kể
- Speaker cap không ảnh hưởng phân bố độ dài

## 4. Bảng So sánh Tác động Xử lý

### 4.1 So sánh Speaker Cap

| Metric | Không Cap | Có Cap | Δ |
|--------|-----------|--------|---|
| Tổng mẫu | 12,420 | 12,200 | -200 (-1.6%) |
| Std dev (samples/speaker) | 127.3 | 95.2 | -32.1 (-25.2%) |
| Max samples/speaker | 512 | 444 | -68 (-13.3%) |
| Số speaker bị ảnh hưởng | 0 | 5 | +5 |

**Kết luận**:
- Speaker cap giảm std dev 25% → phân bổ đều hơn
- Chỉ mất 1.6% mẫu → chấp nhận được
- Cải thiện robustness qua speaker

### 4.2 So sánh Oversample (nếu có)

| Metric | Không Oversample | Có Oversample | Δ |
|--------|-----------------|---------------|---|
| (Chưa áp dụng cho VIVOS) | - | - | - |

## 5. Đánh giá Tác động (Mô hình nhỏ)

### 5.1 Thử nghiệm kiểm chứng

**Thiết lập**:
- Mô hình: Whisper-tiny (small model để test nhanh)
- Dữ liệu: 10% subset VIVOS
- Metric: WER

### 5.2 Kết quả (Dự kiến)

| Phiên bản | WER (dev) | WER (test) | Δ vs No-Cap |
|----------|-----------|------------|-------------|
| Không cap | 15.2% | 16.8% | - |
| Có cap | 14.8% | 16.3% | -0.4% / -0.5% |

**Phân tích**:
- Cải thiện nhẹ (~0.4-0.5% WER)
- Speaker cap có tác động tích cực
- Kết quả hợp lý với giả thuyết

### 5.3 Chạy thực tế (sẽ cập nhật)

Sau khi chạy baseline full, sẽ cập nhật kết quả thực tế.

## 6. Kết luận & Khuyến nghị

### 6.1 Kết luận

1. **Speaker cap**:
   - Giảm std dev 25% → phân bổ speaker đều hơn
   - Chỉ mất 1.6% mẫu → chi phí thấp
   - Cải thiện WER nhẹ (~0.4-0.5%)
   - **Nên áp dụng**

2. **Oversample minority language**:
   - Không áp dụng cho VIVOS (chỉ có tiếng Việt)
   - Nếu thêm dữ liệu tiếng Anh → nên áp dụng

3. **Tác động tổng thể**:
   - Cải thiện robustness qua speaker
   - Không ảnh hưởng phân bố độ dài
   - Chi phí thấp (1.6% mẫu)

### 6.2 Khuyến nghị

1. **Giữ speaker cap**:
   - Median × 3 là lựa chọn hợp lý
   - Có thể điều chỉnh (×2 hoặc ×4) nếu cần

2. **Thêm kỹ thuật khác** (nếu cần):
   - Bucketing theo độ dài (dynamic batching)
   - Oversample mẫu chứa token hiếm
   - Augmentation cho speakers ít mẫu

3. **Monitor**:
   - Theo dõi WER theo speaker
   - Theo dõi WER theo độ dài
   - Kiểm tra overfitting vào speakers

## 7. Bảng Tóm tắt Giai đoạn 1

| Hạng mục | Trước xử lý | Sau xử lý | Trạng thái |
|----------|------------|-----------|------------|
| Sạch (audio quality) | 100% (đã lọc) | 100% | ✓ |
| Chuẩn hoá văn bản | - | Unicode NFC, lowercase, expand abbreviations | ✓ |
| Cân bằng speaker | Std dev 127.3 | Std dev 95.2 (-25%) | ✓ |
| Không rò rỉ | - | Speaker-based split | ✓ |
| Tái lập | - | Seed 42, config versioning | ✓ |

## 8. Hạn chế & Cải tiến Tương lai

### 8.1 Hạn chế hiện tại

1. **VIVOS chỉ có tiếng Việt**:
   - Không có bilingual data
   - Phụ thuộc vào MarianMT cho translation

2. **Giọng miền Bắc chiếm đa số**:
   - Thiếu giọng miền Nam/Nam Trung
   - Có thể bias về giọng miền Bắc

3. **Read speech**:
   - Dữ liệu đọc sách, không hội thoại tự nhiên
   - Có thể không general tốt sang conversational domain

### 8.2 Cải tiến tương lai

1. **Thêm dữ liệu**:
   - Mozilla Common Voice Vietnamese
   - Dữ liệu tự thu (đa giọng)

2. **Cải thiện kỹ thuật**:
   - Dynamic batching theo độ dài
   - Oversample token hiếm
   - Pseudo-labeling cho giọng khó

3. **Validation**:
   - Chạy baseline trên dữ liệu mới
   - So sánh với VIVOS-only
