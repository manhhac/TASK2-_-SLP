# Data Card — Task 2 ASR VI + MT EN

## 1. Nguồn dữ liệu

### 1.1 VIVOS (Vietnamese Speech Corpus)
- **Nguồn**: AILAB, VNUHCM - University of Science
- **Paper**: "A non-expert Kaldi recipe for Vietnamese Speech Recognition System", WLSI/OIAF4HLT 2016
- **Link**: https://zenodo.org/record/7068130
- **License**: Creative Commons Attribution NonCommercial ShareAlike v4.0 (CC BY-NC-SA 4.0)
- **Giấy phép**: Chỉ dùng cho mục đích học thuật, không thương mại

### 1.2 Thống kê VIVOS

| Đặc tính | Train | Test | Tổng |
|----------|-------|------|------|
| Số người nói | 46 | 19 | 65 |
| Nam | 22 | 12 | 34 |
| Nữ | 24 | 7 | 31 |
| Số utterances | 11,660 | 760 | 12,420 |
| Thời lượng | 14:55 | 00:45 | 15:40 |
| Âm tiết duy nhất | 4,617 | 1,692 | - |

### 1.3 Môi trường ghi âm
- Ghi trong phòng yên tĩnh với micro chất lượng cao
- Người đọc được yêu cầu đọc từng câu một
- Sample rate: 16 kHz
- Mono channel

### 1.4 Dữ liệu dịch tiếng Anh
- VIVOS chỉ có transcript tiếng Việt
- Sử dụng MarianMT (opus-mt-vi-en) để tạo bản dịch tiếng Anh cho training MT
- Sẽ đánh giá error propagation từ ASR → MT

## 2. Quy trình xử lý dữ liệu

### 2.1 Pipeline tổng quan

```
Raw Data (VIVOS)
    ↓
1. Thu thập & hợp nhất
    ↓
2. Làm sạch audio
    ↓
3. Làm sạch & chuẩn hoá văn bản
    ↓
4. Kiểm tra căn chỉnh audio–text
    ↓
5. Kiểm tra bản dịch EN
    ↓
6. Loại trùng & chống rò rỉ
    ↓
7. Phân tích mất cân bằng
    ↓
8. Chia tập theo speaker
    ↓
9. Xử lý mất cân bằng
    ↓
10. Data augmentation (train only)
    ↓
Manifests (train/dev/test.jsonl)
```

### 2.2 Quy tắc chuẩn hoá văn bản

#### 2.2.1 Unicode & Lowercase
1. **Unicode NFC**: Chuẩn hoá Unicode Normalization Form C (quan trọng với tiếng Việt)
2. **Lowercase**: Chuyển toàn bộ sang chữ thường

#### 2.2.2 Xử lý ký tự
3. **Loại bỏ ký tự điều khiển**: Bỏ các control characters
4. **Giữ ký tự cho phép**:
   - Chữ cái tiếng Việt (a-z, à-ỹ, đ)
   - ASCII
   - Dấu câu: . , ! ? ; : ' " ( ) -
5. **Xóa ký tự lạ**: Các ký tự không thuộc bảng chữ Việt/ASCII

#### 2.2.3 Xử lý số
6. **Số**: Mặc định tách từng chữ số (e.g., "123" → "1 2 3")
   - Có thể bật `number_to_words` để đọc thành lời (e.g., "123" → "một trăm hai mươi ba")

#### 2.2.4 Từ viết tắt
7. **Mở rộng từ viết tắt phổ biến**:
   - tp./tp → thành phố
   - tw./tw → thị xã
   - q./q → quận
   - p./p → phường
   - nxb → nhà xuất bản
   - vn → việt nam
   - hcm → hồ chí minh
   - hn → hà nội
   - đt → điện thoại
   - ko/k → không
   - dc/đc → được
   - ntn → như thế nào
   - vs → với
   - tv → thành viên

#### 2.2.5 Khoảng trắng & Dấu câu
8. **Chuẩn hoá khoảng trắng**: 1 space, strip 2 đầu
9. **Gộp dấu câu lặp**: !!! → !, ... → .

### 2.3 Làm sạch audio

#### 2.3.1 Bộ lọc chất lượng
- **Thời lượng**: 0.5s ≤ duration ≤ 30s
- **RMS**: ≥ 0.005 (loại im lặng toàn phần)
- **Clipping**: ≤ 20% samples vượt ngưỡng 0.99
- **Khả năng giải mã**: Loại file không đọc được

#### 2.3.2 Chuẩn hoá biên độ
- Peak normalization: chia max và nhân 0.95
- Tránh clipping

### 2.4 Kiểm tra căn chỉnh audio–text

#### 2.4.1 Tốc độ nói (characters per second)
- **Ngưỡng**: 2.0 ≤ cps ≤ 25.0
- Cặp lệch nhãn: quá nhanh/quá chậm so với transcript

#### 2.4.2 Alignment check
- So sánh độ dài transcript với duration
- Loại cặp không hợp lý

### 2.5 Kiểm tra bản dịch EN

#### 2.5.1 Tỉ lệ độ dài
- **Ngưỡng**: 0.3 ≤ len(EN)/len(VI) ≤ 2.5
- Loại cặp quá ngắn/quá dài

#### 2.5.2 Overlap check
- Nếu EN trùng VI > 80% token → coi như chưa dịch
- Loại cặp dịch lỗi

### 2.6 Loại trùng & chống rò rỉ

#### 2.6.1 Hash-based deduplication
- **Audio hash**: MD5 of audio bytes
- **Text hash**: MD5 of transcript UTF-8
- Loại trùng lặp cả audio và text

#### 2.6.2 Speaker-based split
- Chia train/dev/test theo speaker_id
- Stratified split theo language + source
- Đảm bảo không trùng speaker giữa các split

### 2.7 Phân tích mất cân bằng

#### 2.7.1 Các chiều phân tích
- **Language**: tiếng Việt vs tiếng Anh
- **Source**: nguồn dữ liệu
- **Gender**: nam/nữ
- **Region**: vùng miền
- **Duration**: bins: 0-3s, 3-5s, 5-10s, 10-20s, 20-30s
- **Speaker**: phân bố số mẫu mỗi speaker
- **Token frequency**: tần suất từ

### 2.8 Xử lý mất cân bằng

#### 2.8.1 Speaker cap
- **Công thức**: cap = median × 3
- Giới hạn số mẫu mỗi speaker
- Tránh speaker chiếm quá nhiều

#### 2.8.2 Oversample minority language
- **Target**: 85% so với ngôn ngữ chiếm đa số
- Oversample ngôn ngữ ít mẫu hơn
- Đánh dấu `is_augmented = True`

### 2.9 Chia tập dữ liệu

#### 2.9.1 Tỉ lệ
- Train: 70%
- Dev: 15%
- Test: 15%

#### 2.9.2 Phương pháp
- StratifiedGroupKFold theo speaker
- Stratify theo language + source
- Group theo speaker_id (không lọt speaker)

### 2.10 Data Augmentation (Train only)

#### 2.10.1 Waveform augmentation
- **Speed perturbation**: 0.9, 1.0, 1.1 (p=0.35)
- **Gaussian noise**: SNR 10-20dB (p=0.25)
- **Reverb**: RT60 0.2-0.5s (p=0.10)

#### 2.10.2 SpecAugment
- **Freq mask**: n=2, max=27 bands
- **Time mask**: n=2, max=100 frames

### 2.11 Tiền tính đặc trưng

#### 2.11.1 Log-Mel Spectrogram
- **Sample rate**: 16 kHz
- **n_mels**: 80
- **FFT size**: 512
- **Hop length**: 160
- **Win length**: 400

#### 2.11.2 Tokenizer
- **ASR**: Whisper tokenizer (built-in)
- **MT**: MarianTokenizer (built-in with opus-mt-vi-en)

## 3. Thống kê dữ liệu

### 3.1 Trước xử lý (Raw VIVOS)

| Đặc tính | Giá trị |
|----------|---------|
| Tổng mẫu | 12,420 |
| Tổng giờ | 15:40 |
| Số speaker | 65 |
| Speaker trung bình | ~191 mẫu/speaker |
| Độ dài trung bình | ~4.5s |

### 3.2 Sau xử lý (Final Dataset)

| Split | Số mẫu | Số giờ | Số speaker |
|-------|--------|--------|------------|
| Train | ~8,200 | ~10:00 | ~45 |
| Dev | ~1,750 | ~2:00 | ~10 |
| Test | ~1,750 | ~2:00 | ~10 |

*(Số liệu dự kiến, sẽ cập nhật sau khi chạy pipeline)*

### 3.3 Phân bổ theo speaker

- **Speaker cap**: median × 3
- **Speaker range**: min ~50, max ~573 mẫu

### 3.4 Phân bổ theo độ dài

| Bin | Train | Dev | Test |
|-----|-------|-----|------|
| 0-3s | - | - | - |
| 3-5s | - | - | - |
| 5-10s | - | - | - |
| 10-20s | - | - | - |
| 20-30s | - | - | - |

*(Sẽ cập nhật sau khi chạy pipeline)*

## 4. Hạn chế & Thiên lệch đã biết

### 4.1 Domain mismatch
- Dữ liệu đọc sách (read speech) vs hội thoại thực tế (conversational speech)
- Có thể ảnh hưởng đến performance trên domain khác

### 4.2 Thiếu test OOD
- Không có test set cho giọng vùng miền chưa có
- Không có test cho môi trường nhiễu
- Không có test cho hội thoại tự nhiên

### 4.3 Mất cân bằng vẫn tồn tại
- Sau speaker cap và oversample, vẫn có sự lệch nhẹ
- Giọng miền Nam/Nam Trung ít hơn giọng miền Bắc

### 4.4 Dữ liệu dịch tiếng Anh
- Không có bản dịch tiếng Anh gốc từ VIVOS
- Sử dụng MarianMT để tạo translation (có thể có lỗi)
- Error propagation từ ASR → MT cần đánh giá

### 4.5 Vùng miền
- VIVOS chủ yếu là giọng miền Bắc (Hà Nội)
- Thiếu giọng miền Nam (Sài Gòn) và miền Trung

## 5. Versioning & Tái lập

### 5.1 Script tái lập
```bash
python -m src.data.prepare_data --config configs/baseline.yaml
```

### 5.2 Config
- **File**: `configs/baseline.yaml`
- **Seed**: 42 (đảm bảo tái lập)
- **Random seed**: áp dụng cho numpy, random, torch

### 5.3 Outputs
- **Manifests**: `data/manifests/{train,dev,test}.jsonl`
- **Report**: `data/manifests/pipeline_report.json`
- **Data Card**: `data/manifests/data_card.md`
- **Feature stats**: `data/features/feature_stats.json`

### 5.4 Version data
- **Hash**: MD5 của toàn bộ dataset
- **Date**: ngày tạo
- **Config version**: hash của config file

## 6. Chất lượng dữ liệu

### 6.1 Tỉ lệ lỗi (Target ≤ 5%)
- **Alignment errors**: < 2%
- **Translation errors**: < 3%
- **Audio quality issues**: < 1%

### 6.2 Kiểm tra thủ công
- Lấy mẫu 100 cặp ngẫu nhiên
- Kiểm tra transcript chính xác
- Kiểm tra translation chất lượng
- Báo cáo tỉ lệ lỗi thủ công

## 7. Tài liệu tham khảo

1. Luong, H.-T., & Vu, H.-Q. (2016). A non-expert Kaldi recipe for Vietnamese Speech Recognition System. In Proc. WLSI/OIAF4HLT 2016.
2. Radford, A., et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. OpenAI.
3. Tiedemann, J., & Thottingal, E. (2020). OPUS-MT — Building massively parallel multilingual machine translation models. In Proc. EMNLP 2020.

## 8. Liên hệ & Cập nhật

- **Người tạo**: Group members
- **Ngày tạo**: 2026-10-19
- **Cập nhật cuối**: 2026-10-19
- **Version**: 1.0
