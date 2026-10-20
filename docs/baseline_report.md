# Baseline Evaluation Report - Task 2

## Thông tin chung
- **Date**: 2026-10-20
- **Model**: Whisper-small (zero-shot, no fine-tune) + MarianMT cascade
- **Dataset**: Synthetic sample data (20 samples, TTS-generated)
- **Device**: CUDA

## Kết quả Zero-shot Baseline

| Metric | Value | Ghi chú |
|--------|-------|---------|
| ASR WER | 5130.77% | Rất cao do data synthetic |
| ASR CER | 5414.29% | Rất cao do data synthetic |
| MT BLEU (cascade) | 0.0000 | Tệ do ASR lỗi hoàn toàn |
| MT chrF++ (cascade) | 0.3033 | Tệ do ASR lỗi hoàn toàn |
| MT BLEU (oracle) | 25.0000 | Dịch từ transcript chuẩn |
| RTF | 7.8038 | Chậm do GPU có thể chưa tối ưu |
| Error propagation | 25.0000 BLEU | Chênh lệch giữa oracle và cascade |

## Phân tích & Gap Analysis

### 1. Dữ liệu (Yếu tố chính)
- **Dùng**: Synthetic sample data (được tạo bằng TTS)
- **Vấn đề**: Whisper pretrained không hoạt động tốt với audio TTS
- **So sánh với paper**: Paper dùng VIVOS/VoiceCommonVoice (audio thật)
- **Kết luận**: **Không thể so sánh công bằng với paper** do khác biệt lớn về data

### 2. ASR Performance
- Whisper zero-shot không recognize audio TTS tốt
- WER > 5000% → model không hoạt động trên domain này
- **Giải pháp**: Cần dùng audio thật (VIVOS, Common Voice)

### 3. MT Performance
- Oracle BLEU = 25% → MarianMT hoạt động khi input đúng
- Cascade BLEU = 0% → Lỗi ASR lan truyền hoàn toàn sang MT
- **Error propagation**: 25 BLEU điểm

### 4. Limitations hiện tại
1. Dữ liệu mẫu (synthetic) không phản ánh thực tế
2. Chưa fine-tune Whisper trên tiếng Việt
3. Chưa có data train/dev/test với audio thật
4. VIVOS đã tải nhưng chưa convert format + thiếu bản dịch EN

## Bước tiếp theo

### Lựa chọn A: Sử dụng VIVOS (Khuyên dùng)
1. Convert VIVOS format → transcripts.tsv
2. Thêm bản dịch tiếng Anh (dùng MT model hoặc service)
3. Chạy lại pipeline data
4. Chạy evaluation zero-shot
5. Fine-tune Whisper trên VIVOS

### Lựa chọn B: Dữ liệu hiện tại (Demo nhanh)
- Giữ nguyên synthetic data
- Trong báo cáo ghi rõ: "Data là demo, kết quả không phản ánh thực tế"
- Focus vào pipeline và code structure
- Gap analysis giải thích do data kém

## Sản phẩm hiện có
- ✅ Dataset pipeline code
- ✅ Zero-shot evaluation script
- ✅ Baseline config
- ✅ Báo cáo kết quả (với giải thích)
- ⏳ Chưa có kết quả training thật (cần data thật)

## Đề xuất cho Task 2
Để hoàn thành Task 2 với kết quả có ý nghĩa:
1. **Cần dataset thật** (VIVOS hoặc Common Voice VN)
2. Fine-tune Whisper trên data train
3. Đánh giá trên test set fixed
4. Thực nghiệm V1-V4 với data thật
5. Gap analysis có cơ sở thực nghiệm

Nếu tài nguyên hạn chế, có thể dùng subset nhỏ của VIVOS (ví dụ 1-2 giờ) để chạy nhanh.
