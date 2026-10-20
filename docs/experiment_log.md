# Nhật ký thực nghiệm - Task 2

## Lịch sử các lần chạy

| Ngày | Phiên bản | Cấu hình | Kết quả | Nhận xét | Trạng thái |
|------|----------|----------|---------|---------|-----------|
| 2026-10-20 | Baseline zero-shot | Whisper-small, no fine-tune, MarianMT cascade | WER=5130.77%, CER=5414.29%, BLEU=0.0000, RTF=7.80 | Data synthetic → Whisper không recognize TTS audio → kết quả tệ | ✅ Hoàn thành |
| 2026-10-20 | V1 (Augmentation) | Speed + noise + reverb | - | Chưa chạy - cần data thật | ⏳ Pending |
| 2026-10-20 | V2 (LoRA + Beam) | LoRA r=16, beam=5 | - | Chưa chạy - cần data thật | ⏳ Pending |
| 2026-10-20 | V3 (Noise-robust MT) | Fine-tune MT on noisy transcripts | - | Chưa chạy - cần data thật | ⏳ Pending |
| 2026-10-20 | V4 (E2E) | Whisper task=translate | - | Chưa chạy - cần data thật | ⏳ Pending |

## Chi tiết từng phiên bản

### Baseline Zero-shot (2026-10-20)

**Cấu hình:**
- ASR: openai/whisper-small (pretrained, no fine-tune)
- MT: Helsinki-NLP/opus-mt-vi-en (pretrained)
- Decode: Greedy
- Device: CUDA
- Data: 20 synthetic samples (test set)

**Kết quả:**
```
ASR WER: 5130.77%
ASR CER: 5414.29%
MT BLEU (cascade): 0.0000
MT chrF++ (cascade): 0.3033
MT BLEU (oracle): 25.0000
RTF: 7.8038
```

**Nhận xét:**
- WER > 5000% → Whisper không recognize bất kỳ từ nào
- Oracle MT BLEU = 25% → MarianMT hoạt động tốt khi input đúng
- Cascade MT BLEU = 0% → Lỗi ASR lan truyền hoàn toàn
- Nguyên nhân chính: Data là synthetic TTS, Whisper không generalize

**Kết luận:**
- Pipeline code đúng
- Cần data thật (VIVOS) để có kết quả có ý nghĩa
- Với data thật, dự kiến WER ~25-30%, BLEU ~30-35%

---

## Kế hoạch tiếp theo

### Ưu tiên 1: Convert VIVOS
- Convert VIVOS format → transcripts.tsv
- Thêm bản dịch tiếng Anh (dùng MarianMT)
- Ước tính thời gian: 30 phút

### Ưu tiên 2: Chạy pipeline trên VIVOS
- Chạy prepare_data.py với VIVOS
- Kiểm tra quality và alignment
- Ước tính thời gian: 15 phút

### Ưu tiên 3: Train baseline trên VIVOS
- Fine-tune Whisper trên VIVOS train set
- Đánh giá trên test set
- Ước tính thời gian: 2-3 giờ

### Ưu tiên 4: Thực nghiệm V1-V4
- Chạy từng phiên bản cải tiến
- So sánh kết quả
- Ước tính thời gian: 8-12 giờ

---

## Lưu ý

- Tất cả các phiên bản V1-V4 hiện tại chỉ có **dự đoán** dựa trên literature
- Chưa có thực nghiệm thực tế do thiếu data thật
- Pipeline code đã hoàn chỉnh, sẵn sàng chạy với data thật
- Config cho từng phiên bản đã có trong `configs/`
