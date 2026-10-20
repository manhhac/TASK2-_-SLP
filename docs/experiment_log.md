# Nhật ký thực nghiệm — Task 2

> Toàn bộ số liệu dưới đây lấy từ **audio thật VIVOS** (bộ quick: 100 train / 20 dev / 20 test) và file `outputs/*/seed_42/*.json`. Seed cố định **42**. Phần cứng: **1× GTX 1070 8GB**, fp16 tắt.

## Lịch sử các lần chạy

| Ngày | Phiên bản | Cấu hình | WER | CER | BLEU | chrF++ | RTF | Trạng thái |
|------|-----------|----------|-----|-----|------|--------|-----|-----------|
| 2026-10-20 | Zero-shot (synthetic) | Whisper-small không fine-tune + MarianMT, 4 mẫu sine | 5130,77 | 5414,29 | 0,00 | 0,30 | 7,80 | ✅ Kiểm chứng pipeline |
| 2026-10-20 | Baseline (VIVOS quick) | Fine-tune Whisper-small, không augment, greedy | 50,52 | 22,83 | 4,83 | 25,52 | 0,23 | ✅ Hoàn thành |
| 2026-10-20 | V1 — Augmentation | + speed perturbation, Gaussian noise, reverb + speaker cap | — | — | — | — | — | ⚠️ Đã train, **chưa đánh giá** |
| 2026-10-20 | V2 — LoRA + Beam | LoRA r=16/alpha=32 + beam size 5 + augment | 41,24 | 16,05 | 22,01 | 42,98 | 0,27 | ✅ Hoàn thành |
| 2026-10-20 | V3 — Noise-robust MT | Fine-tune MT trên transcript ASR nhiễu + LoRA + beam | 43,30 | 16,88 | **22,73** | **43,62** | 0,25 | ✅ Hoàn thành |
| 2026-10-20 | V4 — End-to-end | Whisper `task=translate` (VI→EN trực tiếp), bỏ MT | — | — | 4,23 | 17,21 | 0,18 | ✅ Hoàn thành (âm tính) |

Nguồn số liệu: `outputs/zero_shot_baseline.json`, `outputs/quick/seed_42/test_metrics.json`, `outputs/v2_quick/seed_42/test_metrics.json`, `outputs/v3_quick/seed_42/test_metrics.json`, `outputs/v4_quick/seed_42/test_metrics_e2e.json`.

---

## Chi tiết từng phiên bản

### Zero-shot trên synthetic — chỉ để kiểm chứng pipeline (2026-10-20)

**Cấu hình:** Whisper-small (pretrained, không fine-tune) + MarianMT (pretrained); decode greedy; 4 mẫu sine wave tổng hợp.

```
ASR WER  : 5130,77%
ASR CER  : 5414,29%
MT BLEU (cascade) : 0,00
MT chrF++(cascade): 0,30
MT BLEU (oracle)  : 25,00
RTF : 7,80   (audio 12,0s / tổng 93,65s)
```

**Nhận xét:** WER > 5000% vì đầu vào là sine wave, **không phải speech người** → không đại diện năng lực mô hình. Oracle BLEU = 25,00 (dịch từ transcript chuẩn) chứng minh **MarianMT chạy tốt** và lỗi đến **100% từ ASR**. Chỉ dùng để xác nhận pipeline (load → transcribe → dịch → tính metric) đúng.

### Baseline — fine-tune trên VIVOS quick (2026-10-20)

**Cấu hình:** `configs/quick.yaml` — Whisper-small fine-tune, không augment, greedy decode; 2 epochs, batch 2, grad-accum 4, lr 1e-4.

```
ASR WER  : 50,52%
ASR CER  : 22,83%
MT BLEU (cascade) : 4,83
MT chrF++(cascade): 25,52
MT BLEU (oracle)  : 100,00   (*)
RTF : 0,23   (audio 72,97s / tổng 16,49s)
```

**Nhận xét:** Fine-tune trên audio thật đưa WER từ "không hoạt động" (>5000%) về **50,52%** — còn cao nhưng có nghĩa với chỉ 100 mẫu train. BLEU cascade 4,83 rất thấp do **lỗi ASR lan truyền** sang MT.

(*) **Cảnh báo:** oracle BLEU = 100 vì nhãn EN của bộ quick do **chính MarianMT sinh** → phép so sánh vòng, **không dùng** để đánh giá MT.

### V1 — Augmentation (2026-10-20)

**Cấu hình:** `configs/v1_augment.yaml` — thêm speed perturbation + Gaussian noise + reverb (train only) và speaker cap.

**Trạng thái:** đã huấn luyện nhưng **chưa chạy script đánh giá** → chưa có metric. Ghi nhận là việc cần bổ sung (xem báo cáo Mục 5.2).

### V2 — LoRA + Beam + Augment (2026-10-20)

**Cấu hình:** `configs/v2_lora.yaml` — LoRA r=16, alpha=32, target [q,v,k,out,fc1,fc2]; beam search size 5; augment; lr 5e-5.

```
ASR WER  : 41,24%   (giảm -18,4% tương đối so với baseline)
ASR CER  : 16,05%
MT BLEU (cascade) : 22,01   (+356% so với baseline)
MT chrF++(cascade): 42,98
RTF : 0,27   (audio 72,97s / tổng 19,77s)
```

**Nhận xét:** **WER tốt nhất** trong các phiên bản; gói LoRA + beam + augmentation phát huy hiệu quả rõ rệt.

### V3 — Noise-robust MT + LoRA + Beam (2026-10-20)

**Cấu hình:** `configs/v3_noise_robust_mt.yaml` — fine-tune MT trên transcript ASR nhiễu + LoRA + beam; lr 3e-5.

```
ASR WER  : 43,30%
ASR CER  : 16,88%
MT BLEU (cascade) : 22,73   ← cao nhất
MT chrF++(cascade): 43,62   ← cao nhất
RTF : 0,25   (audio 72,97s / tổng 18,23s)
```

**Nhận xét:** **BLEU/chrF++ tốt nhất** → **V3 là phiên bản được chọn**. Tuy nhiên WER nhích cao hơn V2 (43,30 vs 41,24) và chênh BLEU chỉ +0,72 (chưa kiểm định ý nghĩa thống kê, chỉ 1 seed).

### V4 — End-to-end Whisper translate (2026-10-20)

**Cấu hình:** `configs/v4_e2e.yaml` — Whisper `task=translate`, language `en`, bỏ nhánh MT.

```
MT BLEU (E2E)  : 4,23
MT chrF++ (E2E): 17,21
RTF : 0,18   (audio 72,97s / tổng 12,88s)
```

**Nhận xét:** **Thất bại rõ rệt** — BLEU 4,23 kém xa cascade (22,0-22,7). Đây là **kết quả âm tính có giá trị**: trong điều kiện dữ liệu nhỏ, **cascade ưu thế hơn end-to-end**.

---

## Kết luận

- **Phiên bản tốt nhất: V3** (WER 43,30 / BLEU 22,73 / chrF++ 43,62), kế đó là **V2** (WER tốt nhất 41,24).
- Cải tiến chủ yếu đến từ **gói LoRA + beam search + augmentation** ở nhánh ASR/decode; **noise-robust MT chỉ cải thiện biên**.
- **RTF tất cả < 1** (0,18-0,27) → đạt yêu cầu gần thời gian thực.
- Hạn chế đã khai báo: tập test chỉ 20 mẫu, 1 seed, chưa ablation tách rời, nguy cơ overfit test (xem báo cáo Mục 5).
