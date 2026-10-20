# Nội dung Slide thuyết trình — Task 2
## Hệ thống ASR tiếng Việt kèm dịch sang tiếng Anh

**Thời lượng:** 10–12 phút trình bày + 5 phút Q&A
**Thành viên:** Nguyễn Đức Mạnh — SE194525
**Repo:** https://github.com/manhhac/TASK2-_-SLP

> Ghi chú: đây là **nội dung từng slide** (để dựng PowerPoint/Google Slides). Mỗi mục `### Slide N` = 1 slide. Phần *Lời dẫn* là gợi ý nói.

---

### Slide 1 — Trang bìa
- **Tiêu đề:** Task 2 — Hệ thống ASR tiếng Việt kèm dịch sang tiếng Anh
- Thành viên: Nguyễn Đức Mạnh (SE194525)
- Môn: Speech Language Processing — Ngày: 20/10/2026
- Repo GitHub: https://github.com/manhhac/TASK2-_-SLP

*Lời dẫn:* Chào thầy/cô và các bạn. Em trình bày Task 2 — xây dựng hệ thống nhận dạng tiếng Việt (ASR) và dịch sang tiếng Anh.

---

### Slide 2 — Bài toán & mục tiêu
- **Input:** file audio tiếng Việt → **Output:** transcript VI + bản dịch EN
- **Kiến trúc chính:** ASR (Whisper-small) **cascade** MT (MarianMT opus-mt-vi-en)
- **3 giai đoạn:**
  1. Dataset Pipeline (làm sạch, chia tập, chống rò rỉ)
  2. Baseline Reproduction + Gap Analysis
  3. Proposed Method V1 → V4
- **Metric:** ASR — WER, CER · MT — BLEU, chrF++ · Hệ thống — RTF, error propagation

*Lời dẫn:* Bài toán gồm 2 phần nối tiếp: nghe tiếng Việt và dịch sang tiếng Anh. Em đánh giá bằng WER/CER cho ASR, BLEU/chrF++ cho MT, và RTF cho tốc độ.

---

### Slide 3 — Dữ liệu: VIVOS
| Đặc tính | Giá trị |
|---|---|
| Nguồn | AILAB – VNUHCM (Zenodo) |
| License | CC BY-NC-SA 4.0 (học thuật) |
| Thô | 12.420 utterances · 15:40 · 65 speaker |
| Chuẩn âm | 16 kHz, mono |
| Sau xử lý | **10.887 mẫu** (train 8.253 / dev 1.084 / test 1.550) |
| Giới tính | Nữ 6.407 / Nam 4.480 |
| Vùng | 100% miền Bắc |

- VIVOS **chỉ có transcript tiếng Việt** → bản dịch EN sinh bằng MarianMT.

*Lời dẫn:* Dữ liệu là VIVOS — corpus tiếng Việt chuẩn, ghi trong phòng yên tĩnh. Sau làm sạch còn 10.887 mẫu.

---

### Slide 4 — Giai đoạn 1: Dataset Pipeline
10 bước: thu thập & hợp nhất → làm sạch audio → chuẩn hoá văn bản → kiểm tra căn chỉnh → kiểm tra dịch EN → loại trùng/chống rò rỉ → phân tích mất cân bằng → chia tập theo speaker → xử lý mất cân bằng → augmentation (chỉ train).

- **Lọc audio:** dur 0.5–30s, RMS ≥ 0.005, clipping ≤ 20%
- **Chuẩn hoá VB:** Unicode NFC, lowercase, mở rộng viết tắt (tp.→thành phố, vn→việt nam…)
- **Chống rò rỉ:** chia theo **speaker** (không để cùng người nói ở 2 tập), dùng SHA-256 loại trùng
- **Xử lý mất cân bằng:** speaker cap = median×3 (**thực tế không cắt mẫu nào** vì max speaker 300 < cap ~450)

*Lời dẫn:* Điểm quan trọng nhất là chia tập theo speaker để tránh rò rỉ dữ liệu — nếu cùng người ở cả train và test thì điểm sẽ ảo.

---

### Slide 5 — Giai đoạn 2: Baseline (kiến trúc & cấu hình)
- **ASR:** `openai/whisper-small` fine-tune
- **MT:** `Helsinki-NLP/opus-mt-vi-en` (cascade)
- Cấu hình: lr 1e-4, batch 2, grad-accum 4, 2 epoch, greedy decode, không augmentation
- **Phần cứng:** 1× GTX 1070 8GB (fp16 tắt) · Seed 42
- **Tập đánh giá:** bộ VIVOS *quick* (100 train / 20 dev / 20 test) — giới hạn VRAM

*Lời dẫn:* Baseline là kiến trúc cascade kinh điển: Whisper nghe tiếng Việt, rồi MarianMT dịch. Do máy chỉ có 1 GPU 8GB nên em chạy trên tập con 140 mẫu.

---

### Slide 6 — Baseline: kết quả & Gap Analysis
| Hạng mục | Paper | Thực nghiệm |
|---|---|---|
| Dữ liệu train | ~15 giờ | ~0,06 giờ (100 mẫu) |
| Kiến trúc | Whisper-small | Whisper-small + MarianMT |
| Phần cứng | Multi-GPU | 1× GTX 1070 |
| WER | không công bố | **50,52** |
| CER | không công bố | 22,83 |
| BLEU (cascade) | không công bố | 4,83 |
| RTF | ~0,1 | 0,23 |

- **Khoảng cách chính:** lượng dữ liệu train (giờ) — chênh **hàng trăm lần**.

*Lời dẫn:* Paper gốc train trên 15 giờ, em chỉ có ~100 mẫu nên WER 50% là hợp lý. Đây là "gap" chính.

---

### Slide 7 — Error Analysis (baseline)
- **WER 50,52%** → khoảng 1 nửa số từ sai
- **BLEU cascade 4,83** rất thấp dù MarianMT tốt khi có transcript đúng
  → **error propagation:** lỗi ASR lan sang MT
- **Bằng chứng:** oracle BLEU cao khi dùng transcript chuẩn, nhưng cascade thấp
- Lỗi thường gặp: sai dấu thanh, mất từ đầu/cuối câu, cắt câu do audio dài

*Lời dẫn:* Điểm đáng chú ý là chất lượng dịch tụt mạnh khi ghép cascade — chứng tỏ MT phụ thuộc rất nhiều vào chất lượng ASR.

---

### Slide 8 — Giai đoạn 3: tiến trình V1 → V4
| Phiên bản | Thay đổi chính | WER | BLEU | chrF++ | RTF |
|---|---|---|---|---|---|
| Baseline | không augment, greedy | 50,52 | 4,83 | 25,52 | 0,23 |
| V1 | + augmentation + speaker cap | — | — | — | — |
| V2 | + LoRA r=16 + beam=5 + augment | **41,24** | 22,01 | 42,98 | 0,27 |
| V3 | + noise-robust MT | 43,30 | **22,73** | **43,62** | 0,25 |
| V4 | Whisper end-to-end translate | — | 4,23 | 17,21 | 0,18 |

- **Nhận xét:** V2 giảm WER mạnh nhất; V3 cho BLEU/chrF++ cao nhất.

*Lời dẫn:* Em thử 4 phiên bản. Bước nhảy lớn nhất đến từ gói LoRA + beam search. V4 là thử nghiệm end-to-end nhưng thất bại.

---

### Slide 9 — Phiên bản chọn: V3
**V3 = noise-robust MT + LoRA + beam search**
- WER 43,30 · CER 16,88 · **BLEU 22,73** · **chrF++ 43,62** · RTF 0,25
- **Vì sao chọn V3:** BLEU/chrF++ cao nhất — mục tiêu cuối là *chất lượng dịch*, không chỉ transcript
- Đánh đổi: WER nhích lên so với V2 (43,30 vs 41,24), chênh BLEU +0,72 (chưa kiểm định ý nghĩa — 1 seed)

*Lời dẫn:* Nếu ưu tiên WER thuần thì V2 tốt hơn, nhưng mục tiêu bài toán là bản dịch nên em chọn V3.

---

### Slide 10 — So sánh Cascade vs End-to-end
| | Cascade (V3) | End-to-end (V4) |
|---|---|---|
| BLEU | **22,73** | 4,23 |
| chrF++ | **43,62** | 17,21 |
| RTF | 0,25 | **0,18** |

- E2E **nhanh hơn** nhưng **chất lượng kém hơn ~5 lần**
- **Kết luận:** với dữ liệu nhỏ, **cascade ưu thế** hơn end-to-end
- → Đây là **kết quả âm tính có giá trị** (kiểm chứng giả thuyết)

*Lời dẫn:* Nhiều người nghĩ end-to-end luôn tốt hơn, nhưng với dữ liệu nhỏ thì cascade thắng rõ rệt. Đó cũng là một đóng góp.

---

### Slide 11 — Kiểm định & Ablation (gián tiếp)
- **Ablation tách rời chưa làm được** (tài nguyên) → chỉ suy luận từ so sánh giữa các version
- Phân rã đóng góp (ước lượng): LoRA + beam ≈ phần lớn cải thiện; noise-robust MT ≈ biên nhỏ
- **Kiểm định:** chưa chạy bootstrap significance do chỉ 1 seed
- **RQ1–RQ4** đều có câu trả lời định lượng

*Lời dẫn:* Em trung thực: chưa tách rời từng thành phần do thiếu tài nguyên, nên đây là ablation "gián tiếp".

---

### Slide 12 — Hạn chế & Minh bạch
1. **Tập test chỉ 20 mẫu**, 1 seed → độ tin cậy thống kê thấp
2. **Nguy cơ overfit test** do dùng chính test set để lặp phiên bản
3. **Nhãn EN sinh bởi MarianMT** → oracle BLEU vô hiệu (so sánh vòng)
4. **V1 đã train nhưng chưa đánh giá** → thiếu 1 mốc trong tiến trình
5. Không dùng dữ liệu synthetic cho kết quả báo cáo (chỉ kiểm chứng pipeline)

*Lời dẫn:* Em liệt kê rõ hạn chế để thầy/cô đánh giá đúng bản chất, không tô hồng kết quả.

---

### Slide 13 — Kết luận & Hướng phát triển
**Kết luận**
- Hoàn thành 3 giai đoạn; hệ thống cascade VI→EN chạy được, RTF < 1
- **V3 là phiên bản tốt nhất**; cascade > end-to-end trong điều kiện dữ liệu nhỏ

**Hướng phát triển**
- Chạy full VIVOS trên GPU mạnh hơn
- Ablation tách rời + kiểm định ý nghĩa (bootstrap, nhiều seed)
- Dùng nhãn EN do người dịch (không dùng MarianMT sinh nhãn)
- Thử Whisper-medium/large + LoRA

**Q&A** — Cảm ơn thầy/cô và các bạn!

*Lời dẫn:* Tóm lại em đã chứng minh được pipeline hoàn chỉnh và rút ra bài học về đánh đổi cascade/E2E. Em xin nhận câu hỏi.

---

## Phân bổ thời gian gợi ý (11 phút)
| Slide | Nội dung | Thời gian |
|---|---|---|
| 1–2 | Mở đầu & bài toán | 1,0 phút |
| 3–4 | Dữ liệu & pipeline | 2,0 phút |
| 5–7 | Baseline + gap + error analysis | 3,0 phút |
| 8–10 | Proposed V1–V4 + so sánh | 3,5 phút |
| 11–12 | Kiểm định & hạn chế | 1,0 phút |
| 13 | Kết luận | 0,5 phút |
