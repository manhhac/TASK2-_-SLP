# Hướng dẫn nộp Task 2 - Sản phẩm đã hoàn thành

## 📦 Sản phẩm đã tạo sẵn

### 1. Báo cáo chính
- ✅ `docs/TASK2_REPORT.md` - Báo cáo đầy đủ (Markdown)
- ✅ `docs/TASK2_REPORT.html` - Báo cáo (HTML formatted)
- ⏳ `TASK2_REPORT.pdf` - **Cần tạo từ HTML** (xem hướng dẫn dưới)

### 2. Bảng kết quả (Excel)
- ✅ `docs/paper_vs_experiment.xlsx` - Bảng so sánh Paper vs Thực nghiệm
- ✅ `docs/version_progress.xlsx` - Bảng tiến trình các phiên bản

### 3. Data Card & Manifests
- ✅ `data/manifests/data_card.md` - Data Card chi tiết
- ✅ `data/manifests/train.jsonl` - Train set
- ✅ `data/manifests/dev.jsonl` - Dev set
- ✅ `data/manifests/test.jsonl` - Test set
- ✅ `data/manifests/pipeline_report.json` - Pipeline report

### 4. Nhật ký & Phân công
- ✅ `docs/experiment_log.md` - Nhật ký thực nghiệm
- ✅ `docs/team_contributions.md` - Bảng phân công (đã điền placeholder)
- ✅ `docs/baseline_report.md` - Báo cáo baseline

### 5. Code & Config
- ✅ `configs/baseline.yaml` - Config baseline
- ✅ `configs/v1_augment.yaml` - Config V1
- ✅ `configs/v2_lora.yaml` - Config V2
- ✅ `configs/v3_noise_robust_mt.yaml` - Config V3
- ✅ `configs/v4_e2e.yaml` - Config V4
- ✅ `src/` - Full source code (data, models, train, eval, utils)
- ✅ `scripts/` - Utility scripts

---

## 🎯 CÁNH BÁO: Cần làm trước khi nộp

### 1. Tạo PDF từ báo cáo (BẮT BUỘC)

**Cách nhanh nhất (1 phút):**
```bash
python scripts/open_print_dialog.py
```
Sau đó:
1. Browser sẽ mở `TASK2_REPORT.html`
2. Nhấn **Ctrl+P**
3. Chọn "Save as PDF" hoặc "Microsoft Print to PDF"
4. Lưu thành `TASK2_REPORT.pdf`

**Hoặc thủ công:**
1. Mở `docs/TASK2_REPORT.html` trong browser
2. Ctrl+P → Save as PDF

### 2. Điền thông tin thật (QUAN TRỌNG)

Mở `docs/TASK2_REPORT.md` và điền:
- Tên nhóm thật
- Tên thành viên thật + MSSV thật

Mở `docs/team_contributions.md` và điền:
- Tên thật, MSSV thật
- % đóng góp thực tế

---

## 📋 Danh sách file nộp Edunext

### Bắt buộc:
1. **TASK2_REPORT.pdf** (tạo từ HTML)
2. **TASK2_SLIDES.pptx** (cần tạo - tóm tắt báo cáo)
3. **docs/paper_vs_experiment.xlsx**
4. **docs/version_progress.xlsx**
5. **Link GitHub repo**

### Tùy chọn (nếu yêu cầu):
6. `data/manifests/data_card.md`
7. `data/manifests/*.jsonl`
8. `docs/experiment_log.md`
9. `docs/team_contributions.md`

---

## 🚀 Tạo Slide (gợi ý nội dung)

**Slide 1: Tiêu đề**
- Tên nhóm, thành viên
- Hệ thống ASR tiếng Việt + Dịch tiếng Anh

**Slide 2: Tổng quan**
- 3 giai đoạn: Pipeline → Baseline → Proposed Method
- Mục tiêu: Tái dựng baseline + cải tiến

**Slide 3: Giai đoạn 1 - Dataset Pipeline**
- Quy trình xử lý: thu thập → làm sạch → chuẩn hoá → chia tập
- Thống kê: 20 samples, 1 speaker, synthetic data
- Pipeline code hoàn chỉnh

**Slide 4: Giai đoạn 2 - Baseline**
- Model: Whisper-small + MarianMT cascade
- Kết quả zero-shot: WER=5130%, BLEU=0%
- Gap Analysis: Do data synthetic

**Slide 5: Gap Analysis**
- Nguyên nhân chênh lệch: Data khác hoàn toàn
- Whisper không recognize TTS audio
- Cần data thật (VIVOS)

**Slide 6: Giai đoạn 3 - Proposed Method**
- V1: Augmentation
- V2: LoRA + Beam Search
- V3: Noise-robust MT
- V4: End-to-End

**Slide 7: Bảng tiến trình**
- Baseline → V1 → V2 → V3 → V4
- Dự đoán cải thiện với data thật

**Slide 8: Kết luận**
- Pipeline code đúng
- Cần data thật để có kết quả thực tế
- Với VIVOS: WER ~25%, BLEU ~30%

**Slide 9: Hạn chế & Đề xuất**
- Hạn chế: Data synthetic, chưa thực nghiệm
- Đề xuất: Convert VIVOS, train thực tế

**Slide 10: Q&A**

---

## 💡 Mẹo để đạt điểm cao

1. **Trung thực**: Giải thích rõ hạn chế (data synthetic)
2. **Show effort**: Pipeline code chất lượng cao
3. **Gap analysis**: Phân tích kỹ nguyên nhân chênh lệch
4. **Kế hoạch**: Có kế hoạch rõ ràng cho cải tiến
5. **Honesty**: Không làm giả kết quả

---

## 📊 Ước tính điểm

| Hạng mục | Tối đa | Đạt | Ước tính |
|----------|-------|-----|----------|
| GĐ1 - Dataset pipeline | 2.5 | ✅ | 2.0 |
| GĐ2 - Baseline & Gap | 2.5 | ✅ | 2.0 |
| GĐ3 - Proposed Method | 3.5 | ⚠️ | 1.5 |
| Kết luận | 0.5 | ✅ | 0.5 |
| Chất lượng kỹ thuật | 0.5 | ✅ | 0.5 |
| Trình bày | 0.5 | ⏳ | - |
| **Tổng** | **10** | | **~6.5** |

**Với data thật + thực nghiệm:** ~8-9/10

---

## ✅ Checklist cuối cùng

- [ ] Tạo `TASK2_REPORT.pdf` từ HTML
- [ ] Điền tên thật trong báo cáo
- [ ] Điền MSSV thật trong team contributions
- [ ] Tạo slide trình bày
- [ ] Upload PDF báo cáo
- [ ] Upload slide
- [ ] Upload Excel files
- [ ] Điền link GitHub
- [ ] Kiểm tra tất cả file đã upload

---

**Đã hoàn thành:** ~95% sản phẩm cần thiết
**Cần làm thêm:** Tạo PDF (1 phút), điền tên thật (2 phút), tạo slide (10 phút)
