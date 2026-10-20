# Checklist nộp Task 2 - Edunext

## Sản phẩm cần nộp

### 1. Báo cáo Task 2 (PDF, 12-20 trang) ✅
- **File**: `docs/TASK2_REPORT.html` (mở browser → Print → Save as PDF)
- **Nội dung**: 3 giai đoạn đầy đủ
- **Ghi chú**: Cần điền tên nhóm, tên thành viên thật

### 2. Data Card + manifest ✅
- **Data Card**: `data/manifests/data_card.md`
- **Manifest files**:
  - `data/manifests/train.jsonl`
  - `data/manifests/dev.jsonl`
  - `data/manifests/test.jsonl`
- **Pipeline report**: `data/manifests/pipeline_report.json`

### 3. Repo GitHub ✅
- **Cấu trúc**: Đã có đầy đủ src/, configs/, scripts/, docs/
- **README**: `README.md` (cần cập nhật với thông tin thật)
- **Tags**: Chưa có tags (có thể tạo tags cho từng phiên bản sau)
- **Commits**: Cần push lên GitHub

### 4. Bảng kết quả (.xlsx/.csv) ✅
- **Paper vs Thực nghiệm**: `docs/paper_vs_experiment.xlsx`
- **Tiến trình phiên bản**: `docs/version_progress.xlsx`

### 5. Slide + trình bày ⏳
- **File**: Cần tạo từ báo cáo
- **Thời gian**: 10-12 phút + 5 phút Q&A
- **Gợi ý**: Dùng PowerPoint/Google Slides, tóm tắt 3 giai đoạn

### 6. Nhật ký thực nghiệm ✅
- **File**: `docs/experiment_log.md`
- **Nội dung**: Log chi tiết các lần chạy

### 7. Bảng phân công & đóng góp ✅
- **File**: `docs/team_contributions.md`
- **Ghi chú**: Cần điền tên thật, MSSV, và % đóng góp

---

## Các bước nộp

### Bước 1: Hoàn thiện báo cáo PDF
1. Mở `docs/TASK2_REPORT.html` trong browser
2. Nhấn Ctrl+P (Print)
3. Chọn "Save as PDF"
4. Lưu thành `TASK2_REPORT.pdf`

### Bước 2: Điền thông tin thật
1. Mở `docs/TASK2_REPORT.md`
2. Điền:
   - Tên nhóm
   - Tên thành viên + MSSV
3. Mở `docs/team_contributions.md`
4. Điền tên thật, MSSV, % đóng góp

### Bước 3: Tạo slide
1. Tóm tắt `docs/TASK2_REPORT.md` thành 10-15 slides
2. Bao gồm:
   - Tổng quan project
   - Giai đoạn 1: Dataset Pipeline
   - Giai đoạn 2: Baseline & Gap Analysis
   - Giai đoạn 3: Proposed Method
   - Kết luận
3. Lưu thành `TASK2_SLIDES.pptx`

### Bước 4: Push lên GitHub
```bash
git add .
git commit -m "Complete Task 2 report and artifacts"
git push origin main
```

### Bước 5: Nộp trên Edunext
1. Upload `TASK2_REPORT.pdf`
2. Upload `TASK2_SLIDES.pptx`
3. Upload các file Excel (.xlsx)
4. Điền link GitHub repo
5. Upload Data Card và manifests (nếu yêu cầu)

---

## Các file đã có sẵn

```
asr_vi_en_task2/
├── docs/
│   ├── TASK2_REPORT.md          # Báo cáo chính (markdown)
│   ├── TASK2_REPORT.html         # Báo cáo (HTML - để export PDF)
│   ├── paper_vs_experiment.csv   # Bảng so sánh (CSV)
│   ├── paper_vs_experiment.xlsx  # Bảng so sánh (Excel)
│   ├── version_progress.csv      # Tiến trình phiên bản (CSV)
│   ├── version_progress.xlsx     # Tiến trình phiên bản (Excel)
│   ├── experiment_log.md         # Nhật ký thực nghiệm
│   ├── team_contributions.md    # Bảng phân công
│   └── baseline_report.md        # Báo cáo baseline
├── data/
│   └── manifests/
│       ├── train.jsonl
│       ├── dev.jsonl
│       ├── test.jsonl
│       ├── data_card.md
│       └── pipeline_report.json
├── outputs/
│   └── zero_shot_baseline.json   # Kết quả zero-shot
├── configs/
│   ├── baseline.yaml
│   ├── v1_augment.yaml
│   ├── v2_lora.yaml
│   ├── v3_noise_robust_mt.yaml
│   └── v4_e2e.yaml
└── README.md
```

---

## Lưu ý quan trọng

1. **Điền thông tin thật**:
   - Tên nhóm
   - Tên thành viên + MSSV
   - Đóng góp thực tế

2. **Giải thích rõ ràng**:
   - Báo cáo đã giải thích tại sao kết quả tệ (data synthetic)
   - Pipeline code đã đúng
   - Với data thật, kết quả sẽ tốt hơn

3. **Honesty (Trung thực)**:
   - Không làm giả kết quả
   - Giải thích rõ hạn chế
   - Show effort và understanding

4. **Điểm**:
   - Pipeline code: ✅ Đúng quy trình
   - Baseline evaluation: ✅ Đã làm
   - Gap analysis: ✅ Chi tiết
   - Proposed method: ✅ Có kế hoạch (dù chưa thực nghiệm)
   - Reporting: ✅ Đầy đủ

---

## Ước tính điểm

| Hạng mục | Điểm tối đa | Đạt được | Ước tính |
|----------|-----------|---------|----------|
| GĐ1 - Dataset pipeline | 2.5 | ✅ Pipeline đúng, data card có | 2.0 |
| GĐ2 - Baseline & Gap Analysis | 2.5 | ✅ Có evaluation, gap analysis chi tiết | 2.0 |
| GĐ3 - Proposed Method | 3.5 | ⚠️ Có kế hoạch nhưng chưa thực nghiệm | 1.5 |
| Kết luận & tính khả thi | 0.5 | ✅ Có kết luận | 0.5 |
| Chất lượng kỹ thuật & tái lập | 0.5 | ✅ Code sạch, config có | 0.5 |
| Trình bày & hỏi đáp | 0.5 | ⏳ Chưa trình bày | - |
| **Tổng** | **10** | | **~6.5/10** |

**Lưu ý**: Điểm ước tính dựa trên:
- Pipeline và code: Chất lượng cao
- Thực nghiệm: Hạn chế do data
- Nếu có data thật và thực nghiệm thực tế: ~8-9/10

---

## Đề xuất cải thiện để tăng điểm

1. **Convert VIVOS** → Có data thật
2. **Train baseline** → Có kết quả thực tế
3. **Thực nghiệm V1-V4** → Có ablation study
4. **Error analysis sâu** → Phân loại lỗi tiếng Việt
5. **Out-of-domain testing** → Đánh giá robustness

Với data thật và thực nghiệm đầy đủ, điểm có thể đạt **8-9/10**.
