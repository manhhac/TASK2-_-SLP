==================================================
DANH SÁCH FILE NỘP TASK 2 - ĐÃ HOÀN THÀNH
==================================================

CÁC FILE ĐÃ TẠO SẴN:

1. BÁO CÁO CHÍNH
   - docs/TASK2_REPORT.md (Markdown gốc)
   - docs/TASK2_REPORT.html (HTML formatted - DÙNG FILE NÀY ĐỂ TẠO PDF)

2. BẢNG KẾT QUẢ (EXCEL)
   - docs/paper_vs_experiment.xlsx
   - docs/version_progress.xlsx

3. DATA CARD & MANIFESTS
   - data/manifests/data_card.md
   - data/manifests/train.jsonl
   - data/manifests/dev.jsonl
   - data/manifests/test.jsonl
   - data/manifests/pipeline_report.json

4. NHẬT KÝ & PHÂN CÔNG
   - docs/experiment_log.md
   - docs/team_contributions.md (đã điền placeholder)
   - docs/baseline_report.md

5. CODE & CONFIG
   - configs/baseline.yaml
   - configs/v1_augment.yaml
   - configs/v2_lora.yaml
   - configs/v3_noise_robust_mt.yaml
   - configs/v4_e2e.yaml
   - src/ (full source code)
   - scripts/ (utility scripts)

==================================================
CÁNH BÁO: CẦN LÀM TRƯỚC KHI NỘP
==================================================

1. TẠO PDF TỪ BÁO CÁO (BẮT BUỘC - 1 PHÚT)

   Cách 1: Dùng HTML (KHUYẾN NGHỊ)
   - Mở file: docs/TASK2_REPORT.html trong browser
   - Nhấn Ctrl+P
   - Chọn "Save as PDF" hoặc "Microsoft Print to PDF"
   - Lưu thành: TASK2_REPORT.pdf

   Cách 2: Chạy script
   - python scripts/open_print_dialog.py
   - Browser sẽ mở, sau đó Ctrl+P → Save as PDF

2. ĐIỀN THÔNG TIN THẬT (QUAN TRỌNG - 2 PHÚT)

   Mở docs/TASK2_REPORT.md:
   - Điền tên nhóm thật
   - Điền tên thành viên thật + MSSV thật

   Mở docs/team_contributions.md:
   - Điền tên thật, MSSV thật
   - Điều chỉnh % đóng góp thực tế

3. TẠO SLIDE (10-15 PHÚT)

   Tóm tắt docs/TASK2_REPORT.md thành 10-15 slides:
   - Slide 1: Tiêu đề + nhóm
   - Slide 2: Tổng quan 3 giai đoạn
   - Slide 3: Giai đoạn 1 - Dataset Pipeline
   - Slide 4: Giai đoạn 2 - Baseline
   - Slide 5: Gap Analysis
   - Slide 6: Giai đoạn 3 - Proposed Method
   - Slide 7: Bảng tiến trình
   - Slide 8: Kết luận
   - Slide 9: Hạn chế & Đề xuất
   - Slide 10: Q&A

   Lưu thành: TASK2_SLIDES.pptx

==================================================
FILE CẦN NỘP EDUNEXT
==================================================

BẮT BUỘC:
1. TASK2_REPORT.pdf (tạo từ HTML)
2. TASK2_SLIDES.pptx (cần tạo)
3. docs/paper_vs_experiment.xlsx
4. docs/version_progress.xlsx
5. Link GitHub repo

TÙY CHỌN (nếu yêu cầu):
6. data/manifests/data_card.md
7. data/manifests/*.jsonl
8. docs/experiment_log.md
9. docs/team_contributions.md

==================================================
ƯỚC TÍNH ĐIỂM: ~6.5/10
==================================================

Điểm cao:
- Pipeline code đúng quy trình (2.0/2.5)
- Baseline evaluation + gap analysis (2.0/2.5)
- Code chất lượng, tái lập được (0.5/0.5)

Điểm thấp:
- Proposed method chưa thực nghiệm (1.5/3.5)

Với data thật + thực nghiệm: ~8-9/10

==================================================
THỜI GIAN HOÀN THÀNH CÒN LẠI
==================================================

- Tạo PDF: 1 phút
- Điền tên thật: 2 phút
- Tạo slide: 10-15 phút
- Upload Edunext: 5 phút

Tổng: ~20 phút

==================================================
