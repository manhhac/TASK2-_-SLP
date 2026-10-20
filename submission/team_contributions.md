# Bảng phân công & đóng góp - Task 2

## Tổng quan

| Thành viên | MSSV | Vai trò | Đóng góp tổng thể |
|-----------|------|---------|------------------|
| Thành viên 1 | 20260001 | Lead Data Pipeline | 30% |
| Thành viên 2 | 20260002 | Lead ASR Model | 30% |
| Thành viên 3 | 20260003 | Lead MT Model | 20% |
| Thành viên 4 | 20260004 | Evaluation & Reporting | 20% |

---

## Chi tiết đóng góp theo giai đoạn

### Giai đoạn 1: Dataset Pipeline

| Thành viên | Công việc | File/Sản phẩm | % đóng góp |
|-----------|----------|---------------|------------|
| Thành viên 1 | Thiết kế pipeline, prepare_data.py | src/data/prepare_data.py | 40% |
| Thành viên 1 | Text normalization | src/data/text_normalize.py | 40% |
| Thành viên 2 | Audio quality check | src/data/prepare_data.py (audio functions) | 20% |
| Thành viên 3 | Translation check | src/data/translation_check.py | 20% |
| Thành viên 4 | Imbalance analysis | src/data/imbalance.py | 20% |

### Giai đoạn 2: Baseline Reproduction

| Thành viên | Công việc | File/Sản phẩm | % đóng góp |
|-----------|----------|---------------|------------|
| Thành viên 2 | Whisper integration | src/models/asr_whisper.py | 40% |
| Thành viên 2 | Training script | src/train/train_asr.py | 30% |
| Thành viên 3 | MarianMT integration | src/models/mt_model.py | 30% |
| Thành viên 3 | Cascade pipeline | src/models/cascade.py | 30% |
| Thành viên 4 | Evaluation script | src/eval/evaluate.py | 40% |
| Thành viên 4 | Metrics | src/eval/metrics.py | 30% |

### Giai đoạn 3: Proposed Method

| Thành viên | Công việc | File/Sản phẩm | % đóng góp |
|-----------|----------|---------------|------------|
| Thành viên 1 | Augmentation | src/data/augment.py | 40% |
| Thành viên 2 | LoRA implementation | src/models/asr_whisper.py (LoRA) | 50% |
| Thành viên 2 | Beam search | src/models/cascade.py | 30% |
| Thành viên 3 | Noise-robust MT | src/train/train_mt.py | 40% |
| Thành viên 4 | E2E translation | configs/v4_e2e.yaml | 30% |
| Thành viên 4 | Ablation study | docs/ablation.md | 50% |

### Báo cáo & Tài liệu

| Thành viên | Công việc | File/Sản phẩm | % đóng góp |
|-----------|----------|---------------|------------|
| Thành viên 1 | Data Card | data/manifests/data_card.md | 40% |
| Thành viên 2 | Gap Analysis | docs/gap_analysis.md | 30% |
| Thành viên 3 | Error Analysis | docs/error_analysis.md | 30% |
| Thành viên 4 | Báo cáo chính | docs/TASK2_REPORT.md | 50% |
| Thành viên 4 | Bảng kết quả | docs/*.csv | 50% |

---

## Lịch sử commit (GitHub)

Để đảm bảo tính minh bạch, mỗi thành viên nên:
1. Tạo branch riêng cho phần của mình
2. Commit thường xuyên với message rõ ràng
3. Pull request với review từ các thành viên khác
4. Merge vào main sau khi được approve

Ví dụ:
```
[1] feat(data): implement prepare_data pipeline
[2] feat(asr): add Whisper integration
[3] feat(mt): add MarianMT cascade
[4] docs: update TASK2_REPORT.md
```

---

## Ký xác nhận

| Thành viên | Chữ ký | Ngày |
|-----------|--------|------|
| Thành viên 1 | | 2026-10-20 |
| Thành viên 2 | | 2026-10-20 |
| Thành viên 3 | | 2026-10-20 |
| Thành viên 4 | | 2026-10-20 |

---

## Lưu ý

- Bảng phân công này là mẫu, cần điền tên thật của các thành viên
- Đóng góp có thể điều chỉnh tùy theo thực tế làm việc
- Đảm bảo mỗi thành viên có đóng góp rõ ràng và minh bạch
- Sử dụng GitHub để track lịch sử commit
