# Bảng phân công & đóng góp - Task 2

## Tổng quan

| Thành viên | MSSV | Vai trò | Đóng góp tổng thể |
|-----------|------|---------|------------------|
| Nguyễn Đức Mạnh | SE194525 | Tất cả các phần | 100% |

---

## Chi tiết đóng góp theo giai đoạn

### Giai đoạn 1: Dataset Pipeline

| Thành viên | Công việc | File/Sản phẩm | % đóng góp |
|-----------|----------|---------------|------------|
| Nguyễn Đức Mạnh | Thiết kế pipeline, prepare_data.py | src/data/prepare_data.py | 100% |
| Nguyễn Đức Mạnh | Text normalization | src/data/text_normalize.py | 100% |
| Nguyễn Đức Mạnh | Audio quality check | src/data/prepare_data.py (audio functions) | 100% |
| Nguyễn Đức Mạnh | Translation check | src/data/translation_check.py | 100% |
| Nguyễn Đức Mạnh | Imbalance analysis | src/data/imbalance.py | 100% |

### Giai đoạn 2: Baseline Reproduction

| Thành viên | Công việc | File/Sản phẩm | % đóng góp |
|-----------|----------|---------------|------------|
| Nguyễn Đức Mạnh | Whisper integration | src/models/asr_whisper.py | 100% |
| Nguyễn Đức Mạnh | Training script | src/train/train_asr.py | 100% |
| Nguyễn Đức Mạnh | MarianMT integration | src/models/mt_model.py | 100% |
| Nguyễn Đức Mạnh | Cascade pipeline | src/models/cascade.py | 100% |
| Nguyễn Đức Mạnh | Evaluation script | src/eval/evaluate.py | 100% |
| Nguyễn Đức Mạnh | Metrics | src/eval/metrics.py | 100% |

### Giai đoạn 3: Proposed Method

| Thành viên | Công việc | File/Sản phẩm | % đóng góp |
|-----------|----------|---------------|------------|
| Nguyễn Đức Mạnh | Augmentation | src/data/augment.py | 100% |
| Nguyễn Đức Mạnh | LoRA implementation | src/models/asr_whisper.py (LoRA) | 100% |
| Nguyễn Đức Mạnh | Beam search | src/models/cascade.py | 100% |
| Nguyễn Đức Mạnh | Noise-robust MT | src/train/train_mt.py | 100% |
| Nguyễn Đức Mạnh | E2E translation | configs/v4_e2e.yaml | 100% |
| Nguyễn Đức Mạnh | Ablation study | docs/ablation.md | 100% |

### Báo cáo & Tài liệu

| Thành viên | Công việc | File/Sản phẩm | % đóng góp |
|-----------|----------|---------------|------------|
| Nguyễn Đức Mạnh | Data Card | data/manifests/data_card.md | 100% |
| Nguyễn Đức Mạnh | Gap Analysis | docs/gap_analysis.md | 100% |
| Nguyễn Đức Mạnh | Error Analysis | docs/error_analysis.md | 100% |
| Nguyễn Đức Mạnh | Báo cáo chính | docs/TASK2_REPORT.md | 100% |
| Nguyễn Đức Mạnh | Bảng kết quả | docs/*.csv | 100% |

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
| Nguyễn Đức Mạnh | | 2026-10-20 |

---

## Lưu ý

- Bảng phân công này là mẫu, cần điền tên thật của các thành viên
- Đóng góp có thể điều chỉnh tùy theo thực tế làm việc
- Đảm bảo mỗi thành viên có đóng góp rõ ràng và minh bạch
- Sử dụng GitHub để track lịch sử commit
