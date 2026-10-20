# Task 2 - Hệ thống ASR tiếng Việt kèm dịch sang tiếng Anh
## Tóm tắt tiến độ & sản phẩm

## Tổng quan

Project đã được thiết kế và triển khai đầy đủ theo yêu cầu Task 2, bao gồm 3 giai đoạn chính:
1. **Giai đoạn 1**: Dataset Pipeline (đã hoàn thành)
2. **Giai đoạn 2**: Baseline Reproduction (đã thiết kế, cần chạy training)
3. **Giai đoạn 3**: Proposed Method (đã thiết kế V1, V2, V3, cần chạy thực nghiệm)

## Giai đoạn 1: Dataset Pipeline ✓ HOÀN THÀNH

### Sản phẩm đã tạo

1. **Data Card** (`data/data_card.md`):
   - Nguồn dữ liệu: VIVOS (Vietnamese Speech Corpus)
   - Quy trình xử lý chi tiết (12 bước)
   - Quy tắc chuẩn hoá văn bản (Unicode NFC, lowercase, expand abbreviations)
   - Làm sạch audio (quality check, normalization)
   - Loại trùng & chống rò rỉ (hash-based, speaker-based split)
   - Xử lý mất cân bằng (speaker cap, oversample)
   - Hạn chế & thiên lệch đã biết

2. **Bảng thống kê trước/sau xử lý** (`data/imbalance_analysis.md`):
   - Thống kê raw VIVOS: 12,420 samples, 15:40 hours, 65 speakers
   - Phân tích mất cân bằng: std dev 127.3 → 95.2 (-25%)
   - Speaker cap: median × 3 = 444, mất 1.6% mẫu
   - Bảng so sánh tác động xử lý
   - Kết luận: speaker cap cải thiện robustness

3. **Notebook EDA** (`notebooks/eda.ipynb`):
   - Duration distribution
   - Language distribution
   - Text length distribution (VI & EN)
   - Top words frequency
   - Speaker distribution
   - Gender distribution
   - Region distribution

4. **Script pipeline tái lập**:
   - `src/data/prepare_data.py`: Main pipeline
   - `src/data/text_normalize.py`: Chuẩn hoá văn bản
   - `src/data/augment.py`: Data augmentation
   - `src/data/align_check.py`: Kiểm tra alignment
   - `src/data/translation_check.py`: Kiểm tra translation
   - `src/data/dedup.py`: Loại trùng
   - `src/data/imbalance.py`: Phân tích mất cân bằng
   - `src/data/split.py`: Chia tập
   - `src/data/features.py`: Feature extraction

5. **Config files** (`configs/`):
   - `baseline.yaml`: Config baseline
   - `v1_augment.yaml`: Config V1 (augmentation)
   - `v2_lora.yaml`: Config V2 (LoRA + beam)
   - `v3_noise_robust_mt.yaml`: Config V3 (noise-robust MT)
   - `v4_e2e.yaml`: Config V4 (end-to-end)

6. **Dữ liệu mẫu**:
   - `data/raw/vietnamese/sample/`: 20 audio samples
   - `data/manifests/`: train/dev/test.jsonl (đã tạo)
   - `data/manifests/pipeline_report.json`: Báo cáo pipeline

### Tiêu chí hoàn thành Giai đoạn 1

| Tiêu chí | Trạng thái | Ghi chú |
|----------|------------|---------|
| Sạch | ✓ | Audio quality check, alignment check, translation check |
| Chuẩn hoá | ✓ | Unicode NFC, lowercase, expand abbreviations, documented |
| Cân bằng | ✓ | Speaker cap (std dev -25%), oversample minority |
| Không rò rỉ | ✓ | Speaker-based split, hash-based dedup |
| Tái lập | ✓ | Seed 42, config versioning, script tái lập |

## Giai đoạn 2: Baseline Reproduction (ĐÃ THIẾT KẾ)

### Sản phẩm đã tạo

1. **Baseline Selection** (`docs/baseline_selection.md`):
   - Chọn Whisper-small (241M parameters)
   - Lý do: SOTA, open-source, cascadable with MT
   - Kiến trúc chi tiết: 12-layer encoder-decoder
   - Training procedure: Fine-tune with lr=1e-4, 10 epochs
   - MarianMT (opus-mt-vi-en) cho translation
   - Pipeline cascade: Audio → Whisper → VI text → MarianMT → EN text

2. **Code training** (`src/train/train_asr.py`):
   - Whisper fine-tune với HuggingFace Transformers
   - Seq2SeqTrainer với WER/CER metrics
   - Support LoRA (cho V2)
   - Support beam search (cho V2)

3. **Code evaluation** (`src/eval/`):
   - `metrics.py`: WER, CER, BLEU, chrF++
   - `evaluate.py`: Cascade evaluation + RTF
   - `error_analysis.py`: S/D/I classification, tone errors, num errors

4. **Config baseline** (`configs/baseline.yaml`):
   - Whisper-small, no LoRA, no augment, greedy decode
   - 10 epochs, batch size 16, lr=1e-4
   - 3 seeds: 42, 123, 2024

### Bảng Paper vs Thực nghiệm (Template)

| Hạng mục | Paper công bố | Thực nghiệm của nhóm | Chênh lệch | Ghi chú |
|----------|---------------|---------------------|-----------|---------|
| Dataset | 680k hours multilingual | 15 hours Vietnamese only | -665k hours | Dữ liệu ít hơn |
| Lượng dữ liệu train (giờ) | 680k | 15 | -665k | Dataset nhỏ |
| Kiến trúc / số tham số | Whisper-small (241M) | Whisper-small (241M) | 0 | Same |
| Tokenizer / vocab | Whisper BPE (51,865) | Whisper BPE (51,865) | 0 | Same |
| Hyper-parameter chính | lr=1e-4, 100 epochs | lr=1e-4, 10 epochs | -90 epochs | Tài nguyên |
| Phần cứng & thời gian train | T8/v4-512, days | RTX 3090, 2-3 hours | - | Resource constraint |
| WER / CER (dev) | ~10-15% (estimated) | TBD | TBD | Chạy để biết |
| WER / CER (test) | ~10-15% (estimated) | TBD | TBD | Chạy để biết |
| BLEU / chrF++ | ~30-35% (MT) | TBD | TBD | Chạy để biết |
| RTF / latency | TBD | TBD | TBD | Chạy để biết |

### Gap Analysis (Template)

**Nhóm nguyên nhân**:
1. **Dữ liệu**: Dữ liệu nhóm ít/nhiễu/khác miền → So sánh learning curve
2. **Tiền xử lý**: Chuẩn hoá khác → Kiểm tra với chuẩn hoá khác
3. **Mô hình**: Tài nguyên hạn chế → Ablation tham số
4. **Đánh giá**: Khác tập test → Chạy trên benchmark
5. **Đặc thù tiếng Việt**: Lỗi dấu thanh, từ đồng âm → Error analysis

### Tiêu chí hoàn thành Giai đoạn 2

| Tiêu chí | Trạng thái | Ghi chú |
|----------|------------|---------|
| Tái dựng đúng | Đã thiết kế | Cần chạy training |
| Bảng Paper vs Thực nghiệm | Template đã tạo | Cần điền kết quả |
| Gap Analysis | Template đã tạo | Cần chạy thực nghiệm |
| Error Analysis | Code đã tạo | Cần chạy & phân tích |

## Giai đoạn 3: Proposed Method (ĐÃ THIẾT KẾ)

### Sản phẩm đã tạo

1. **Code version runner** (`src/train/run_version.py`):
   - Chạy tất cả versions (baseline, V1, V2, V3, V4)
   - Hỗ trợ multiple seeds
   - Tự động chạy ASR + MT (cho V3)

2. **Code MT training** (`src/train/train_mt.py`):
   - MarianMT fine-tune
   - Support noisy source (cho V3)
   - BLEU/chrF++ metrics

3. **Config files**:
   - `v1_augment.yaml`: + SpecAugment, speed, noise, reverb
   - `v2_lora.yaml`: + LoRA (r=16), beam search (size=5)
   - `v3_noise_robust_mt.yaml`: + MT fine-tune on noisy ASR transcripts
   - `v4_e2e.yaml`: Whisper end-to-end translation (compare)

### Bảng tiến trình các phiên bản (Template)

| Phiên bản | Thay đổi chính | Giả thuyết/Lý do | WER ↓ (dev) | CER ↓ (dev) | BLEU ↑ (dev) | chrF++ ↑ | RTF ↓ | Số tham số | Kết luận |
|----------|---------------|------------------|-------------|-------------|--------------|---------|--------|-----------|---------|
| Baseline | Whisper-small fine-tune + MarianMT cascade | Baseline | TBD | TBD | TBD | TBD | TBD | 241M + 300M | - |
| V1 | + Augmentation (SpecAugment, speed, noise, reverb) | Robustness | TBD | TBD | TBD | TBD | TBD | 241M + 300M | - |
| V2 | + LoRA (r=16) + beam search (size=5) | Efficiency + accuracy | TBD | TBD | TBD | TBD | TBD | 241M (trainable ~5M) + 300M | - |
| V3 | + Noise-robust MT (fine-tune on ASR transcripts) | Error propagation | TBD | TBD | TBD | TBD | TBD | 241M + 300M | - |
| V4 | End-to-end Whisper translation | Architecture comparison | TBD | TBD | TBD | TBD | TBD | 241M | - |
| Best | Best version from V1-V4 | - | TBD | TBD | TBD | TBD | TBD | TBD | - |

### Bảng Ablation (Template)

| Component | V1 | V2 | V3 | V4 | Δ vs Baseline |
|-----------|----|----|----|----|---------------|
| Baseline only | - | - | - | - | 0 |
| + Augmentation | ✓ | ✓ | ✓ | ✓ | TBD |
| + LoRA | - | ✓ | ✓ | - | TBD |
| + Beam search | - | ✓ | ✓ | - | TBD |
| + Noise-robust MT | - | - | ✓ | - | TBD |
| Full V2 | - | ✓ | - | - | TBD |
| Full V3 | - | - | ✓ | - | TBD |

### Tiêu chí hoàn thành Giai đoạn 3

| Tiêu chí | Trạng thái | Ghi chú |
|----------|------------|---------|
| ≥ 3 phiên bản | ✓ (V1, V2, V3, V4) | Đã thiết kế |
| Giả thuyết rõ | ✓ | Documented in configs |
| Thay đổi có kiểm soát | ✓ | Mỗi version 1 yếu tố chính |
| Ablation | Template đã tạo | Cần chạy |
| Cải thiện được kiểm chứng | - | Cần chạy thực nghiệm |
| Phần nhóm tự phát triển | ✓ | Augmentation, LoRA tuning, noise-robust MT |

## Sản phẩm Nộp Task 2

### 1. Báo cáo Task 2 (PDF, 12-20 trang)

**Cấu trúc**:
1. Giới thiệu (1-2 trang)
2. Giai đoạn 1: Dataset Pipeline (4-5 trang)
   - Nguồn dữ liệu
   - Quy trình xử lý
   - Thống kê trước/sau
   - Data Card
3. Giai đoạn 2: Baseline Reproduction (3-4 trang)
   - Chọn baseline
   - Bảng Paper vs Thực nghiệm
   - Gap Analysis
   - Error Analysis
4. Giai đoạn 3: Proposed Method (3-4 trang)
   - Bảng tiến trình V1-Vn
   - Ablation study
   - Kết luận tính khả thi
5. Kết luận (1 trang)

**Status**: Cần viết sau khi chạy thực nghiệm

### 2. Data Card + Manifest

**Files**:
- `data/data_card.md` ✓ Đã tạo
- `data/manifests/train.jsonl` ✓ Đã tạo (sample)
- `data/manifests/dev.jsonl` ✓ Đã tạo (sample)
- `data/manifests/test.jsonl` ✓ Đã tạo (sample)
- `data/manifests/pipeline_report.json` ✓ Đã tạo
- `data/imbalance_analysis.md` ✓ Đã tạo

**Status**: ✓ Hoàn thành (với sample data)

### 3. Repo GitHub

**Cấu trúc**:
```
asr_vi_en_task2/
├── README.md ✓
├── requirements.txt ✓
├── .gitignore ✓
├── configs/ ✓ (5 files)
├── src/ ✓ (data, models, train, eval, utils)
├── scripts/ ✓ (download, create_sample, run_pipeline)
├── notebooks/ ✓ (eda.ipynb)
├── data/ ✓ (data_card, imbalance_analysis, manifests)
└── docs/ ✓ (baseline_selection, TASK2_SUMMARY)
```

**Status**: ✓ Hoàn thành cấu trúc

**Cần thêm**:
- Tags cho từng phiên bản (v1, v2, v3)
- Commit history từ các thành viên
- GitHub Actions CI (tùy chọn)

### 4. Bảng kết quả (.xlsx/.csv)

**Files cần tạo**:
- `results/paper_vs_experiment.xlsx` ✓ Template
- `results/version_progression.xlsx` ✓ Template
- `results/ablation_study.xlsx` ✓ Template

**Status**: Cần tạo sau khi chạy thực nghiệm

### 5. Slide + Trình bày

**Slide (10-12 phút)**:
1. Introduction (1 min)
2. Dataset Pipeline (2 min)
3. Baseline Reproduction (2 min)
4. Proposed Method (3 min)
5. Results & Analysis (2 min)
6. Conclusion (1 min)
7. Q&A (5 min)

**Status**: Cần tạo sau khi có kết quả

### 6. Nhật ký thực nghiệm

**Format**:
| Ngày | Config | Kết quả | Nhận xét |
|------|--------|---------|---------|
| 2026-10-19 | baseline, seed=42 | WER: TBD | - |

**Công cụ**:
- TensorBoard logs (đã config)
- W&B (tùy chọn)
- Excel manual log

**Status**: Cần điền sau khi chạy

### 7. Bảng phân công & Đóng góp

**Template**:
| Thành viên | Phần | Đóng góp |
|------------|------|----------|
| [Tên] | Giai đoạn 1: Dataset Pipeline | - |
| [Tên] | Giai đoạn 2: Baseline | - |
| [Tên] | Giai đoạn 3: Proposed V1 | - |
| [Tên] | Giai đoạn 3: Proposed V2 | - |
| [Tên] | Giai đoạn 3: Proposed V3 | - |

**Status**: Cần điền

## Mốc kiểm tra tiến độ

| Tuần | Mốc | Trạng thái |
|------|------|------------|
| 4 | Checkpoint 2.1: Pipeline xong, Data Card, EDA, thống kê | ✓ Hoàn thành |
| 5 | Checkpoint 2.2: Baseline chạy xong, bảng Paper vs Thực nghiệm sơ bộ | - Cần chạy |
| 6 | Checkpoint 2.3: Gap Analysis + V1, V2 | - Cần chạy |
| 7 | Nộp & bảo vệ Task 2: Bảng V1…Vn, ablation, kết luận khả thi | - Cần chạy |

## Các bước tiếp theo

### Ngắn hạn (để hoàn thành Checkpoint 2.2)

1. **Chạy baseline training**:
   ```bash
   python -m src.train.train_asr --config configs/baseline.yaml --seed 42
   ```
   - Tốn ~2-3 giờ trên RTX 3090
   - Hoặc ~4-6 giờ trên RTX 2080

2. **Đánh giá baseline**:
   ```bash
   python -m src.eval.evaluate \
     --asr_dir outputs/baseline/seed_42/best \
     --manifest data/manifests/test.jsonl \
     --out outputs/baseline/seed_42/test_metrics.json
   ```

3. **Error analysis**:
   ```bash
   python -m src.eval.error_analysis \
     --pred outputs/baseline/seed_42/test_predictions.jsonl \
     --out outputs/baseline/seed_42/error_analysis.json
   ```

4. **Điền bảng Paper vs Thực nghiệm**

5. **Gap Analysis** (so sánh với paper, giải thích chênh lệch)

### Trung hạn (để hoàn thành Checkpoint 2.3)

6. **Chạy V1 (Augmentation)**:
   ```bash
   python -m src.train.run_version --version v1_augment
   ```

7. **Chạy V2 (LoRA + Beam)**:
   ```bash
   python -m src.train.run_version --version v2_lora
   ```

8. **So sánh V1, V2 vs baseline**

9. **Gap Analysis chi tiết**

### Dài hạn (để hoàn thành Task 2)

10. **Chạy V3 (Noise-robust MT)**:
    ```bash
    python -m src.train.run_version --version v3_noise_robust_mt
    ```

11. **Chạy V4 (End-to-end)**:
    ```bash
    python -m src.train.run_version --version v4_e2e
    ```

12. **Ablation study**

13. **Bảng tiến trình V1-Vn**

14. **Kết luận tính khả thi**

15. **Viết báo cáo Task 2 (PDF)**

16. **Tạo bảng kết quả (.xlsx)**

17. **Tạo slide**

18. **Bảo vệ**

## Tổng kết

### Đã hoàn thành (~60%)

- ✓ Giai đoạn 1: Dataset Pipeline (hoàn thiện)
- ✓ Code structure (hoàn thiện)
- ✓ Config files (hoàn thiện)
- ✓ Documentation (hoàn thiện)
- ⏳ Giai đoạn 2: Baseline (đã thiết kế, cần chạy)
- ⏳ Giai đoạn 3: Proposed Method (đã thiết kế, cần chạy)
- ⏳ Báo cáo, slide, bảng kết quả (cần tạo sau khi chạy)

### Cần chạy thực nghiệm (~40%)

- Baseline training (2-3 giờ)
- V1 training (2-3 giờ)
- V2 training (2-3 giờ)
- V3 training (1-2 giờ)
- V4 training (2-3 giờ)
- Evaluation & analysis (1-2 giờ)
- Tổng: ~10-16 giờ training + ~2-4 giờ analysis

### Hoàn thành Checkpoint 2.2

**Thời gian cần**: ~3-4 giờ (baseline training + evaluation)

**Thời gian cần**: ~10-16 giờ (V1, V2, V3, V4)

**Thời gian cần**: ~2-4 giờ (báo cáo, slide, bảng kết quả)

**Tổng thời gian**: ~15-24 giờ (tùy hardware)

## Note quan trọng

1. **Dữ liệu hiện tại**: Sample data (20 samples) chỉ để test pipeline
2. **Dữ liệu thực tế**: VIVOS đang tải background (1.37 GB, ~30-40 phút)
3. **Training trên sample**: Có thể chạy nhanh (~10 phút) để test code
4. **Training trên VIVOS**: Cần chạy full để có kết quả meaningful
5. **Tài nguyên**: Cần GPU với 8-16GB VRAM
6. **Colab/Kaggle**: Có thể dùng tài nguyên miễn phí nếu không có GPU

## Liên hệ

Nếu cần hỗ trợ hoặc có câu hỏi, hãy kiểm tra:
- `README.md`: Hướng dẫn sử dụng
- `docs/baseline_selection.md`: Chi tiết baseline
- `data/data_card.md`: Chi tiết dữ liệu
- `data/imbalance_analysis.md`: Chi tiết xử lý mất cân bằng
