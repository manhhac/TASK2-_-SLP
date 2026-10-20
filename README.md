# Task 2 — ASR tiếng Việt + Dịch sang tiếng Anh

**Baseline**: Whisper-small fine-tune ASR VI + MarianMT (opus-mt-vi-en) cascade.
**Proposed**: V1 (augment) → V2 (LoRA + beam) → V3 (noise-robust MT) → V4 (E2E so sánh).

**GitHub:** https://github.com/manhhac/TASK2-_-SLP

## Project Status

- ✅ **Giai đoạn 1**: Dataset Pipeline (hoàn thành) — VIVOS sau xử lý: 10.887 mẫu (train 8.253 / dev 1.084 / test 1.550).
- ✅ **Giai đoạn 2**: Baseline Reproduction (hoàn thành) — Whisper-small fine-tune + MarianMT cascade trên bộ VIVOS quick; WER 50,52 · CER 22,83 · BLEU 4,83.
- ✅ **Giai đoạn 3**: Proposed Method (hoàn thành) — V1 (augment) → V2 (LoRA + beam) → V3 (noise-robust MT) → V4 (E2E). **Phiên bản chọn: V3** (WER 43,30 · BLEU 22,73).

Xem chi tiết tiến độ ở [docs/TASK2_REPORT.md](docs/TASK2_REPORT.md).

## Cấu trúc Project

```
asr_vi_en_task2/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── v1_augment.yaml
│   ├── v2_lora.yaml
│   ├── v3_noise_robust_mt.yaml
│   └── v4_e2e.yaml
├── src/
│   ├── data/
│   │   ├── prepare_data.py
│   │   ├── text_normalize.py
│   │   ├── align_check.py
│   │   ├── translation_check.py
│   │   ├── dedup.py
│   │   ├── imbalance.py
│   │   ├── split.py
│   │   ├── augment.py
│   │   └── features.py
│   ├── models/
│   │   ├── asr_whisper.py
│   │   ├── mt_model.py
│   │   └── cascade.py
│   ├── train/
│   │   ├── train_asr.py
│   │   ├── train_mt.py
│   │   └── run_version.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   └── error_analysis.py
│   └── utils/
│       ├── seed.py
│       ├── config.py
│       └── logger.py
├── scripts/
│   ├── run_pipeline.sh
│   ├── run_pipeline.bat
│   └── run_all_versions.sh
└── notebooks/eda.ipynb
```

## Tái lập

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Tải dữ liệu train

**Cách 1: Tạo dữ liệu mẫu để test nhanh (Khuyên dùng cho demo)**
```bash
python scripts/create_sample_data.py --output_dir data/raw/vietnamese/sample --n_samples 20
```

**Cách 2: Tải VIVOS từ Zenodo (Dataset thực tế)**
```bash
python scripts/download_vivos_hf.py --output_dir data/raw/vietnamese/vivos
```

VIVOS là dataset tiếng Việt cho ASR với:
- 15 giờ audio
- 11,660 utterances
- 65 speakers
- License: CC BY-NC-SA 4.0

**Cách 3: Tải thủ công từ các nguồn khác**

- **VIVOS từ Zenodo**: https://zenodo.org/record/7068130
- **Mozilla Common Voice Vietnamese**: https://voice.mozilla.org/en/datasets
- **PhoMT (Vietnamese-English parallel corpus)**: https://github.com/VinAIResearch/PhoMT

Sau khi tải, đặt dữ liệu vào `data/raw/{vietnamese,english}/` với file `transcripts.tsv`:

```
audio_filename.txt	text_vi	text_en	speaker_id	gender	region
sample1.wav	xin chào	hello	unknown	male	north
...
```

### 3. Chạy pipeline dữ liệu

```bash
python -m src.data.prepare_data --config configs/baseline.yaml
```

Kết quả:
- `data/manifests/{train,dev,test}.jsonl`
- `data/features/feature_stats.json`
- `data/data_card.md`
- `data/manifests/pipeline_report.json`

### 3. Chạy pipeline baseline

**Linux/Mac:**
```bash
bash scripts/run_pipeline.sh
```

**Windows:**
```bash
scripts\run_pipeline.bat
```

### 4. Chạy các phiên bản experiment

```bash
# Chạy tất cả versions
bash scripts/run_all_versions.sh

# Hoặc chạy từng version
python -m src.train.run_version --version baseline
python -m src.train.run_version --version v1_augment
python -m src.train.run_version --version v2_lora
python -m src.train.run_version --version v3_noise_robust_mt
python -m src.train.run_version --version v4_e2e
```

## Các phiên bản Experiment

### 1. Baseline
- Whisper-small fine-tune trên tiếng Việt
- MarianMT cascade (opus-mt-vi-en)
- Không augmentation
- Greedy decode
- 3 seeds: 42, 123, 2024

### 2. V1: Augment
- Thêm 3 kỹ thuật augmentation:
  - Speed perturbation (0.9, 1.0, 1.1)
  - Gaussian noise (SNR 10-20dB)
  - Reverb (RT60 0.2-0.5s)
- Speaker cap (median × 3)
- Oversample minority language

### 3. V2: LoRA + Beam Search
- LoRA fine-tune (r=16, alpha=32)
- Beam search (size=5)
- Length penalty
- Combined với V1 augment

### 4. V3: Noise-Robust MT
- Fine-tune MarianMT trên ASR transcripts nhiễu
- Yêu cầu: chạy ASR inference trước để tạo noisy source
- Tăng robustness cho pipeline cascade

### 5. V4: End-to-End (Comparison)
- Whisper với task=translate (VI→EN trực tiếp)
- So sánh với cascade approach
- Không cần MT model riêng

## Metrics

### ASR Metrics
- **WER** (Word Error Rate)
- **CER** (Character Error Rate)

### MT Metrics
- **BLEU** (sacrebleu)
- **chrF++** (character-level F-score)

### Cascade Metrics
- RTF (Real-Time Factor)
- Error propagation analysis (oracle MT vs cascade MT)

## Config & Seeds

### Cấu hình các phiên bản
Tất cả config files nằm trong thư mục `configs/`:

- `baseline.yaml`: Config baseline cơ bản
- `v1_augment.yaml`: Config với augmentation
- `v2_lora.yaml`: Config với LoRA fine-tuning
- `v3_noise_robust_mt.yaml`: Config với noise-robust MT
- `v4_e2e.yaml`: Config với end-to-end translation

### Random Seeds
Để đảm bảo tái lập được, tất cả experiment sử dụng 3 seeds:
- Seed 42
- Seed 123
- Seed 2024

Seed được set ở:
- `src/utils/seed.py`: Set seed cho Python, NumPy, PyTorch
- Mỗi config file có field `seeds: [42, 123, 2024]`

### Logging
Kết quả training và evaluation được lưu trong:
- `outputs/{version}/seed_{seed}/logs/`: Training logs
- `outputs/{version}/seed_{seed}/metrics/`: Evaluation metrics
- `outputs/{version}/seed_{seed}/checkpoints/`: Model checkpoints

## Utils

### Kiểm tra alignment
```bash
python -m src.data.align_check --manifest data/manifests/test.jsonl
```

### Kiểm tra translation
```bash
python -m src.data.translation_check --manifest data/manifests/test.jsonl
```

### Loại trùng
```bash
python -m src.data.dedup --manifest data/manifests/train.jsonl --output data/manifests/train_dedup.jsonl
```

### Phân tích mất cân bằng
```bash
python -m src.data.imbalance --manifest data/manifests/train.jsonl --action analyze
```

### Đánh giá model
```bash
python -m src.eval.evaluate \
  --asr_dir outputs/baseline/seed_42/best \
  --manifest data/manifests/test.jsonl \
  --out outputs/baseline/seed_42/test_metrics.json
```

### Error analysis
```bash
python -m src.eval.error_analysis \
  --pred outputs/baseline/seed_42/test_predictions.jsonl \
  --out outputs/baseline/seed_42/error_analysis.json
```

## Quy tắc chuẩn hoá văn bản

1. Unicode NFC (quan trọng với tiếng Việt: tổ hợp dấu thanh)
2. Lowercase toàn bộ
3. Loại bỏ ký tự điều khiển, ký tự lạ không thuộc bảng chữ Việt/ASCII
4. Số: mặc định tách từng chữ số; có thể bật number_to_words để đọc thành lời
5. Mở rộng từ viết tắt phổ biến (tp., q., p., vn, hcm, ...)
6. Chuẩn hoá khoảng trắng (1 space), strip 2 đầu
7. Gộp dấu câu lặp (!!! → !, ... → .)
8. Dấu câu giữ lại: . , ! ? ; : ' " ( ) -

## Requirements

- Python >= 3.8
- PyTorch >= 2.1.0
- CUDA hỗ trợ (tùy chọn, recommended cho training)

## Bảng phân công

| Thành viên | MSSV | Phần phụ trách | Đóng góp |
|-----------|------|---------------|----------|
| Nguyễn Đức Mạnh | SE194525 | Tất cả các phần (Dataset Pipeline, Baseline, Proposed Method, Báo cáo) | 100% |

## Notes

- Model Whisper và MarianMT sẽ được download tự động từ HuggingFace Hub
- Training với FP16 yêu cầu GPU có hỗ trợ
- Memory requirement: ~8GB VRAM cho batch_size=16
- Thời gian training baseline: ~2-3 giờ trên single GPU (RTX 3090)

## License

This project uses pre-trained models from HuggingFace. Please check individual model licenses:
- Whisper: MIT License
- MarianMT: Apache 2.0 License
