# Hướng dẫn Training Baseline

## 1. Chuẩn bị

### 1.1 Cài đặt dependencies

```bash
cd asr_vi_en_task2
pip install -r requirements.txt
```

**Lưu ý**: `webrtcvad` có thể không cài được trên Windows, nhưng không bắt buộc cho training.

### 1.2 Chuẩn bị dữ liệu

**Option A: Dữ liệu mẫu (để test code)**
```bash
python scripts/create_sample_data.py --output_dir data/raw/vietnamese/sample --n_samples 20
python -m src.data.prepare_data --config configs/baseline.yaml
```

**Option B: Dữ liệu VIVOS thực tế**
```bash
python scripts/download_vivos_hf.py --output_dir data/raw/vietnamese/vivos
# Sau khi tải xong, cập nhật config:
# data.raw_dirs: ["data/raw/vietnamese/vivos"]
python -m src.data.prepare_data --config configs/baseline.yaml
```

## 2. Chạy Training Baseline

### 2.1 Chạy với module (Khuyên dùng)

```bash
cd asr_vi_en_task2
python -m src.train.train_asr --config configs/baseline.yaml --seed 42
```

### 2.2 Chạy file trực tiếp

```bash
cd asr_vi_en_task2
python src/train/train_asr.py --config configs/baseline.yaml --seed 42
```

### 2.3 Tham số

| Tham số | Mô tả | Giá trị |
|---------|-------|--------|
| `--config` | File config YAML | Cần chỉ định |
| `--seed` | Random seed | 42, 123, 2024 |
| `--asr_only` | Chỉ train ASR, không MT | (optional) |

## 3. Các phiên bản khác

### 3.1 V1: Augmentation

```bash
python -m src.train.train_asr --config configs/v1_augment.yaml --seed 42
```

### 3.2 V2: LoRA + Beam Search

```bash
python -m src.train.train_asr --config configs/v2_lora.yaml --seed 42
```

### 3.3 V3: Noise-robust MT

```bash
# Bước 1: Chạy ASR trước để tạo noisy transcripts
python -m src.train.train_asr --config configs/v3_noise_robust_mt.yaml --seed 42

# Bước 2: Chạy MT với noisy source
python -m src.train.train_mt --config configs/v3_noise_robust_mt.yaml --seed 42 --asr_dir outputs/v3/seed_42/best
```

### 3.4 Chạy tất cả versions cùng lúc

```bash
python -m src.train.run_version --version all
```

Hoặc từng version:
```bash
python -m src.train.run_version --version baseline
python -m src.train.run_version --version v1_augment
python -m src.train.run_version --version v2_lora
python -m src.train.run_version --version v3_noise_robust_mt
```

## 4. Expected Training Time

| Version | Dataset Size | Expected Time (RTX 3090) | Expected Time (RTX 2080) |
|---------|--------------|---------------------------|--------------------------|
| Baseline (sample 20) | 20 samples | ~10 phút | ~15 phút |
| Baseline (VIVOS) | 8,200 samples | ~2-3 giờ | ~4-6 giờ |
| V1 (VIVOS) | 8,200 samples | ~2-3 giờ | ~4-6 giờ |
| V2 (VIVOS) | 8,200 samples | ~2-3 giờ | ~4-6 giờ |
| V3 (VIVOS) | 8,200 samples | ~1-2 giờ (MT) | ~2-3 giờ (MT) |

## 5. Monitor Training

### 5.1 TensorBoard

Training sẽ tự động log metrics lên TensorBoard. Để xem:

```bash
tensorboard --logdir outputs/baseline/seed_42
```

Mở browser tại: http://localhost:6006

### 5.2 Console output

Training sẽ in progress mỗi 50 steps:
```
Step 50: loss=..., wer=..., cer=...
Step 100: loss=..., wer=..., cer=...
...
```

## 6. Output Files

Sau khi training xong, các file sẽ được lưu tại:

```
outputs/
└── baseline/
    └── seed_42/
        ├── best/                    # Model checkpoint tốt nhất
        │   ├── config.json
        │   ├── pytorch_model.bin
        │   ├── tokenizer.json
        │   └── ...
        ├── checkpoint-500/         # Checkpoint 500 steps
        ├── checkpoint-1000/
        └── ...
```

## 7. Đánh giá Model

### 7.1 Đánh giá trên test set

```bash
python -m src.eval.evaluate \
  --asr_dir outputs/baseline/seed_42/best \
  --manifest data/manifests/test.jsonl \
  --out outputs/baseline/seed_42/test_metrics.json
```

### 7.2 Error Analysis

```bash
python -m src.eval.error_analysis \
  --pred outputs/baseline/seed_42/test_predictions.jsonl \
  --out outputs/baseline/seed_42/error_analysis.json
```

## 8. Troubleshooting

### 8.1 Out of Memory (OOM)

**Lỗi**: `CUDA out of memory`

**Giải pháp**:
- Giảm batch size trong config:
  ```yaml
  train:
    batch_size: 8  # hoặc 4
    grad_accum: 4  # tăng gradient accumulation
  ```

### 8.2 Dữ liệu không tìm thấy

**Lỗi**: `FileNotFoundError: data/manifests/train.jsonl`

**Giải pháp**:
- Chạy `prepare_data.py` trước:
  ```bash
  python -m src.data.prepare_data --config configs/baseline.yaml
  ```

### 8.3 Model download lỗi

**Lỗi**: Network error khi download Whisper model

**Giải pháp**:
- Download thủ công từ HuggingFace:
  ```bash
  huggingface-cli download openai/whisper-small --local-dir models/whisper-small
  ```
- Cập nhật config:
  ```yaml
  model:
    asr_model: "models/whisper-small"
  ```

### 8.4 Training không giảm loss

**Lỗi**: Loss không giảm, WER không cải thiện

**Giải pháp**:
- Kiểm tra learning rate (giảm nếu cần)
- Kiểm tra data quality
- Tăng số epochs
- Check Overfitting (train loss << dev loss)

## 9. Tips

### 9.1 Dùng GPU

Training cần GPU với ít nhất 8GB VRAM. Nếu không có:
- Sử dụng Google Colab (free tier)
- Sử dụng Kaggle Kernels
- Sử dụng cloud GPU (AWS, GCP, Azure)

### 9.2 Chạy background

Để chạy training trong background (Linux/Mac):
```bash
nohup python -m src.train.train_asr --config configs/baseline.yaml --seed 42 > train.log 2>&1 &
```

Để chạy background (Windows PowerShell):
```powershell
Start-Process python -ArgumentList "-m","src.train.train_asr","--config","configs/baseline.yaml","--seed","42" -NoNewWindow
```

### 9.3 Resume training

Nếu training bị gián đoạn, có thể resume từ checkpoint:
```python
# Trong train_asr.py, đã có load_best_model_at_end=True
# Trainer sẽ tự động resume từ checkpoint gần nhất
```

## 10. Next Steps sau Training

1. **Kiểm tra metrics**: Đọc file `test_metrics.json`
2. **Điền bảng Paper vs Thực nghiệm**
3. **Gap Analysis**: So sánh với paper, giải thích chênh lệch
4. **Error Analysis**: Phân loại lỗi S/D/I, tone errors
5. **Chạy V1, V2, V3**: So sánh với baseline
6. **Bảng tiến trình**: Điền kết quả vào bảng
7. **Kết luận**: Đánh giá tính khả thi

## 11. Ví dụ Workflow Đầy đủ

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Chuẩn bị dữ liệu
python scripts/create_sample_data.py --output_dir data/raw/vietnamese/sample --n_samples 20
python -m src.data.prepare_data --config configs/baseline.yaml

# 3. Chạy baseline (3 seeds)
for seed in 42 123 2024; do
  python -m src.train.train_asr --config configs/baseline.yaml --seed $seed
done

# 4. Đánh giá (seed tốt nhất)
python -m src.eval.evaluate \
  --asr_dir outputs/baseline/seed_42/best \
  --manifest data/manifests/test.jsonl \
  --out outputs/baseline/seed_42/test_metrics.json

# 5. Error analysis
python -m src.eval.error_analysis \
  --pred outputs/baseline/seed_42/test_predictions.jsonl \
  --out outputs/baseline/seed_42/error_analysis.json

# 6. Chạy V1
python -m src.train.run_version --version v1_augment

# 7. Chạy V2
python -m src.train.run_version --version v2_lora

# 8. So sánh kết quả
# Điền vào bảng tiến trình
```

## 12. Liên hệ

Nếu gặp lỗi không giải quyết được:
- Kiểm tra `docs/TASK2_SUMMARY.md` cho tổng quan
- Kiểm tra `docs/baseline_selection.md` cho chi tiết baseline
- Kiểm tra `data/data_card.md` cho chi tiết dữ liệu
- Kiểm tra `README.md` cho hướng dẫn sử dụng
