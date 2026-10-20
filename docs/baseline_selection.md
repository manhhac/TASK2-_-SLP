# Baseline Selection & Description

## 1. Chọn Baseline: Whisper (OpenAI)

### 1.1 Lý do chọn

1. **State-of-the-art performance**:
   - Whisper đạt SOTA trên nhiều benchmarks ASR đa ngôn ngữ
   - Đặc biệt mạnh với tiếng Việt (được train trên ~680 giờ đa ngôn ngữ, bao gồm tiếng Việt)

2. **Mã nguồn mở**:
   - Code và model có sẵn trên HuggingFace
   - Dễ dàng tái lập và fine-tune
   - Cộng đồng lớn, nhiều resources

3. **Cascadable với MT**:
   - Whisper fine-tune ASR tiếng Việt
   - MarianMT (opus-mt-vi-en) dịch tiếng Việt → tiếng Anh
   - Pipeline cascade rõ ràng, dễ đánh giá từng component

4. **Tài nguyên khả thi**:
   - Whisper-small: 241M parameters → train được trên single GPU (8-16GB VRAM)
   - Không yêu cầu tài nguyên quá lớn
   - Thời gian training hợp lý (~2-3 giờ trên RTX 3090)

5. **Relevant với Task 1**:
   - Nếu Task 1 đã nghiên cứu Whisper, có thể tận dụng knowledge
   - Hoặc nếu Task 1 nghiên cứu model khác, Whisper là baseline tốt để so sánh

### 1.2 Mô tả chi tiết Whisper

#### 1.2.1 Kiến trúc

**Whisper Architecture**:
- **Encoder**: Transformer encoder (32 layers for base, 12 layers for small)
- **Decoder**: Transformer decoder with cross-attention
- **Input**: Log-Mel spectrogram (80 mel bands, 3000 frames)
- **Output**: Token sequence (via tokenizer)

**Whisper-small** (chọn cho baseline):
- **Parameters**: 241M
- **Encoder layers**: 12
- **Decoder layers**: 12
- **Hidden size**: 768
- **Attention heads**: 12
- **FFN dimension**: 3072

#### 1.2.2 Training Procedure (Original)

**Pre-training**:
- **Data**: 680,000 hours multilingual audio (96 languages)
- **Languages**: English (68%) + 95 other languages (32%, bao gồm tiếng Việt)
- **Task**: Speech recognition + translation (task embedding)
- **Objective**: Cross-entropy loss
- **Optimizer**: AdamW
- **Learning rate**: 1e-4 with cosine decay
- **Batch size**: 256
- **Warmup**: 2000 steps
- **Epochs**: Until convergence (~100 epochs)

**Fine-tuning** (khuyến nghị):
- **Learning rate**: 1e-5 (smaller than pre-training)
- **Batch size**: 16-32 (tùy GPU)
- **Epochs**: 5-10 (early stopping)
- **Gradient accumulation**: 2-4 (nếu batch size nhỏ)

#### 1.2.3 Tokenizer

**Whisper Tokenizer**:
- **Vocabulary size**: 51,865 tokens
- **Type**: Byte-level BPE (like GPT-2)
- **Special tokens**: Start of transcript, end of transcript, task tokens (transcribe, translate), language tokens

**Vietnamese Language Token**:
- Whisper có dedicated token cho tiếng Việt: `<|vi|>`
- Tự động detect language hoặc force via config

#### 1.2.4 Decoding Strategy

**Default**: Greedy decoding (beam size = 1)

**Options**:
- **Beam search**: beam size 5, length penalty
- **Sampling**: temperature sampling (cho creative tasks)
- **Conditional generation**: force language, force task

**For baseline**:
- Greedy decoding (beams=1)
- Force language=vi, task=transcribe

## 2. MarianMT (opus-mt-vi-en) cho Translation

### 2.1 Lý do chọn

1. **State-of-the-art for VI-EN**:
   - opus-mt-vi-en là model tốt nhất cho tiếng Việt → tiếng Anh
   - Fine-tune trên OPUS corpus (song ngữ lớn)

2. **Lightweight**:
   - ~300M parameters
   - Train/inference nhanh

3. **Cascadable**:
   - Input: Transcript tiếng Việt từ Whisper
   - Output: Text tiếng Anh
   - Dễ dàng đánh giá error propagation

### 2.2 Kiến trúc

**MarianMT Architecture**:
- **Type**: Transformer encoder-decoder
- **Parameters**: ~300M
- **Layers**: 6 encoder + 6 decoder
- **Hidden size**: 512
- **Attention heads**: 8

**Training**:
- **Data**: OPUS VI-EN parallel corpus
- **Objective**: Cross-entropy loss
- **Learning rate**: 1e-4 (pre-trained)

## 3. Pipeline Cascade

### 3.1 Architecture

```
Audio (16 kHz)
    ↓
Log-Mel Spectrogram (80 bands)
    ↓
Whisper Encoder
    ↓
Whisper Decoder → Vietnamese Text
    ↓
MarianMT Encoder
    ↓
MarianMT Decoder → English Text
```

### 3.2 Metrics

**ASR Metrics**:
- **WER** (Word Error Rate): (S + D + I) / (S + D + I + H)
- **CER** (Character Error Rate): Tương tự WER nhưng ở character level

**MT Metrics**:
- **BLEU** (Bilingual Evaluation Understudy): 0-100, cao hơn tốt hơn
- **chrF++** (character-level F-score): 0-100, cao hơn tốt hơn

**Cascade Metrics**:
- **RTF** (Real-Time Factor): inference_time / audio_duration
- **Latency**: time từ input audio đến output text
- **Error propagation**: BLEU(transcript_ASR) vs BLEU(transcript_ref)

## 4. Configuration Baseline

### 4.1 ASR Config (Whisper-small)

```yaml
model:
  asr_model: "openai/whisper-small"
  language: "vi"
  task: "transcribe"
  use_lora: false

train:
  output_dir: "outputs/baseline"
  num_epochs: 10
  batch_size: 16
  grad_accum: 2
  lr: 1.0e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
  fp16: true
  eval_steps: 500
  save_steps: 500
  logging_steps: 50
  use_augment: false
  use_beam_search: false
  beam_size: 1
```

### 4.2 MT Config (MarianMT)

```yaml
model:
  mt_model: "Helsinki-NLP/opus-mt-vi-en"

train:
  num_epochs: 5
  batch_size: 32
  lr: 3.0e-5
  fp16: true
```

## 5. Training Procedure

### 5.1 Hardware

**Minimum**:
- GPU: 8GB VRAM (RTX 2060, GTX 1080)
- RAM: 16GB
- Storage: 50GB (model + data + checkpoints)

**Recommended**:
- GPU: 16GB VRAM (RTX 3090, A100)
- RAM: 32GB
- Storage: 100GB SSD

### 5.2 Expected Training Time

**Whisper-small fine-tune**:
- 10 epochs, batch size 16, grad accum 2
- ~2-3 hours on RTX 3090
- ~4-6 hours on RTX 2080

**MarianMT** (nếu fine-tune):
- 5 epochs, batch size 32
- ~1-2 hours

### 5.3 Seeds

**Seeds**: [42, 123, 2024]

**Purpose**:
- Đảm bảo tái lập
- Báo cáo trung bình ± std khi có thể
- Chọn seed tốt nhất cho final model

## 6. Evaluation Protocol

### 6.1 Dev Set

**Purpose**:
- Chọn model checkpoint tốt nhất
- Tune hyperparameters
- Early stopping

**Metric**: WER (ASR primary)

### 6.2 Test Set

**Purpose**:
- Báo cáo kết quả cuối
- So sánh với paper
- Không dùng để chọn model (không data leakage)

**Metrics**:
- WER, CER (ASR)
- BLEU, chrF++ (MT)
- RTF, latency

### 6.3 Oracle Analysis

**Oracle MT**:
- Dịch từ transcript chuẩn (not ASR output)
- Đo error propagation từ ASR → MT
- BLEU_oracle - BLEU_cascade = error propagation

## 7. Reference Paper

### 7.1 Whisper Paper

**Title**: Robust Speech Recognition via Large-Scale Weak Supervision

**Authors**: Alec Radford, et al. (OpenAI)

**Conference**: (arXiv preprint, 2022)

**Link**: https://arxiv.org/abs/2212.04356

**Key Results (Whisper-small)**:
- LibriSpeech test-clean: WER 3.2%
- LibriSpeech test-other: WER 7.6%
- Multilingual: WER varies by language

**Vietnamese specific**:
- No direct Vietnamese benchmark in paper
- Trained on Vietnamese data (~2% of total)
- Expected WER: ~10-15% on clean read speech

### 7.2 MarianMT Paper

**Title**: OPUS-MT — Building massively parallel multilingual machine translation models

**Authors**: Jörg Tiedemann & Santhosh Thottingal

**Conference**: EMNLP 2020

**Link**: https://aclanthology.org/2020.emnlp-main.623/

**Key Results (opus-mt-vi-en)**:
- BLEU: ~30-35 on VI-EN benchmarks
- State-of-the-art for VI-EN translation

## 8. Differences from Paper

### 8.1 Expected Differences

| Aspect | Paper | Nhóm (Expected) | Reason |
|--------|-------|-----------------|--------|
| Dataset | 680k hours multilingual | 15 hours Vietnamese only | Dữ liệu ít hơn |
| Domain | Diverse domains | Read speech only | Domain mismatch |
| Hardware | T8/v4-512 | Single GPU (RTX 3090) | Tài nguyên hạn chế |
| Fine-tuning | Full fine-tune | Full fine-tune | Same |
| Epochs | 100+ | 10 | Tài nguyên |

### 8.2 Mitigation

1. **Data scarcity**:
   - Focus on fine-tuning quality
   - Use data augmentation
   - Monitor overfitting

2. **Domain mismatch**:
   - Document limitation
   - Test on conversational data (nếu có)
   - Error analysis per domain

3. **Resource constraints**:
   - Use smaller model (Whisper-small)
   - Fewer epochs with early stopping
   - Gradient accumulation to simulate larger batch

## 9. Success Criteria

### 9.1 Technical Success

- [ ] Baseline runs without errors
- [ ] Converges (WER decreases)
- [ ] Reproducible (same seed → same result)
- [ ] Metrics reasonable (WER < 30%, BLEU > 20)

### 9.2 Documentation Success

- [ ] Bảng Paper vs Thực nghiệm đầy đủ
- [ ] Gap Analysis có bằng chứng
- [ ] Error Analysis chi tiết
- [ ] Config và code tái lập được

## 10. Next Steps

1. Chạy baseline với dữ liệu mẫu (đã có manifests)
2. Tạo bảng Paper vs Thực nghiệm
3. Chạy gap analysis (so sánh với paper + error analysis)
4. Nếu baseline hợp lý → tiếp tục Giai đoạn 3 (Proposed Method)
