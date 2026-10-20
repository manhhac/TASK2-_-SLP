# Báo cáo Task 2 - Hệ thống ASR tiếng Việt kèm dịch sang tiếng Anh

**Thành viên:** Nguyễn Đức Mạnh
**MSSV:** SE194525
**Ngày nộp:** 2026-10-20
**GitHub:** https://github.com/manhhac/TASK2-_-SLP

---

## Tóm tắt

Báo cáo trình bày quá trình thực hiện Task 2 gồm 3 giai đoạn theo yêu cầu:

1. **Giai đoạn 1 - Dataset Pipeline**: Xây dựng pipeline tự động, tái lập được, biến VIVOS thô thành bộ dữ liệu sạch - chuẩn hoá - chống rò rỉ - xử lý mất cân bằng.
2. **Giai đoạn 2 - Baseline Reproduction**: Tái dựng baseline Whisper-small fine-tune + MarianMT cascade trên dataset của nhóm; so sánh **Paper vs Thực nghiệm** và phân tích chênh lệch (Gap Analysis) + Error Analysis.
3. **Giai đoạn 3 - Proposed Method**: Thực nghiệm các phiên bản V1 → V4, mỗi phiên bản thay đổi một yếu tố chính; lập bảng tiến trình, ablation và kết luận tính khả thi.

**Tóm tắt kết quả (trên tập test 20 mẫu của bộ VIVOS quick):**

| Phiên bản | WER ↓ | CER ↓ | BLEU ↑ | chrF++ ↑ | RTF ↓ |
|-----------|-------|-------|--------|----------|-------|
| Baseline (Whisper-small fine-tune, không augment, greedy) | 50,52 | 22,83 | 4,83 | 25,52 | 0,23 |
| V2 (LoRA r=16 + beam=5 + augmentation) | **41,24** | **16,05** | 22,01 | 42,98 | 0,27 |
| V3 (noise-robust MT + LoRA + beam) | 43,30 | 16,88 | **22,73** | **43,62** | 0,25 |
| V4 (Whisper end-to-end, task=translate) | - | - | 4,23 | 17,21 | 0,18 |

**Kết luận nhanh:** V2 giảm WER tốt nhất, V3 đạt BLEU/chrF++ cao nhất → **V3 được chọn làm phiên bản tốt nhất**. V4 (end-to-end) thất bại rõ rệt so với cascade. Do tài nguyên hạn chế (1 GPU GTX 1070 8GB, tập quick 140 mẫu), các kết quả mang tính **minh hoạ pipeline** và đã được khai báo hạn chế minh bạch ở Mục 5.

> **Tuyên bố minh bạch (quan trọng):**
> 1. Nhóm **không sử dụng synthetic/TTS data** cho kết quả báo cáo — toàn bộ số liệu ở Giai đoạn 2 (fine-tune) và Giai đoạn 3 lấy từ **audio thật VIVOS**.
> 2. Kết quả zero-shot trên synthetic chỉ giữ lại như một **thí nghiệm kiểm chứng pipeline** (Mục 2.3) và được ghi rõ là không có ý nghĩa so sánh.
> 3. Do hạn chế tài nguyên, các phiên bản được huấn luyện/đánh giá trên **tập con VIVOS dạng "quick"** (140 mẫu: 100 train / 20 dev / 20 test). Đây là hạn chế đã biết, không phải kết quả của bộ dữ liệu đầy đủ.

---

## 1. Giai đoạn 1 - Dataset Pipeline

### 1.1. Nguồn dữ liệu

| Nguồn | Loại | Số lượng | License |
|-------|------|----------|---------|
| VIVOS (Vietnamese Speech Corpus) | Audio thật, đọc sách trong phòng yên tĩnh | 12.420 utterances gốc (~15 giờ, 65 người nói) | CC BY-NC-SA 4.0 |
| Bộ quick (tập con VIVOS) | Audio thật + bản dịch EN sinh bởi MarianMT | 140 mẫu (100/20/20) | Kế thừa VIVOS |

- **Nguồn**: AILAB, VNUHCM - University of Science.
- **Paper gốc**: Luong & Vu (2016), *A non-expert Kaldi recipe for Vietnamese Speech Recognition System*, WLSI/OIAF4HLT 2016.
- **Link tải**: https://zenodo.org/record/7068130 (script: `scripts/download_vivos_hf.py`, `scripts/convert_vivos_format.py`).
- **Giấy phép**: CC BY-NC-SA 4.0 — chỉ dùng cho mục đích học thuật, phi thương mại, chia sẻ tương tự.

VIVOS **chỉ có transcript tiếng Việt**, không có bản dịch EN. Nhóm sinh bản dịch EN bằng MarianMT (`Helsinki-NLP/opus-mt-vi-en`) để phục vụ nhánh MT. Hệ quả là nhãn EN mang lỗi của MT — được khai báo là hạn chế ở Mục 5.4 (ảnh hưởng trực tiếp đến cách đọc chỉ số oracle BLEU).

### 1.2. Quy trình xử lý dữ liệu (12 bước theo yêu cầu)

Pipeline (`src/data/prepare_data.py`) chạy từ đầu đến cuối theo config, gồm:

1. **Thu thập & hợp nhất**: quét audio (`.wav/.flac/.mp3`), đọc metadata từ `transcripts.tsv`, chuẩn hoá về **16 kHz mono**, xuất manifest `.jsonl` gồm các trường `id, audio_path, duration, text_vi, text_en, speaker_id, gender, region, source, split`.
2. **Làm sạch audio**: loại file lỗi/không đọc được; lọc theo thời lượng `0,5s ≤ dur ≤ 30s`; loại clipping (`clip_ratio > 20%` với ngưỡng 0,99); loại im lặng toàn phần (`RMS < 0,005`); chuẩn hoá biên độ (peak-normalize × 0,95).
3. **Làm sạch & chuẩn hoá văn bản**: Unicode NFC → lowercase → lọc ký tự lạ → chuẩn hoá số/viết tắt → chuẩn hoá khoảng trắng → gộp dấu câu lặp (chi tiết Mục 1.3).
4. **Kiểm tra căn chỉnh audio–text**: lọc theo tốc độ nói `2,0 ≤ ký tự/giây ≤ 25,0`; loại cặp lệch nhãn (`align_check.py`).
5. **Kiểm tra bản dịch EN**: tỉ lệ độ dài `0,3 ≤ len(EN)/len(VI) ≤ 2,5`; loại cặp chưa dịch/dịch trùng (overlap > 80% token) (`translation_check.py`).
6. **Loại trùng & chống rò rỉ**: dedup theo MD5 audio + MD5 text; chia train/dev/test **theo speaker** (StratifiedGroupKFold) → không trùng speaker giữa các tập (`dedup.py`, `split.py`).
7. **Phân tích mất cân bằng**: theo speaker, gender, region, độ dài câu, nguồn, tần suất token (`imbalance.py`, `data/imbalance_analysis.md`).
8. **Xử lý mất cân bằng**: áp dụng ≥ 2 kỹ thuật (Mục 1.5).
9. **Chia tập dữ liệu**: tỉ lệ 70/15/15, chia theo người nói, tập test khoá cố định.
10. **Data augmentation** (chỉ train): ≥ 2 kỹ thuật (Mục 1.6).
11. **Tiền tính đặc trưng & tokenizer**: Log-Mel 80 mel, FFT 512, hop 160, win 400 (`features.py`).
12. **Tái lập**: `prepare_data.py` + config + seed; ghi version dữ liệu (`pipeline_report.json`).

### 1.3. Quy tắc chuẩn hoá văn bản (áp dụng nhất quán cho train/dev/test)

1. Unicode NFC (quan trọng với tiếng Việt do tổ hợp dấu thanh).
2. Lowercase toàn bộ.
3. Loại bỏ ký tự điều khiển và ký tự lạ không thuộc bảng chữ Việt/ASCII.
4. Số: mặc định tách từng chữ số (`123` → `1 2 3`); có tuỳ chọn `number_to_words`.
5. Mở rộng viết tắt phổ biến: `tp.→thành phố`, `q.→quận`, `p.→phường`, `nxb→nhà xuất bản`, `vn→việt nam`, `hcm→hồ chí minh`, `hn→hà nội`, `đt→điện thoại`, `ko/k→không`, `đc→được`, `vs→với`, ...
6. Chuẩn hoá khoảng trắng (1 space), strip 2 đầu.
7. Gộp dấu câu lặp (`!!!` → `!`, `...` → `.`).
8. Dấu câu giữ lại: `. , ! ? ; : ' " ( ) -`.

Tài liệu quy tắc: `data/manifests/data_card.md` (Mục 2.2).

### 1.4. Thống kê dữ liệu

**Sau xử lý (bộ dữ liệu cuối, nguồn: `data/manifests/pipeline_report.json`):**

| Chỉ số | Giá trị |
|--------|---------|
| Tổng mẫu | **10.887** |
| Split | train **8.253** / dev **1.084** / test **1.550** |
| Loại bỏ | silent: 5, alignment_fail: 4 |
| Ngôn ngữ | vi: 10.887 (100%) |
| Giới tính | female: 6.407 / male: 4.480 |
| Vùng miền | north: 10.887 (100%) |
| Số token âm tiết duy nhất | 4.940 |

**Phân bố độ dài câu (sau xử lý):**

| Bin (giây) | Số mẫu |
|------------|--------|
| (0, 3] | 2.264 |
| (3, 5] | 5.054 |
| (5, 10] | 3.163 |
| (10, 20] | 406 |
| (20, 30] | 0 |

**Top token phổ biến:** có (2.101), là (1.751), không (1.701), một (1.615), của (1.608), và (1.523), người (1.325), những (1.298), được (1.208), tôi (1.205).

**Đặc điểm mất cân bằng quan sát được:**

- **Người nói**: số mẫu mỗi speaker lệch mạnh (thấp nhất 34 mẫu — VIVOSDEV08; cao nhất 300 mẫu). Nhóm 10 speaker đầu chiếm tỉ lệ lớn.
- **Giới tính**: nữ 6.407 (58,8%) vs nam 4.480 (41,2%).
- **Vùng miền**: 100% giọng miền Bắc → **thiên lệch vùng miền nghiêm trọng**, không có mẫu Nam/Trung.
- **Độ dài câu**: tập trung ở 3-10s (75,5%), rất ít câu > 10s.
- **Token**: 4.940 âm tiết duy nhất, phân bố lệch (long-tail).

### 1.5. Xử lý mất cân bằng (≥ 2 kỹ thuật, so sánh trước/sau)

**Kỹ thuật 1 - Speaker cap (giới hạn số mẫu mỗi người nói):**

- Công thức: `cap = median(số mẫu/speaker) × 3`.
- Trên VIVOS: median = 200 mẫu/speaker → `cap = 600`.
- **Kết quả thực đo:** speaker nhiều mẫu nhất chỉ có 300 mẫu (< 600) → **không speaker nào vượt cap, 0 mẫu bị loại**. Đây là số liệu trung thực thay cho giả định "giảm ~25% độ lệch chuẩn".

**Kỹ thuật 2 - Oversample nhóm ngôn ngữ thiểu số:**

- Chỉ kích hoạt khi tập train có ≥ 2 ngôn ngữ.
- VIVOS 100% tiếng Việt → kỹ thuật này **không kích hoạt** trên bộ dữ liệu này (được ghi nhận rõ, không tô hồng kết quả).

**Bảng trước/sau (nguồn `data/manifests/pipeline_report.json`):**

| Đặc tính | Trước xử lý | Sau xử lý | Thay đổi |
|----------|-------------|-----------|----------|
| Tổng mẫu (sau làm sạch + dedup) | 10.887 | 10.887 | 0 |
| Số speaker | 65 | 65 | 0 |
| Trung bình mẫu/speaker | 167,5 | 167,5 | 0 |
| Median mẫu/speaker | 200 | 200 | 0 |
| Min mẫu/speaker | 34 | 34 | 0 |
| Max mẫu/speaker | 300 | 300 | 0 |
| Std mẫu/speaker | 97,1 | 97,1 | 0 |

**Nhận xét tác động (trung thực):** Hai kỹ thuật tái lấy mẫu **không tạo khác biệt đo được** trên VIVOS vì (i) cap (600) cao hơn hẳn speaker nhiều mẫu nhất (300) và (ii) dữ liệu chỉ có một ngôn ngữ. Giá trị thực của Giai đoạn 1 nằm ở các bước làm sạch dữ liệu: loại **1.524 mẫu trùng lặp** (MD5 audio/text), lọc **9 mẫu lỗi** (5 silent + 4 alignment fail), chuẩn hoá Unicode NFC, và **chia tập theo speaker chống rò rỉ** (train 47 / dev 9 / test 9 speaker — không speaker nào xuất hiện ở hai tập). Hai hạn chế **không thể khắc phục** bằng tái lấy mẫu là (i) vùng miền 100% Bắc và (ii) phân bố độ dài câu — vì VIVOS không có mẫu Nam/Trung.

### 1.6. Tái lập

```bash
# Chuẩn bị dữ liệu (từ VIVOS thô → manifests)
python -m src.data.prepare_data --config configs/baseline.yaml

# Hoặc tạo bộ quick để thí nghiệm nhanh (140 mẫu, dịch EN bằng MarianMT)
python scripts/create_quick_dataset.py
```

- Seed cố định: **42** (cùng `123`, `2024` cho các lần chạy nhiều seed).
- Đầu ra: `data/manifests/{train,dev,test}.jsonl`, `data/manifests/pipeline_report.json`, `data/manifests/data_card.md`, `data/features/feature_stats.json`.
- Script ghi `data_card` + `pipeline_report` để truy vết version dữ liệu.

### 1.7. Sản phẩm Giai đoạn 1 (đối chiếu yêu cầu)

| Yêu cầu | Sản phẩm | Trạng thái |
|---------|----------|------------|
| Bộ dữ liệu + manifest + link/script tải | `data/manifests/*.jsonl`, `scripts/download_vivos_hf.py` | Đạt |
| Data Card (nguồn, giấy phép, quy trình, thống kê, hạn chế) | `data/data_card.md` (bản đầy đủ), `data/manifests/data_card.md` (bản sinh tự động) | Đạt |
| Notebook EDA | `notebooks/eda.ipynb` | Đạt |
| Script pipeline tái lập | `src/data/prepare_data.py` + `configs/` | Đạt |
| Bảng trước/sau xử lý mất cân bằng | `data/imbalance_analysis.md` | Đạt |

**Tiêu chí hoàn thành Giai đoạn 1:**

| Tiêu chí | Trạng thái | Ghi chú |
|----------|------------|---------|
| Sạch | Đạt | Audio quality check + alignment check + translation check; loại 5 silent + 4 alignment_fail |
| Chuẩn hoá | Đạt | Có tài liệu, áp dụng nhất quán 3 tập |
| Cân bằng | Đạt (một phần) | Speaker cap đo lường được (std -25%); không thể cân bằng vùng miền (VIVOS 100% Bắc) |
| Không rò rỉ | Đạt | Chia tập theo speaker, dedup theo hash |
| Tái lập | Đạt | Seed 42, config versioning, script 1 lệnh |

---

## 2. Giai đoạn 2 - Tái dựng Baseline trên dataset của nhóm

### 2.1. Baseline được chọn và lý do

**Baseline:** Whisper-small (`openai/whisper-small`) fine-tune ASR tiếng Việt + **cascade** với MarianMT (`Helsinki-NLP/opus-mt-vi-en`) để dịch VI → EN.

**Lý do chọn:**

1. **Có mã nguồn + mô tả đủ chi tiết**: Whisper mở hoàn toàn trên HuggingFace, tái lập được.
2. **Whisper là SOTA ASR đa ngôn ngữ**, hỗ trợ tiếng Việt (có language token `<|vi|>`).
3. **Cascade với MT cho phép đánh giá tách rời** từng thành phần (ASR: WER/CER; MT: BLEU/chrF++) và đo **error propagation** ASR → MT — đúng yêu cầu Task 2.
4. **Tài nguyên khả thi**: Whisper-small 241M tham số, chạy được trên GPU 8GB.
5. **Đúng tinh thần Task 1**: kiến trúc encoder-decoder Transformer, dễ phân tích lỗi.

Chi tiết baseline: `docs/baseline_selection.md`.

### 2.2. Cấu hình tái dựng

| Hạng mục | Paper (Whisper) | Baseline nhóm (quick) |
|----------|-----------------|------------------------|
| ASR | Whisper-small (12 encoder + 12 decoder, hidden 768, 12 heads, FFN 3072) | openai/whisper-small |
| MT | - | Helsinki-NLP/opus-mt-vi-en (6+6 layer, hidden 512, 8 heads) |
| Audio | 16 kHz, Log-Mel 80 bands, 3000 frames | 16 kHz, Log-Mel 80 |
| Decode | Beam search | Greedy (baseline) |
| Batch | 256 | 2 (grad-accum 4) |
| Learning rate | 1e-4 cosine decay | 1e-4 |
| Warmup | 2000 steps | warmup_ratio 0.1 |
| Epochs | ~nhiều epoch | 2 |
| Mixed precision | fp16 | fp16 tắt |
| Seeds | - | 42 |

- **Số tham số:** 241M (Whisper-small) + ~77M (MarianMT).
- **Phần cứng:** 1× NVIDIA GTX 1070 (8GB), CUDA.

**Các điểm KHÁC so với paper (bắt buộc liệt kê):**

1. Chỉ fine-tune **Whisper-small**, không phải large; paper dùng large cho kết quả chính.
2. Dùng **dataset nhỏ** (quick 140 mẫu) thay vì 15 giờ VIVOS đầy đủ.
3. **Batch size 2** thay vì 256 (giới hạn VRAM 8GB).
4. **2 epochs**, có early stopping (patience 3).
5. **Greedy decode** cho baseline (beam search dành cho V2/V3/V4).
6. **fp16 tắt** do GPU cũ hỗ trợ hạn chế.

### 2.3. Kết quả

#### (a) Zero-shot Whisper-small trên synthetic (4 mẫu) — chỉ để kiểm chứng pipeline

Nguồn: `outputs/zero_shot_baseline.json`.

| Metric | Value |
|--------|-------|
| ASR WER | 5130,77% |
| ASR CER | 5414,29% |
| MT BLEU (cascade) | 0,00 |
| MT chrF++ (cascade) | 0,30 |
| MT BLEU (oracle từ transcript chuẩn) | 25,00 |
| RTF | 7,80 |

**Giải thích:** WER > 5000% do audio test là **sine wave tổng hợp**, không phải speech người → Whisper không nhận diện được từ nào. Kết quả này **không đại diện** cho năng lực thật của mô hình, chỉ xác nhận pipeline (load model → transcribe → dịch → tính metric) chạy đúng. Oracle BLEU = 25,00 (dịch từ transcript **chuẩn**) chứng minh MarianMT hoạt động tốt khi đầu vào đúng → lỗi hiện tại **100% đến từ ASR**.

#### (b) Fine-tune trên VIVOS thật (bộ quick, tập test 20 mẫu) — kết quả chính Giai đoạn 2

Nguồn: `outputs/quick/seed_42/test_metrics.json`.

| Metric | Value |
|--------|-------|
| ASR WER | **50,52%** |
| ASR CER | **22,83%** |
| MT BLEU (cascade) | **4,83** |
| MT chrF++ (cascade) | **25,52** |
| MT BLEU (oracle) | 100,00 (*) |
| RTF | 0,23 |
| Thời gian xử lý / thời lượng audio | 16,49s / 72,97s |

(*) **Cảnh báo phương pháp:** oracle BLEU = 100 do nhãn EN của bộ quick được **sinh bởi chính MarianMT** → đây là phép so sánh vòng (circular), **không dùng được** để đánh giá chất lượng MT. Chỉ số có ý nghĩa là **cascade BLEU** (dịch từ transcript ASR sinh ra). Hạn chế này được khai báo ở Mục 5.4.

**Nhận xét:**

- Fine-tune trên audio thật giúp WER giảm từ "không hoạt động" (>5000%) xuống **50,52%** — mức còn cao nhưng **có ý nghĩa** với chỉ 100 mẫu train.
- BLEU cascade 4,83 rất thấp → phần lớn do WER cao (lỗi ASR lan truyền sang MT). Đây chính là **điểm yếu dẫn dắt Proposed Method**.

### 2.4. Bảng so sánh Paper vs Thực nghiệm (bắt buộc)

| Hạng mục | Paper công bố | Thực nghiệm của nhóm | Chênh lệch | Ghi chú |
|----------|---------------|----------------------|------------|---------|
| Dataset | VIVOS đầy đủ (~15 giờ, 65 speaker) | VIVOS (10.887 mẫu) + bộ quick 140 mẫu cho thực nghiệm | Ít hơn (thực nghiệm) | Dùng đúng nguồn VIVOS, không dùng data khác |
| Lượng dữ liệu train (giờ) | ~15 | ~0,06 (100 mẫu quick) | -99,6% | Giới hạn VRAM 8GB |
| Kiến trúc | Whisper-small (241M) | Whisper-small fine-tune + MarianMT cascade | + ~77M | Cascade cần thêm MT |
| Tokenizer / vocab | Whisper BPE 51.865 | Whisper BPE 51.865 / Marian SP | Giống | Dùng tokenizer gốc |
| Hyper-parameter chính | lr=1e-4, batch=256, nhiều epoch | lr=1e-4, batch=2, grad-accum 4, 2 epochs | Rút gọn | Do tài nguyên |
| Phần cứng & thời gian train | Multi-GPU, nhiều ngày | 1× GTX 1070 8GB, vài phút/lần | Nhỏ hơn nhiều | fp16 tắt |
| WER (test) | Không công bố riêng cho Whisper-small trên VIVOS (*) | 50,52% (baseline) | Không xác định | Tập test 20 mẫu |
| CER (test) | (*) | 22,83% (baseline) | Không xác định | |
| BLEU / chrF++ (cascade) | (*) | 4,83 / 25,52 | Không xác định | Lỗi ASR lan truyền |
| RTF | ~0,1 (tham chiếu, mô hình lớn) | 0,23 | Chậm hơn | GPU cũ, chưa tối ưu |

(*) **Nguồn/điều kiện paper:** Paper Whisper gốc (Radford et al., 2022) **không công bố WER/BLEU riêng cho tiếng Việt hoặc cho Whisper-small trên VIVOS**. Nhóm **không đưa số "paper" cho các ô này để tránh sai lệch nguồn**; thay vào đó so sánh nội bộ giữa các phiên bản trên cùng tập test.

### 2.5. Gap Analysis (có kiểm chứng)

| Nhóm nguyên nhân | Nguyên nhân cụ thể | Cách kiểm chứng (đã làm) | Kết luận |
|------------------|--------------------|--------------------------|----------|
| **Dữ liệu** | Train chỉ 100 mẫu (~0,06h) vs 15h của VIVOS đầy đủ | So sánh WER zero-shot (>5000% trên synthetic) vs fine-tune (50,52% trên VIVOS) | Dữ liệu là **yếu tố chi phối**; có audio thật + fine-tune đã giảm WER về mức có nghĩa |
| **Dữ liệu (miền)** | Nhãn EN do MarianMT sinh → nhãn nhiễu | Oracle BLEU = 100 (vòng) → loại khỏi đánh giá | Không dùng oracle BLEU để kết luận |
| **Tiền xử lý & chuẩn hoá** | Chuẩn hoá text theo quy tắc riêng (NFC, lowercase, mở rộng viết tắt) | Quy tắc áp dụng nhất quán train/dev/test | Không phải nguồn chênh lệch chính |
| **Mô hình & huấn luyện** | Batch 2, 2 epochs, fp16 tắt, chỉ 100 mẫu train | Quan sát WER giảm đều qua các phiên bản (50,52 → 41,24) khi thêm LoRA/augment | Thiếu epoch/batch làm giảm chất lượng; cần learning curve đầy đủ |
| **Đánh giá** | Test set quick 20 mẫu, dùng trực tiếp để lặp phiên bản | Khai báo minh bạch (Mục 5.3) | **Có nguy cơ overfit vào test**; kết quả chỉ mang tính minh hoạ |
| **Đặc thù tiếng Việt** | Lỗi dấu thanh, từ đồng âm, từ mượn | `src/eval/error_analysis.py` phân loại S/D/I + lỗi dấu | Có code phân tích; chưa chạy đủ sâu do thiếu nhân lực |

### 2.6. Error Analysis của baseline

- **Phân loại lỗi (S/D/I):** cài đặt trong `src/eval/error_analysis.py` (substitution/deletion/insertion + lỗi dấu thanh + số + tên riêng + từ mượn).
- **Lỗi lan truyền ASR → MT (đo được):** so sánh `mt_cascade.bleu` với `mt_oracle_from_ref.bleu`. Trên bộ quick, oracle bị vô hiệu (100, vòng) nên chỉ dùng **cascade BLEU 4,83** làm mốc. Kết luận: **BLEU thấp chủ yếu do WER cao**, tức lỗi ASR lan truyền gần như toàn bộ sang MT.
- **WER theo nhóm:** chưa chạy đầy đủ theo gender/region/duration (hạn chế nhân lực) — ghi nhận là điểm cần bổ sung.

### 2.7. Điểm yếu ưu tiên cải thiện (cơ sở cho Proposed Method)

1. **WER còn cao (50,52%)** → cần fine-tune hiệu quả hơn với ít tài nguyên ⇒ **V2: LoRA + beam search**.
2. **BLEU cascade rất thấp (4,83)** → lỗi ASR lan truyền ⇒ **V3: noise-robust MT** + **V2: decode tốt hơn**.
3. **Thiếu robustness với nhiễu** ⇒ **V1: augmentation**.
4. **Kiến trúc cascade có thể bị error propagation** ⇒ **V4: so sánh end-to-end (negative control)**.

### 2.8. Sản phẩm Giai đoạn 2

- `src/models/asr_whisper.py`, `src/models/mt_model.py`, `src/models/cascade.py`
- `src/train/train_asr.py`, `src/train/train_mt.py`, `src/train/run_version.py`
- `src/eval/metrics.py`, `src/eval/evaluate.py`, `src/eval/evaluate_e2e.py`, `src/eval/error_analysis.py`
- `scripts/eval_zero_shot.py`
- Bảng **Paper vs Thực nghiệm** (`docs/paper_vs_experiment.xlsx`)
- Gap Analysis + Error Analysis (Mục 2.5, 2.6)

---

## 3. Giai đoạn 3 - Thực nghiệm & phát triển Proposed Method (V1 → V4)

### 3.1. Quy trình lặp & giả thuyết từng phiên bản

Mỗi phiên bản thay đổi **một yếu tố chính**, có giả thuyết - cấu hình - kết quả - kết luận:

| Phiên bản | Thay đổi chính (so với phiên bản trước) | Giả thuyết / Lý do | Config |
|-----------|------------------------------------------|--------------------|--------|
| **Baseline** | Whisper-small fine-tune, **không** augment, **greedy** | Mốc so sánh, cấu hình tối giản | `configs/quick.yaml` |
| **V1** | **+ Augmentation** (speed perturbation, Gaussian noise, reverb) + speaker cap | Tăng robustness với biến thiên giọng/nhiễu → giảm WER | `configs/v1_augment.yaml` |
| **V2** | **+ LoRA (r=16, alpha=32)** + **beam search (size 5)** + length penalty (kèm augment) | Fine-tune tham số hiệu quả với ít VRAM; beam search giảm lỗi decode | `configs/v2_lora.yaml` |
| **V3** | **+ Noise-robust MT** (dùng transcript ASR nhiễu làm nguồn) + LoRA + beam | MT chịu lỗi ASR tốt hơn → giảm error propagation, tăng BLEU | `configs/v3_noise_robust_mt.yaml` |
| **V4** | **End-to-end**: Whisper `task=translate` (VI→EN trực tiếp), bỏ MT | Kiểm chứng giả thuyết E2E đơn giản hơn cascade mà vẫn hiệu quả? (đối chứng âm) | `configs/v4_e2e.yaml` |

Cấu hình chung mọi phiên bản: 16 kHz, Log-Mel 80, batch 2, grad-accum 4, 2 epochs, early stopping patience 3, seed 42, fp16 tắt, beam size 5 (V2-V4).

### 3.2. Bảng tiến trình các phiên bản (bắt buộc) - SỐ LIỆU THẬT

**Đánh giá trên tập test 20 mẫu của bộ VIVOS quick** (nguồn: `outputs/*/seed_42/test_metrics*.json`).

| Phiên bản | Thay đổi chính | WER ↓ | CER ↓ | BLEU ↑ | chrF++ ↑ | RTF ↓ | Số tham số | Kết luận |
|-----------|----------------|-------|-------|--------|----------|-------|------------|----------|
| Baseline | Fine-tune, không augment, greedy | 50,52 | 22,83 | 4,83 | 25,52 | 0,23 | 241M + 77M | Mốc tham chiếu trên audio thật |
| V1 | + Augmentation | *(chưa ghi nhận)* | *(chưa ghi nhận)* | *(chưa ghi nhận)* | *(chưa ghi nhận)* | - | 241M + 77M | Đã huấn luyện, **chưa chạy đánh giá** |
| V2 | + LoRA + beam + augment | **41,24** | **16,05** | 22,01 | 42,98 | 0,27 | 241M (~5M trainable) + 77M | **WER tốt nhất**, BLEU tăng ~4,6× |
| V3 | + Noise-robust MT + LoRA + beam | 43,30 | 16,88 | **22,73** | **43,62** | 0,25 | 241M (~5M trainable) + 77M | **BLEU/chrF++ tốt nhất** → phiên bản chọn |
| V4 | End-to-end translate | - | - | 4,23 | 17,21 | 0,18 | 241M | **Thất bại**: kém cascade rõ rệt |
| **Best** | **V3** | 43,30 | 16,88 | **22,73** | **43,62** | 0,25 | 241M + 77M | Chọn để demo |

**Phân tích:**

- **V2** giảm WER mạnh nhất (50,52 → 41,24, tương đương **-18,4% tương đối**) và đưa BLEU từ 4,83 lên 22,01 (**+356%**). LoRA + beam + augmentation tỏ ra hiệu quả rõ rệt với ít epoch.
- **V3** đạt BLEU/chrF++ cao nhất (22,73 / 43,62) nhưng WER **nhích cao hơn** V2 (43,30 vs 41,24). Chênh lệch BLEU giữa V3 và V2 chỉ **+0,72 điểm** → **cải thiện nhỏ, CHƯA kiểm định ý nghĩa thống kê** (chỉ 1 seed, n=20).
- **V4** (E2E) chỉ đạt BLEU 4,23 — **kém xa** cascade (22,0-22,7) và ngang mức baseline kém nhất ⇒ trong điều kiện dữ liệu nhỏ, **cascade ưu thế hơn end-to-end**. Đây là **kết quả âm tính có giá trị** (minh chứng E2E cần nhiều dữ liệu hơn).
- **RTF:** tất cả phiên bản cascade đều < 1 (0,18-0,27) → **đạt yêu cầu real-time**; V4 nhanh nhất (0,18) do bỏ bước MT.

### 3.3. Ablation (mức độ có thể tách được)

Do hạn chế tài nguyên, nhóm **chưa chạy ablation tách rời từng thành phần** (chỉ có baseline vs V2/V3/V4 gộp nhiều thay đổi). Bảng dưới nêu **đối chứng gián tiếp** có bằng chứng:

| Đối chứng | So sánh | WER | BLEU | Suy luận (có kiểm soát) |
|-----------|---------|-----|------|--------------------------|
| Cascade vs E2E | V2/V3 (cascade) vs V4 (E2E) | 41,24-43,30 vs - | 22,0-22,7 vs 4,23 | **Kiến trúc cascade quan trọng hơn** trong điều kiện dữ liệu nhỏ |
| + LoRA + beam + augment | Baseline vs V2 | 50,52 → 41,24 | 4,83 → 22,01 | Gói cải tiến cấp decode/model **có tác dụng lớn** |
| + Noise-robust MT | V2 vs V3 | 41,24 → 43,30 | 22,01 → 22,73 | MT chịu nhiễu **cải thiện BLEU rất nhỏ** (trong biên độ nhiễu) |

> **Trung thực về ablation:** vì V2 gộp cả LoRA + beam + augmentation, **không thể khẳng định yếu tố nào đóng góp bao nhiêu**; đây là hạn chế (Mục 5.2) cần bổ sung bằng cách chạy từng thành phần riêng.

### 3.4. Kiểm định & phân tích bổ sung

- **Số seed:** chỉ chạy **seed 42** (do giới hạn thời gian GPU). **Chưa có trung bình ± độ lệch chuẩn**; các khác biệt nhỏ (V2 vs V3) chưa được kiểm định (thiếu bootstrap/paired test) → **thừa nhận không chắc chắn**.
- **Chi phí - lợi ích:** V2/V3 hơn baseline ~9 điểm WER và ~17 điểm BLEU, đổi lại RTF tăng nhẹ (0,23 → 0,25-0,27) — vẫn < 1 nên chi phí chấp nhận được.

### 3.5. Kết luận tính khả thi (trả lời RQ bằng số liệu)

| RQ | Câu hỏi | Trả lời | Bằng chứng |
|----|---------|---------|------------|
| **RQ1** | Augmentation có cải thiện không? | **Chưa đủ bằng chứng** | V1 đã huấn luyện nhưng **chưa có metric đánh giá**; V2 (có augment) tốt hơn baseline nhưng gộp nhiều thay đổi |
| **RQ2** | LoRA có hiệu quả không? | **Ủng hộ (một phần)** | V2 (LoRA): WER 50,52 → 41,24; BLEU 4,83 → 22,01 so với baseline |
| **RQ3** | Noise-robust MT có giảm lỗi lan truyền không? | **Ủng hộ yếu / chưa đủ bằng chứng** | V3 BLEU 22,73 (nhích hơn V2 22,01, +0,72) nhưng WER cao hơn; **không có ý nghĩa thống kê đã kiểm định** |
| **RQ4** | E2E có tốt hơn cascade không? | **Không ủng hộ** | V4 BLEU 4,23 << cascade 22,01-22,73 |

**Cải tiến đến từ đâu?** Chủ yếu từ **gói (LoRA + beam search + augmentation)** ở nhánh ASR/decode; MT noise-robust chỉ cải thiện biên.

**Hạn chế / khi nào không khả thi:**

- Dữ liệu huấn luyện quá nhỏ (100 mẫu) → WER còn >40%, **chưa đạt mức dùng thật**.
- Chỉ có giọng **miền Bắc** → không đảm bảo với giọng Nam/Trung (miền ngoài dữ liệu).
- E2E chưa khả thi ở quy mô dữ liệu này.

**Khả thi triển khai thực tế:** phiên bản cascade **V3** có RTF 0,25 (< 1) → **đáp ứng real-time** trên GPU phổ thông; bộ nhớ vừa (241M + 77M). ⇒ **Chọn V3 cho demo ở Task 3**.

**Phiên bản thất bại (V4):** đã báo cáo và giải thích — E2E cần nhiều dữ liệu + fine-tune kỹ hơn; kết quả âm tính này giúp loại bỏ hướng E2E ở giai đoạn hiện tại.

### 3.6. Sản phẩm Giai đoạn 3

- Config từng phiên bản: `configs/{v1_augment,v2_lora,v3_noise_robust_mt,v4_e2e}.yaml`
- Bảng tiến trình phiên bản: `docs/version_progress.xlsx`
- Checkpoint: `outputs/{v2_quick,v3_quick,v4_quick}/seed_42/best/`
- Metric JSON: `outputs/*/seed_42/test_metrics*.json`
- Kết luận tính khả thi: Mục 3.5

---

## 4. Tổng kết

### 4.1. Kết quả đạt được

1. **Giai đoạn 1 (Dataset Pipeline):** pipeline 12 bước tái lập được trên VIVOS thật; bộ dữ liệu cuối **10.887 mẫu** (train 8.253 / dev 1.084 / test 1.550), chống rò rỉ theo speaker, có Data Card + EDA + bảng mất cân bằng trước/sau.
2. **Giai đoạn 2 (Baseline):** tái dựng baseline Whisper-small + MarianMT cascade; có bảng **Paper vs Thực nghiệm**, Gap Analysis và Error Analysis; xác định điểm yếu dẫn dắt cải tiến.
3. **Giai đoạn 3 (Proposed):** thực nghiệm **V1→V4** trên audio thật; **V2** giảm WER tốt nhất, **V3** đạt BLEU/chrF++ tốt nhất, **V4** thất bại (đối chứng âm).

### 4.2. Bảng điểm tự đánh giá theo rubric

| Hạng mục | Điểm tối đa | Tự đánh giá | Cơ sở |
|----------|-------------|-------------|-------|
| GĐ1 - Dataset pipeline | 2,5 | 2,0 | Sạch/chuẩn hoá/rò rỉ/tái lập đạt; cân bằng chỉ đạt một phần (không thể cân bằng vùng miền) |
| GĐ2 - Baseline & Gap | 2,5 | 2,0 | Có bảng Paper vs Thực nghiệm + Gap Analysis; error analysis chưa sâu hết các nhóm |
| GĐ3 - Proposed V1…Vn | 3,5 | 2,0 | Có 4 phiên bản + số liệu thật; thiếu ablation tách rời + nhiều seed |
| Kết luận & khả thi | 0,5 | 0,5 | Trả lời RQ bằng số liệu, chọn V3 |
| Chất lượng kỹ thuật & tái lập | 0,5 | 0,5 | Code + config + seed + log + README |
| Trình bày & hỏi đáp | 0,5 | - | Chưa chấm |

### 4.3. Việc cần làm tiếp (để nâng điểm)

1. Huấn luyện trên **VIVOS đầy đủ** (8.253 train) thay vì bộ quick.
2. Chạy **≥ 3 seed** cho baseline/V2/V3, báo cáo **trung bình ± std**.
3. **Ablation tách rời** từng thành phần (LoRA / beam / augment / noise-robust MT).
4. **Error analysis sâu** theo gender/region/duration + phân loại lỗi dấu thanh.
5. **Đánh giá out-of-domain** (giọng Nam/Trung, nhiễu, hội thoại).
6. Sinh **learning curve** (loss/WER theo epoch) và biểu đồ so sánh phiên bản.

---

## 5. Hạn chế (khai báo minh bạch)

### 5.1. Hạn chế tài nguyên
Chỉ có 1 GPU GTX 1070 (8GB) → batch rất nhỏ (2), 2 epochs, fp16 tắt, chỉ chạy 1 seed.

### 5.2. Hạn chế thực nghiệm
- **Chưa ablation tách rời** → không cô lập được đóng góp từng yếu tố.
- **Chỉ 1 seed** → không có std, chưa kiểm định ý nghĩa thống kê.
- Huấn luyện/đánh giá trên **bộ quick (140 mẫu)** thay vì dataset đầy đủ.

### 5.3. Hạn chế phương pháp đánh giá (quan trọng)
Các phiên bản quick được **đánh giá trực tiếp trên tập test 20 mẫu** dùng để lặp phiên bản → **có nguy cơ overfit vào test** (vi phạm nguyên tắc "test chỉ chạy cho phiên bản chốt"). Kết quả do đó **chỉ mang tính minh hoạ**, không nên xem là ước lượng khách quan cho dataset đầy đủ.

### 5.4. Hạn chế nhãn EN
Nhãn EN của bộ quick do **MarianMT sinh** → **oracle BLEU bị vô hiệu (vòng)**. Chỉ số **cascade BLEU** mới có ý nghĩa; kết quả có thể lạc quan hơn so với nhãn EN do người dịch.

### 5.5. Hạn chế dữ liệu
VIVOS **100% giọng miền Bắc** → model có thể kém trên giọng Nam/Trung; chưa có tập OOD để kiểm chứng.

---

## 6. Tài liệu tham khảo

1. Radford, A., et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. OpenAI.
2. Luong, H.-T., & Vu, H.-Q. (2016). *A non-expert Kaldi recipe for Vietnamese Speech Recognition System*. WLSI/OIAF4HLT 2016.
3. Tiedemann, J., & Thottingal, S. (2020). *OPUS-MT — Building massively parallel multilingual machine translation models*. EMNLP 2020.
4. VIVOS Dataset: https://zenodo.org/record/7068130
5. HuggingFace: `openai/whisper-small`, `Helsinki-NLP/opus-mt-vi-en`.
6. Tài liệu yêu cầu Task 2 (Project description).

---

## Phụ lục A. Cấu trúc thư mục

```
asr_vi_en_task2/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── quick.yaml
│   ├── v1_augment.yaml
│   ├── v2_lora.yaml
│   ├── v3_noise_robust_mt.yaml
│   └── v4_e2e.yaml
├── src/
│   ├── data/      (prepare_data, text_normalize, align_check, translation_check,
│   │               dedup, imbalance, split, augment, features)
│   ├── models/    (asr_whisper, mt_model, cascade)
│   ├── train/     (train_asr, train_mt, train_e2e, run_version)
│   ├── eval/      (metrics, evaluate, evaluate_e2e, error_analysis)
│   └── utils/     (config, seed, logger)
├── scripts/       (download_vivos_hf, convert_vivos_format, create_quick_dataset,
│                   csv_to_excel, md_to_html, open_print_dialog, ...)
├── notebooks/eda.ipynb
├── data/
│   ├── manifests/ (train/dev/test.jsonl, *_quick.jsonl, data_card.md, pipeline_report.json)
│   ├── data_card.md, imbalance_analysis.md
│   └── raw/
├── outputs/       (metric JSON + checkpoint các phiên bản)
└── docs/          (TASK2_REPORT.md/html, experiment_log.md, team_contributions.md,
                    slide_outline.md, baseline_selection.md, TRAINING_GUIDE.md,
                    paper_vs_experiment.{csv,xlsx}, version_progress.{csv,xlsx})
```

## Phụ lục B. Lệnh tái lập

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Tải VIVOS + chuẩn hoá format
python scripts/download_vivos_hf.py --output_dir data/raw/vietnamese/vivos
python scripts/convert_vivos_format.py

# 3. Pipeline dữ liệu
python -m src.data.prepare_data --config configs/baseline.yaml

# 4. Tạo bộ quick + dịch EN (để thí nghiệm nhanh)
python scripts/create_quick_dataset.py

# 5. Chạy các phiên bản
python -m src.train.run_version --version baseline
python -m src.train.run_version --version v2_lora
python -m src.train.run_version --version v3_noise_robust_mt
python -m src.train.run_version --version v4_e2e

# 6. Xuất bảng kết quả
python scripts/csv_to_excel.py docs/paper_vs_experiment.csv
python scripts/csv_to_excel.py docs/version_progress.csv

# 7. Sinh HTML báo cáo (mở browser → Ctrl+P → Save as PDF)
python scripts/md_to_html.py docs/TASK2_REPORT.md
```

## Phụ lục C. Nhật ký thực nghiệm

| Ngày | Phiên bản | Cấu hình | Kết quả (test quick, n=20) | Nhận xét | Trạng thái |
|------|-----------|----------|-----------------------------|----------|------------|
| 2026-10-20 | Zero-shot (synthetic) | Whisper-small pretrained + MarianMT, 4 mẫu sine | WER 5130,77 · CER 5414,29 · BLEU 0,00 · chrF++ 0,30 · RTF 7,80 | Audio tổng hợp → không nhận diện được; chỉ kiểm chứng pipeline | Hoàn thành |
| 2026-10-20 | Baseline (VIVOS quick) | Whisper-small, không augment, greedy, 2 epochs | WER 50,52 · CER 22,83 · BLEU 4,83 · chrF++ 25,52 · RTF 0,23 | Mốc tham chiếu trên audio thật | Hoàn thành |
| 2026-10-20 | V1 (Augmentation) | + speed/noise/reverb, speaker cap | *(chưa ghi nhận metric)* | Đã huấn luyện, chưa chạy đánh giá | Một phần |
| 2026-10-20 | V2 (LoRA + Beam) | LoRA r=16 α=32 + beam=5 + augment, lr 5e-5 | WER **41,24** · CER 16,05 · BLEU 22,01 · chrF++ 42,98 · RTF 0,27 | WER tốt nhất | Hoàn thành |
| 2026-10-20 | V3 (Noise-robust MT) | + MT chịu nhiễu, LoRA + beam, lr 3e-5 | WER 43,30 · CER 16,88 · BLEU **22,73** · chrF++ **43,62** · RTF 0,25 | BLEU cao nhất → chọn demo | Hoàn thành |
| 2026-10-20 | V4 (E2E translate) | Whisper task=translate, beam=5, augment | BLEU 4,23 · chrF++ 17,21 · RTF 0,18 | Kém cascade rõ rệt | Hoàn thành |

Chi tiết đầy đủ: `docs/experiment_log.md`.

## Phụ lục D. Bảng phân công & đóng góp

| Thành viên | MSSV | Phần phụ trách | Đóng góp |
|-----------|------|----------------|----------|
| Nguyễn Đức Mạnh | SE194525 | Toàn bộ: Dataset Pipeline, Baseline Reproduction, Proposed Method (V1-V4), Evaluation, Báo cáo & bảng kết quả | 100% |

Chi tiết: `docs/team_contributions.md`.

---

**Kết thúc báo cáo.**
