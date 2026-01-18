# Scalable Acoustic Retrieval System

Query-by-Example Spoken Term Detection (QbE-STD) using self-supervised speech embeddings and Subsequence DTW matching.

**Status:** Phase 1 (Foundation) Complete

---

## Quick Start

### 1. Install Dependencies

Recommended: use conda for environment management.

```bash
pip install -r requirements.txt

# dtaidistance may require conda-forge for C-optimized version:
conda install -c conda-forge dtaidistance

# Verify installation
python -c "import torch; import transformers; print('Dependencies OK')"
```

### 2. Run Demo

```bash
python scripts/01_minimal_demo.py \
    --query data/raw_audio/queries/kanadskej_zertik_2.wav \
    --corpus data/raw_audio/corpus/kancl_1.wav \
    --model wavlm-base \
    --normalize mvn \
    --window 25 \
    --layers 12
```

### 3. Understanding the Output

```
Best match found in: kancl_1.wav
  Distance: 0.1234          # Lower = better match
  Time range: 5.23s - 6.45s # Where query was found
  Duration: 1.22s           # Length of match
```

---

## Available Options

| Option | Values | Description |
|--------|--------|-------------|
| `--model` | `xlsr-53`, `xls-r-300m`, `xls-r-1b`, `xls-r-2b`, `czech`, `czech2`, `wavlm-base`, `wavlm-base-plus`, `wavlm-large` | Feature extraction model |
| `--normalize` | `none`, `mvn`, `cmn` | Speaker normalization (mvn=mean-variance, cmn=cepstral-mean) |
| `--window` | integer (e.g., 25) | Sakoe-Chiba DTW constraint (limits temporal deviation) |
| `--layers` | integer (e.g., 12) | Number of last transformer layers for embeddings |
| `--device` | `cpu`, `cuda` | Compute device |
| `--top-k` | integer (default: 3) | Number of top matches per corpus file |

---

## Project Status

### What's Built

| Module | Status | Description |
|--------|--------|-------------|
| Feature Extraction | Complete | XLSR-53, XLS-R, WavLM embedding extraction |
| Speaker Normalization | Complete | MVN and CMN normalization |
| DTW Matching | Complete | Subsequence DTW with Sakoe-Chiba constraint |
| End-to-End Demo | Complete | Working QbE-STD pipeline |

### Current Capabilities

- Extract high-quality acoustic features from any audio file
- Search for spoken queries in corpus recordings
- Get precise timestamps (start/end) of matches
- Multiple model options (XLSR, WavLM families)
- Speaker normalization for cross-speaker matching

### Limitations

- **Corpus length:** Currently limited to a few minutes per file before running out of memory
- No LSH indexing yet (linear search only, fine for <100 files)
- No embedding cache (re-extracts features every run)
- No batching (processes files sequentially)
- No corpus segmentation (long files consume high memory)

---

## Architecture

```
Audio → Embedding Model → [Optional: Normalize] → [Verify: S-DTW] → Timestamps
              ↓
        WavLM/XLSR-53
         (768-1024D)
```

**Pipeline Flow:**
```
1. Query Audio (WAV)
   ↓
2. Feature Extraction (WavLM/XLSR) → [768-1024D embeddings per frame]
   ↓
3. Speaker Normalization (optional) → [MVN/CMN normalized embeddings]
   ↓
4. Subsequence DTW Matching → Find best alignment
   ↓
5. Frame-to-Time Conversion → Precise timestamps
```

---

## Technical Details

### Supported Models

| Model | Embedding Dim | Notes |
|-------|---------------|-------|
| `xlsr-53` | 1024 | Facebook's cross-lingual model |
| `xls-r-300m/1b/2b` | 1024 | Larger XLS-R variants |
| `wavlm-base` | 768 | Microsoft WavLM |
| `wavlm-base-plus` | 768 | WavLM with more data |
| `wavlm-large` | 1024 | Large WavLM model |
| `czech`, `czech2` | varies | Czech-specific fine-tuned models |

### Audio Processing

- **Sample Rate:** 16,000 Hz (enforced)
- **Hop Length:** 320 samples (~20ms per frame)

### Frame-to-Time Conversion

```python
time_sec = (frame_idx * 320) / 16000

# Example: frame 100
# 100 * 320 / 16000 = 2.0 seconds
```

### DTW Configuration

- **Library:** dtaidistance (C-optimized)
- **Distance Metric:** Cosine (recommended for neural embeddings)
- **Window Constraint:** Sakoe-Chiba band (e.g., 25 frames)

---

## Project Structure

```
acoustic-retrieval/
├── README.md              ← This file
├── requirements.txt       ← Dependencies
├── config/
│   └── model_config.yaml  ← Model settings
├── src/
│   ├── features/          ← Feature extraction
│   │   ├── wav2vec2_wavlm_extractor.py
│   │   ├── audio_preprocessing.py
│   │   ├── speaker_normalization.py
│   │   └── frame_conversion.py
│   ├── matching/          ← DTW matching
│   │   ├── subsequence_dtw.py
│   │   └── distance_metrics.py
│   ├── indexing/          ← LSH (TODO)
│   ├── pipeline/          ← End-to-end system (TODO)
│   └── corpus/            ← Corpus management (TODO)
├── scripts/
│   └── 01_minimal_demo.py ← Working demo
└── data/
    ├── raw_audio/
    │   ├── queries/       ← Query audio files
    │   └── corpus/        ← Corpus audio files
    └── embeddings/        ← Cached features (future)
```

---

## Development Roadmap

**Phase 1: Foundation** (Complete)
- [x] Feature extraction (XLSR-53, WavLM)
- [x] Speaker normalization (MVN, CMN)
- [x] DTW matching with constraints
- [x] Minimal end-to-end demo

**Phase 2: Indexing** (Planned)
- [ ] MinHash LSH prototype
- [ ] Learned hash functions
- [ ] FAISS integration

**Phase 3: Production** (Planned)
- [ ] Corpus segmentation
- [ ] Embedding caching
- [ ] REST API
- [ ] Docker deployment

---

## Troubleshooting

### "Model not found" error

First run downloads models from Hugging Face (~1-3GB):

```bash
# Pre-download a model
python -c "from transformers import Wav2Vec2Model; \
           Wav2Vec2Model.from_pretrained('microsoft/wavlm-base')"
```

### Out of Memory

- Process shorter audio files (< 1 minute)
- Reduce corpus size
- Use GPU: `--device cuda`

### Slow Performance

- **Normal:** First run downloads model (~2 minutes)
- **Expected DTW time:** ~1-5 seconds per corpus file
- **GPU acceleration:** `--device cuda` (10-20x faster)

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers (Hugging Face)
- librosa
- dtaidistance

GPU optional but recommended for large-scale deployment.

---

## Contributing

To extend the system:

1. **Add new feature extractor:** Implement in `src/features/`
2. **Add new LSH method:** Implement in `src/indexing/`
3. **Add new distance metric:** Implement in `src/matching/distance_metrics.py`
