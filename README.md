# Scalable Acoustic Retrieval System

Query-by-Example Spoken Term Detection (QbE-STD) using self-supervised speech embeddings and Subsequence DTW matching.

**Status:** Complete

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

### 2. Run Batch Evaluation

```bash
python scripts/02_evaluate.py
```

Walks `data/raw_audio/` automatically, runs every query against its corpus, and writes a CSV to `results/`. Configuration (model, window, device, top-k) is hardcoded at the top of the script.

### 3. Run Single-File Demo

```bash
python scripts/01_minimal_demo.py \
    --query  data/raw_audio/<lang>/<source>/<topic>/queries/<query>.wav \
    --corpus data/raw_audio/<lang>/<source>/<topic>/corpus/<corpus>.wav \
    --model wavlm-base \
    --window 25
```

### 4. Understanding the Output

**Batch evaluation CSV** (`results/top5_wavlm-base_cpu_window25_<timestamp>.csv`):

| Column | Description |
|--------|-------------|
| `language` | `EN` or `CZ` |
| `query_file` | Path to query wav (relative to project root) |
| `query_length` | Query duration in seconds |
| `corpus_file` | Path to corpus wav (relative to project root) |
| `corpus_length` | Corpus duration in seconds |
| `match_rank` | Rank 1–5 (1 = best match) |
| `match_distance` | DTW distance (lower = better) |
| `match_start` | Match start time in corpus (seconds) |
| `match_end` | Match end time in corpus (seconds) |

**Single-file demo:**
```
Best match found in: <corpus>.wav
  Distance: 0.1234          # Lower = better match
  Time range: 5.23s - 6.45s # Where query was found
  Duration: 1.22s           # Length of match
```

---

## Available Options

| Option | Values | Description |
|--------|--------|-------------|
| `--model` | `xlsr-53`, `xls-r-300m`, `xls-r-1b`, `xls-r-2b`, `czech`, `czech2`, `wavlm-base`, `wavlm-base-plus`, `wavlm-large` | Feature extraction model |
| `--window` | integer (e.g., 25) | Sakoe-Chiba DTW constraint (limits temporal deviation) |
| `--layer-min` | integer (e.g., 0) | Minimum layer index (0-based) for averaging |
| `--layer-max` | integer (e.g., 11) | Maximum layer index (0-based) for averaging |
| `--device` | `cpu`, `cuda` | Compute device |
| `--top-k` | integer (default: 3) | Number of top matches per corpus file |

---

## Project Status

### What's Built

| Module | Status | Description |
|--------|--------|-------------|
| Feature Extraction | Complete | XLSR-53, XLS-R, WavLM embedding extraction |
| DTW Matching | Complete | Subsequence DTW with Sakoe-Chiba constraint |
| End-to-End Demo | Complete | Working QbE-STD pipeline |
| Batch Evaluation | Complete | CSV output for all queries/corpora |

### Current Capabilities

- Extract high-quality acoustic features from any audio file
- Search for spoken queries in corpus recordings
- Get precise timestamps (start/end) of matches
- Multiple model options (XLSR, WavLM families)

### Limitations

- **Corpus length:** Currently limited to a few minutes per file before running out of memory
- No LSH indexing yet (linear search only, fine for <100 files)
- No embedding cache (re-extracts features every run)
- No batching (processes files sequentially)
- No corpus segmentation (long files consume high memory)

---

## Architecture

```
Audio → Embedding Model → [Verify: S-DTW] → Timestamps
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
3. Subsequence DTW Matching → Find best alignment
   ↓
4. Frame-to-Time Conversion → Precise timestamps
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
diplomova-prace/
├── README.md
├── requirements.txt
├── src/
│   ├── features/
│   │   ├── ssl_speech_extractor.py
│   │   ├── audio_preprocessing.py
│   │   └── frame_conversion.py
│   └── matching/
│       ├── subsequence_dtw.py
│       └── distance_metrics.py
├── scripts/
│   ├── 01_minimal_demo.py   ← single query/corpus pair, prints results
│   └── 02_evaluate.py       ← batch evaluation, outputs CSV
├── results/                 ← CSV outputs from 02_evaluate.py
└── data/
    └── raw_audio/           ← gitignored; see required structure below
        ├── en/
        │   └── <source>/
        │       └── <topic>/
        │           ├── corpus/    ← single .wav file
        │           └── queries/   ← one or more .wav files
        └── cz/
            └── <source>/
                └── <topic>/
                    ├── corpus/
                    └── queries/
```

Each topic directory contains:
- `corpus/` — a single `.wav` file (the recording to search in)
- `queries/` — one or more `.wav` files (spoken terms to search for)

> **This directory structure is mandatory.** `02_evaluate.py` discovers all query/corpus pairs by walking `raw_audio/` and looking for `queries/` directories. The language (`EN`/`CZ`) is derived from the first path component (`en`/`cz`). Deviating from this layout will cause topics to be silently skipped.

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
