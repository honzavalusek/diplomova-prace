# Usage

Two scripts live in `scripts/`:

- `01_search.py` — search a single query against a single corpus file and print top matches.
- `02_evaluate.py` — walk `data/raw_audio/` and run every query against its corpus, writing results to a CSV.

## Single-query search (`01_search.py`)

```bash
python scripts/01_search.py \
    --query  data/raw_audio/<lang>/<source>/<topic>/queries/<query>.wav \
    --corpus data/raw_audio/<lang>/<source>/<topic>/corpus/<corpus>.wav \
    --model wavlm-base \
    --window 25
```

### Options

| Option | Values | Description |
|--------|--------|-------------|
| `--model` | `xlsr-53`, `xls-r-300m`, `xls-r-1b`, `xls-r-2b`, `czech`, `czech2`, `wavlm-base`, `wavlm-base-plus`, `wavlm-large` | Feature extraction model |
| `--window` | integer (e.g., 25) | Sakoe-Chiba DTW constraint (limits temporal deviation) |
| `--layer-min` | integer (e.g., 0) | Minimum layer index (0-based) for averaging |
| `--layer-max` | integer (e.g., 11) | Maximum layer index (0-based) for averaging |
| `--device` | `cpu`, `cuda` | Compute device |
| `--top-k` | integer (default: 3) | Number of top matches per corpus file |

### Output

```
Best match found in: <corpus>.wav
  Distance: 0.1234          # Lower = better match
  Time range: 5.23s - 6.45s # Where query was found
  Duration: 1.22s           # Length of match
```

## Batch evaluation (`02_evaluate.py`)

```bash
python scripts/02_evaluate.py
```

Walks `data/raw_audio/` automatically, runs every query against its corpus, and writes a CSV to `results/`. Configuration (model, window, device, top-k) is hardcoded at the top of the script.

### CSV columns

`results/top5_wavlm-base_cpu_window25_<timestamp>.csv`:

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

## Troubleshooting

### "Model not found" error

First run downloads models from Hugging Face (~1–3 GB):

```bash
python -c "from transformers import Wav2Vec2Model; \
           Wav2Vec2Model.from_pretrained('microsoft/wavlm-base')"
```

### Out of memory

- Process shorter audio files (< 1 minute).
- Reduce corpus size.
- Use GPU: `--device cuda`.

### Slow performance

- First run downloads the model (~2 minutes).
- Expected DTW time: ~1–5 seconds per corpus file.
- GPU acceleration: `--device cuda` (10–20× faster).
