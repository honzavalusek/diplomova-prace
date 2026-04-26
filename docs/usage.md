# Usage

Two scripts live in `scripts/`:

- `search.py` — search a single query against a single corpus file and print top matches.
- `batch_evaluate.py` — walk `data/raw_audio/` and run every query against its corpus, writing results to a CSV.

## Single-query search (`search.py`)

```bash
python scripts/search.py \
    --query  data/raw_audio/<lang>/<source>/<topic>/queries/<query>.wav \
    --corpus data/raw_audio/<lang>/<source>/<topic>/corpus/<corpus>.wav \
    --model wavlm-base \
    --window 25
```

### Options

| Option | Values | Description |
|--------|--------|-------------|
| `--model` | `xlsr-53`, `xls-r-300m`, `xls-r-1b`, `xls-r-2b`, `wavlm-base`, `wavlm-base-plus`, `wavlm-large` | Feature extraction model |
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

## Batch evaluation (`batch_evaluate.py`)

```bash
python scripts/batch_evaluate.py \
    --input-dir  data/raw_audio \
    --output-dir results
```

Walks the input directory recursively, runs every query against its corpus, and writes a CSV to the output directory. The CSV filename is auto-generated from the chosen model, device, window, and top-k.

### Options

| Option | Values | Description |
|--------|--------|-------------|
| `--input-dir` | path (required) | Directory to walk for query/corpus pairs |
| `--output-dir` | path (required) | Directory to write the CSV into; filename is auto-generated |
| `--model` | `xlsr-53`, `xls-r-300m`, `xls-r-1b`, `xls-r-2b`, `wavlm-base`, `wavlm-base-plus`, `wavlm-large` | Feature extraction model (default: `wavlm-base`) |
| `--top-k` | integer | Number of top matches per query (default: 5) |
| `--device` | `cpu`, `cuda` | Compute device (default: `cpu`) |
| `--window` | integer | Sakoe-Chiba DTW constraint (default: 25) |
| `--tolerance` | float | Seconds of slack on both span ends when comparing predictions to `metadata.csv` ground truth (default: 0.3). Only applied when `metadata.csv` is present. |

### Data layout

```
input-dir/
├── <lang1>/
│   └── <source>/
│       └── <topic>/
│           ├── corpus/    ← single .wav file
│           └── queries/   ← one or more .wav files
└── <lang2>/
    └── <source>/
        └── <topic>/
            ├── corpus/
            └── queries/
```

Each topic directory contains:
- `corpus/` — a single `.wav` file (the recording to search in)
- `queries/` — one or more `.wav` files (spoken terms to search for)

> **This directory structure is mandatory.** `batch_evaluate.py` discovers all query/corpus pairs by walking the input directory and looking for `queries/` directories. The language (`LANG1`/`LANG2`) is derived from the first path component (`lang1`/`lang2`). Deviating from this layout will cause topics to be silently skipped. There can be more languages. There can also be more sources per language, and more topics per source. But the `corpus/` + `queries/` structure must be preserved.

### Ground-truth metadata (optional)

If a file named `metadata.csv` is present at the root of `--input-dir`, `batch_evaluate.py` picks it up automatically and adds two judgement columns to the output CSV (`is_correct (same phrase)` and `is_same (same audio)`), so predictions can be auto-scored against known occurrences instead of hand-annotated. If the file is absent, the evaluation runs without those columns.

The file is a CSV with a header row and exactly these four columns:

| Column | Description |
|--------|-------------|
| `corpus_file` | Path to a corpus `.wav`, **relative to `--input-dir`** |
| `match_start` | Ground-truth occurrence start time in the corpus (seconds) |
| `match_end` | Ground-truth occurrence end time in the corpus (seconds) |
| `same_query` | Optional path to a query `.wav` (relative to `--input-dir`) whose audio is the exact corpus segment at `[match_start, match_end]`. Leave empty for ground-truth occurrences not extracted from any query. |

One row per known occurrence — multiple rows per corpus file are expected (one per occurrence). A prediction counts as a hit when both its start and end are within the configured tolerance (`--tolerance`, default **±0.3 s**) of a ground-truth row.

Example:

```csv
corpus_file,match_start,match_end,same_query
cz/kancl_ep_1/kanadskej_zertik/corpus/kancl_ep_1__27_29.wav,12.8,13.42,cz/kancl_ep_1/kanadskej_zertik/queries/kancl_ep_1_kanadskej_zertik_1_potichu.wav
cz/kancl_ep_1/kanadskej_zertik/corpus/kancl_ep_1__27_29.wav,101.8,102.46,cz/kancl_ep_1/kanadskej_zertik/queries/kancl_ep_1_kanadskej_zertik_2.wav
en/zuck/quest/corpus/zuck_quest.wav,17.26,17.54,
```

The third row has an empty `same_query` — it documents a known occurrence in the corpus that no query was cut from, so it can contribute to `is_correct` but never to `is_same`.

### CSV columns

`results/top5_wavlm-base_cpu_window25_<timestamp>.csv`:

| Column | Description |
|--------|-------------|
| `language` | `EN` or `CZ` |
| `query_file` | Path to query wav (relative to project root) |
| `query_length` | Query duration in seconds |
| `corpus_file` | Path to corpus wav (relative to project root) |
| `corpus_length` | Corpus duration in seconds |
| `match_rank` | Rank 1–`top-k` (1 = best match) |
| `match_distance` | DTW distance (lower = better) |
| `match_start` | Match start time in corpus (seconds) |
| `match_end` | Match end time in corpus (seconds) |

When `metadata.csv` is present, two extra columns are appended:

| Column | Description |
|--------|-------------|
| `is_correct (same phrase)` | `True` if the predicted span matches *any* ground-truth occurrence for this corpus within `--tolerance` seconds on both ends |
| `is_same (same audio)` | `True` only if the predicted span matches a ground-truth occurrence whose `same_query` equals this query file (i.e., the query was cut from that exact corpus segment), within `--tolerance` |

If `metadata.csv` is present but the row's `corpus_file` is not listed in it, both columns are left blank for that row.

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
