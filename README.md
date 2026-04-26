# Scalable Acoustic Retrieval System

Query-by-Example Spoken Term Detection (QbE-STD) using self-supervised speech embeddings and Subsequence DTW matching.

## Documentation

- [Usage](docs/usage.md) — running the scripts, options, output format, troubleshooting
- [Architecture](docs/architecture.md) — pipeline overview and core modules

## Install

```bash
pip install -r requirements.txt

# Optional: C-optimized dtaidistance
conda install -c conda-forge dtaidistance
```

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
│   ├── search.py            ← single query/corpus pair, prints results
│   └── 02_evaluate.py       ← batch evaluation, outputs CSV
├── docs/
│   ├── usage.md
│   └── architecture.md
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
