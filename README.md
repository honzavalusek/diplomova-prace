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
│   └── batch_evaluate.py    ← batch evaluation, outputs CSV
└── docs/
    ├── usage.md
    └── architecture.md
```
