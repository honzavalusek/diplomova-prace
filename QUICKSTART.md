# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python -c "import torch; import transformers; print('✓ Dependencies OK')"
```

## Test with Sample Audio (10 minutes)

Place your audio files:
```
data/raw_audio/
├── queries/
│   └── your_query.wav     # Short clip (1-3 seconds)
└── corpus/
    ├── recording1.wav      # Longer recordings
    └── recording2.wav
```

Then run:
```bash
python scripts/01_minimal_demo.py \
    --query data/raw_audio/queries/your_query.wav \
    --corpus data/raw_audio/corpus/*.wav
```

## Understanding the Output

```
Best match found in: recording1.wav
  Distance: 0.1234          # Lower = better match
  Time range: 5.23s - 6.45s # Where query was found
  Duration: 1.22s           # Length of match
```

## What Happens Under the Hood

```
1. Query Audio (WAV)
   ↓
2. XLSR-53 Feature Extraction → [1024-D embeddings per frame]
   ↓
3. Subsequence DTW Matching → Find best alignment
   ↓
4. Frame-to-Time Conversion → 5.23s - 6.45s
```

## Troubleshooting

### "Model not found" error
First run downloads ~1.2GB XLSR model from Hugging Face:
```bash
# Pre-download the model
python -c "from transformers import Wav2Vec2Model; \
           Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')"
```

### Out of Memory (CPU)
- Process shorter audio files (< 1 minute)
- Reduce corpus size
- Use GPU if available: `--device cuda`

### Slow Performance
- **Normal:** First run downloads model (~2 minutes)
- **Expected DTW time:** ~1-5 seconds per corpus file
- **GPU acceleration:** Add `--device cuda` (10-20x faster)

## Next Steps

Once the minimal demo works:

1. **Add LSH Indexing** (Module 2) - Scale to 100+ hours
2. **Batch Processing** - Pre-compute all embeddings
3. **Caching** - Store embeddings on disk
4. **Production API** - REST endpoint for queries

See `README.md` for full architecture details.
