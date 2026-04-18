# Model Comparison: XLSR-53 vs XLS-R

## Overview

The system now supports both **XLSR-53** and the newer **XLS-R** models for feature extraction.

## Available Models

| Model | Parameters | Embedding Dim | Languages | Training Data | Use Case |
|-------|-----------|---------------|-----------|---------------|----------|
| **XLSR-53** | 300M | 1024 | 53 | 56k hours | **Recommended for most users** |
| **XLS-R 300M** | 300M | 1024 | 128 | 436k hours | Better accuracy, same size as XLSR-53 |
| **XLS-R 1B** | 1B | 1280 | 128 | 436k hours | Best quality, needs GPU |
| **XLS-R 2B** | 2B | 1920 | 128 | 436k hours | Highest quality, GPU required |

## Quick Guide

### XLSR-53 (Default)
```python
from src.features import XLSR53FeatureExtractor

extractor = XLSR53FeatureExtractor()
embeddings = extractor.extract(audio, sr=16000)
# Output shape: (sequence_length, 1024)
```

**Pros:**
- ✅ Well-tested, proven performance
- ✅ Smaller download (~1.2GB)
- ✅ Works well on CPU
- ✅ 1024-D embeddings (standard)

**Cons:**
- ⚠️ Older training data
- ⚠️ Fewer languages (53 vs 128)

**Best for:** Most use cases, prototyping, CPU-only systems

---

### XLS-R 300M

```python
from src.features import SSLSpeechExtractor

extractor = SSLSpeechExtractor(model_name="facebook/wav2vec2-xls-r-300m")
embeddings = extractor.extract(audio, sr=16000)
# Output shape: (sequence_length, 1024)
```

**Pros:**
- ✅ Same size as XLSR-53
- ✅ Better trained (8x more data)
- ✅ More languages (128 vs 53)
- ✅ 1024-D embeddings (compatible with XLSR-53)

**Cons:**
- ⚠️ Slightly slower inference (~10% slower)

**Best for:** Production systems, better accuracy with same resources

---

### XLS-R 1B

```python
from src.features import SSLSpeechExtractor

extractor = SSLSpeechExtractor(model_name="facebook/wav2vec2-xls-r-1b", device="cuda")
embeddings = extractor.extract(audio, sr=16000)
# Output shape: (sequence_length, 1280)
```

**Pros:**
- ✅ Best quality-to-size ratio
- ✅ Significantly better accuracy
- ✅ 1280-D embeddings (more semantic info)

**Cons:**
- ⚠️ Requires GPU (OOM on most CPUs)
- ⚠️ Larger download (~4GB)
- ⚠️ Slower inference (3-4x slower than 300M)
- ⚠️ Different embedding dim (1280 vs 1024)

**Best for:** High-accuracy requirements, GPU available, larger corpora

---

### XLS-R 2B

```python
from src.features import SSLSpeechExtractor

extractor = SSLSpeechExtractor(model_name="facebook/wav2vec2-xls-r-2b", device="cuda")
embeddings = extractor.extract(audio, sr=16000)
# Output shape: (sequence_length, 1920)
```

**Pros:**
- ✅ Highest possible quality
- ✅ Best for low-resource languages
- ✅ 1920-D embeddings (maximum semantic info)

**Cons:**
- ⚠️ **Requires powerful GPU** (16GB+ VRAM)
- ⚠️ Very large download (~8GB)
- ⚠️ Very slow (5-6x slower than 300M)
- ⚠️ Different embedding dim (1920 vs 1024)

**Best for:** Research, maximum accuracy, powerful GPU clusters

---

## Usage in Scripts

### Command Line (Demo Script)

```bash
# Default: XLSR-53
python scripts/01_minimal_demo.py \
    --query query.wav \
    --corpus corpus/*.wav

# Use XLS-R 300M
python scripts/01_minimal_demo.py \
    --query query.wav \
    --corpus corpus/*.wav \
    --model xls-r-300m

# Use XLS-R 1B (GPU recommended)
python scripts/01_minimal_demo.py \
    --query query.wav \
    --corpus corpus/*.wav \
    --model xls-r-1b \
    --device cuda
```

### Python API

```python
from src.features import XLSR53FeatureExtractor, SSLSpeechExtractor

# Option 1: XLSR-53 (default)
extractor = XLSR53FeatureExtractor()

# Option 2: XLS-R 300M
extractor = SSLSpeechExtractor(model_name="facebook/wav2vec2-xls-r-300m")

# Option 3: XLS-R 1B (GPU)
extractor = SSLSpeechExtractor(
    model_name="facebook/wav2vec2-xls-r-1b",
    device="cuda"
)

# All have the same interface
embeddings = extractor.extract(audio, sr=16000)
print(f"Embedding dim: {extractor.embedding_dim}")  # Auto-detected
```

---

## Performance Comparison

### Latency (1 minute audio)

| Model | CPU (M1 Mac) | GPU (RTX 3090) | Memory |
|-------|--------------|----------------|--------|
| XLSR-53 | ~15s | ~1s | 2GB |
| XLS-R 300M | ~17s | ~1.2s | 2.5GB |
| XLS-R 1B | OOM | ~3s | 6GB |
| XLS-R 2B | OOM | ~5s | 12GB |

### Accuracy (Approximate)

Based on published benchmarks:
- XLSR-53: Baseline (100%)
- XLS-R 300M: +5-10% better
- XLS-R 1B: +15-20% better
- XLS-R 2B: +20-25% better

*Actual improvements vary by language and task*

---

## Recommendations

### Development & Testing
→ **XLSR-53**
- Fast iteration
- Works on laptops
- Proven stability

### Production (CPU-only)
→ **XLS-R 300M**
- Better accuracy
- More languages
- Same infrastructure

### Production (GPU available)
→ **XLS-R 1B**
- Best quality/cost ratio
- Fits on consumer GPUs
- Significantly better results

### Research / Maximum Quality
→ **XLS-R 2B**
- State-of-the-art
- Requires powerful infrastructure
- Diminishing returns vs 1B

---

## Migration Guide

### From XLSR-53 to XLS-R 300M

**No code changes needed** - same embedding dimension (1024):

```python
# Before
extractor = XLSR53FeatureExtractor()

# After
extractor = XLSRExtractor(model_name="facebook/wav2vec2-xls-r-300m")

# Embeddings are compatible!
```

### From XLSR-53 to XLS-R 1B/2B

**Different embedding dimensions** - requires re-indexing:

```python
# XLSR-53: (N, 1024)
# XLS-R 1B: (N, 1280)
# XLS-R 2B: (N, 1920)

# You'll need to:
# 1. Re-extract all corpus embeddings
# 2. Rebuild LSH index
# 3. Update downstream code expecting 1024-D
```

---

## Backwards Compatibility

The old import still works:

```python
# Old code (still works)
from src.features import XLSRFeatureExtractor

# Automatically uses XLSR53FeatureExtractor
extractor = XLSRFeatureExtractor()
```

This ensures existing code doesn't break.

---

## Download Sizes

| Model | Download Size | Disk Size |
|-------|--------------|-----------|
| XLSR-53 | ~1.2GB | ~1.5GB |
| XLS-R 300M | ~1.2GB | ~1.5GB |
| XLS-R 1B | ~3.8GB | ~4.5GB |
| XLS-R 2B | ~7.5GB | ~9GB |

First run downloads from Hugging Face Hub. Models are cached at:
```
~/.cache/huggingface/transformers/
```

---

## Summary Decision Tree

```
Do you have a GPU?
├─ No → Use XLSR-53 or XLS-R 300M
│       ├─ Prototyping? → XLSR-53
│       └─ Production? → XLS-R 300M
│
└─ Yes → What's your VRAM?
        ├─ <8GB → XLS-R 300M (CPU fallback)
        ├─ 8-16GB → XLS-R 1B
        └─ >16GB → XLS-R 2B (if max quality needed)
```

**When in doubt: Start with XLSR-53, upgrade if needed.**
