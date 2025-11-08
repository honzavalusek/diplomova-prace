# Scalable Acoustic Retrieval System

Query-by-Example Spoken Term Detection using XLSR-53 embeddings, LSH filtering, and Subsequence DTW matching.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Minimal Demo (No LSH)

Test the core concept on small audio files:

```bash
# Example: Search for a spoken word in audio files
python scripts/01_minimal_demo.py \
    --query data/raw_audio/queries/hello.wav \
    --corpus data/raw_audio/corpus/*.wav
```

This validates:
- ✅ XLSR-53 feature extraction
- ✅ Subsequence DTW matching
- ✅ Frame-to-timestamp conversion

**Expected output:**
```
Best match found in: recording1.wav
  Time range: 5.23s - 6.45s
  Distance: 0.1234
```

### 3. Project Structure

```
src/
├── features/           # Module 1: XLSR-53 embedding extraction
├── matching/           # Module 3: Subsequence DTW
├── indexing/          # Module 2: LSH (TODO)
└── pipeline/          # Module 4: End-to-end system (TODO)

scripts/
├── 01_minimal_demo.py # Start here - no LSH required
├── 02_extract_features.py  # Batch processing (TODO)
└── 03_build_index.py       # LSH indexing (TODO)
```

## Development Roadmap

**Phase 1: Foundation** (✅ COMPLETE)
- [x] Feature extraction (XLSR-53)
- [x] DTW matching
- [x] Minimal end-to-end demo

**Phase 2: Indexing** (⏳ IN PROGRESS)
- [ ] MinHash LSH prototype
- [ ] Learned hash functions
- [ ] FAISS integration

**Phase 3: Production**
- [ ] Corpus segmentation
- [ ] Embedding caching
- [ ] REST API
- [ ] Docker deployment

## Architecture

```
Audio → XLSR-53 → [Filter: LSH] → [Verify: S-DTW] → Timestamps
         (1024D)    (sub-linear)    (high precision)
```

**Two-Stage Pipeline:**
1. **Filter (LSH):** Reduce billions of vectors to top-k candidates
2. **Verify (S-DTW):** Frame-accurate temporal alignment

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers (Hugging Face)
- librosa
- dtaidistance

GPU optional but recommended for large-scale deployment.

## Citation

Based on research in self-supervised speech representation and acoustic retrieval:
- XLSR-53: Cross-lingual speech representation learning
- DTW: Dynamic time warping for speech alignment
- LSH: Locality-sensitive hashing for scalable search
