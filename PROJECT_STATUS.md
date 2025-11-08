# Project Status: Acoustic Retrieval System

**Last Updated:** 2025-11-08
**Phase:** 1 (Foundation) - ✅ COMPLETE

---

## ✅ What's Built (Ready to Use)

### Core Modules

| Module | Status | Files | Description |
|--------|--------|-------|-------------|
| **Module 1: Feature Extraction** | ✅ Complete | `src/features/` | XLSR-53 embedding extraction, audio preprocessing |
| **Module 3: DTW Matching** | ✅ Complete | `src/matching/` | Subsequence DTW with frame-accurate alignment |
| **End-to-End Demo** | ✅ Complete | `scripts/01_minimal_demo.py` | Working QbE-STD without LSH |

### Capabilities (Right Now)

You can already:
- ✅ Extract high-quality acoustic features from any audio file
- ✅ Search for spoken queries in corpus recordings
- ✅ Get precise timestamps (start/end) of matches
- ✅ Process queries in ~2-5 seconds per corpus file

**Limitation:** No LSH indexing yet = linear search only (fine for <100 files)

---

## 🚀 How to Start (3 Commands)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add your audio files
# Place query: data/raw_audio/queries/query.wav
# Place corpus: data/raw_audio/corpus/*.wav

# 3. Run
python scripts/01_minimal_demo.py \
    --query data/raw_audio/queries/query.wav \
    --corpus data/raw_audio/corpus/*.wav
```

**See:** `QUICKSTART.md` for detailed instructions + test audio generation

---

## 📊 Performance Benchmarks (Estimated)

| Corpus Size | Without LSH | With LSH (Phase 2) |
|-------------|-------------|---------------------|
| 10 files (~5 min audio) | ~20s | ~2s |
| 100 files (~50 min) | ~3 min | ~5s |
| 1000 files (~8 hours) | ~30 min | ~10s |
| 10,000 files (~80 hours) | ❌ Infeasible | ~15s |

*Times based on CPU, single-threaded. GPU 10-20x faster.*

---

## ⏳ Next Steps (Phase 2: Indexing)

### Priority 1: LSH Filtering (Module 2)

**Why:** Enable sub-linear search for large corpora (1000+ hours)

**Tasks:**
1. Implement MinHash LSH (prototype)
   - File: `src/indexing/minhash_lsh.py`
   - Integrate with demo script

2. Add learned hash functions (production)
   - File: `src/indexing/learned_hash.py`
   - Train on embeddings

3. FAISS integration (optional, for massive scale)
   - File: `src/indexing/faiss_index.py`
   - HNSW or IVF-PQ index

**Expected Result:** Search 10,000 files in <15 seconds

### Priority 2: Corpus Management

**Why:** Pre-compute embeddings once, reuse forever

**Tasks:**
1. Audio segmentation (split long files)
   - File: `src/corpus/segmentation.py`
   - Split files into 1-5 minute chunks

2. Embedding cache (store .npy files)
   - File: `src/corpus/embedding_cache.py`
   - Save/load embeddings from disk

3. Metadata tracking (file IDs, timestamps)
   - File: `src/corpus/metadata_manager.py`
   - CSV/DB with segment info

**Expected Result:** Instant re-querying (no re-extraction)

### Priority 3: Production Pipeline

**Why:** Package as deployable service

**Tasks:**
1. End-to-end pipeline class
   - File: `src/pipeline/qbe_std_pipeline.py`
   - Single API: `search(query_path) → results`

2. REST API (FastAPI)
   - File: `deployment/api/app.py`
   - Endpoints: `/upload_query`, `/search`

3. Docker deployment
   - File: `deployment/docker/Dockerfile`
   - Container with all dependencies

---

## 📁 Current Directory Structure

```
acoustic-retrieval/
├── README.md              ← Architecture overview
├── QUICKSTART.md          ← Getting started guide
├── PROJECT_STATUS.md      ← This file
├── requirements.txt       ← Dependencies
├── config/
│   └── model_config.yaml  ← XLSR settings
├── src/
│   ├── features/          ← ✅ Module 1 (complete)
│   │   ├── xlsr_extractor.py
│   │   ├── audio_preprocessing.py
│   │   └── frame_conversion.py
│   ├── matching/          ← ✅ Module 3 (complete)
│   │   ├── subsequence_dtw.py
│   │   └── distance_metrics.py
│   ├── indexing/          ← ⏳ Module 2 (TODO)
│   ├── pipeline/          ← ⏳ Module 4 (TODO)
│   └── corpus/            ← ⏳ Corpus mgmt (TODO)
├── scripts/
│   └── 01_minimal_demo.py ← ✅ Working demo
├── data/
│   ├── raw_audio/
│   │   ├── queries/       ← Put query audio here
│   │   └── corpus/        ← Put corpus audio here
│   └── embeddings/        ← Cached features (future)
└── tests/                 ← Unit tests (TODO)
```

---

## 🎯 Recommended Development Path

### Week 1: Validation
- [x] Run demo on real audio files
- [ ] Validate accuracy on known queries
- [ ] Benchmark performance (latency, memory)

### Week 2: Indexing
- [ ] Implement MinHash LSH
- [ ] Integrate with demo script
- [ ] Compare recall vs. latency trade-offs

### Week 3: Optimization
- [ ] Add embedding cache
- [ ] Implement learned hash (if needed)
- [ ] Optimize DTW parameters (window size)

### Week 4: Production
- [ ] Build full pipeline
- [ ] Add REST API
- [ ] Write comprehensive tests

---

## 📚 Key Technical Details

### XLSR-53 Embeddings
- **Dimension:** 1024
- **Sample Rate:** 16,000 Hz (enforced)
- **Hop Length:** 320 samples (~20ms per frame)
- **Output:** Contextualized (last_hidden_state)

### Subsequence DTW
- **Library:** dtaidistance (C-optimized)
- **Distance Metric:** Euclidean (default), Cosine (recommended for neural embeddings)
- **Returns:** (distance, start_frame, end_frame)

### Frame-to-Time Conversion
```python
time_sec = (frame_idx * 320) / 16000

# Example: frame 100
# 100 * 320 / 16000 = 2.0 seconds
```

---

## 🔧 Configuration Files

### `config/model_config.yaml`
```yaml
model:
  name: "facebook/wav2vec2-large-xlsr-53"
  output_layer: "last_hidden_state"

audio:
  target_sample_rate: 16000
  hop_length: 320

device: "cpu"  # Change to "cuda" for GPU
```

---

## 🐛 Known Limitations

1. **No LSH indexing** → Linear search (O(n) per query)
2. **No embedding cache** → Re-extracts features every run
3. **No batching** → Processes files sequentially
4. **CPU only (default)** → GPU support requires `--device cuda`
5. **No corpus segmentation** → Long files consume high memory

**All addressed in Phase 2.**

---

## 📖 References

See blueprint document for:
- Architecture diagrams
- Algorithm details
- Cited research papers
- Performance analysis

---

## 🤝 Contributing

To extend the system:

1. **Add new feature extractor:** Implement in `src/features/`
2. **Add new LSH method:** Implement in `src/indexing/`
3. **Add new distance metric:** Implement in `src/matching/distance_metrics.py`
4. **Add tests:** Create in `tests/`

---

**Status:** Ready for real-world testing on small-to-medium corpora (<100 files).
**Next Milestone:** Implement LSH indexing for large-scale deployment.
