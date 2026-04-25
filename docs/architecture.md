# Architecture

## Pipeline

```
Audio → Feature Extraction (SSL model) → Subsequence DTW → Frame-to-Time → Timestamps
```

1. **Query / corpus audio** is loaded at 16 kHz.
2. **Feature extraction** runs a self-supervised speech model (WavLM, XLSR-53, XLS-R) and returns contextualized embeddings as `(frames, dim)`.
3. **Subsequence DTW** aligns the query against any contiguous span of the corpus using cosine distance and a Sakoe-Chiba band.
4. **Frame-to-time** converts DTW frame indices to seconds using `time_sec = frame_idx * 320 / 16000` (hop length 320 ≈ 20 ms/frame).

## Core modules

**`src/features/`**
- `ssl_speech_extractor.py` — `SSLSpeechExtractor`: loads HuggingFace SSL speech models, extracts contextualized embeddings as `(frames, dim)` arrays. Supports layer-range averaging and half-precision.
- `audio_preprocessing.py` — `load_audio()`: librosa-based loader enforcing 16 kHz.
- `frame_conversion.py` — `frames_to_seconds()` and inverse; bridges DTW frame indices to timestamps.

**`src/matching/`**
- `subsequence_dtw.py` — `SubsequenceDTWMatcher`: wraps `dtaidistance` for subsequence alignment. L2-normalizes embeddings to use cosine distance. Sakoe-Chiba window constraint speeds matching ~4×. Returns `MatchResult` with start/end frames and distance score.

## Models

| Model | Embedding dim | Notes |
|-------|---------------|-------|
| `xlsr-53` | 1024 | Facebook cross-lingual |
| `xls-r-300m` / `1b` / `2b` | 1024 | Larger XLS-R variants |
| `wavlm-base` / `base-plus` | 768 | Microsoft WavLM |
| `wavlm-large` | 1024 | Large WavLM |
| `czech`, `czech2` | varies | Czech-specific fine-tuned models |

## Key constants

- Sample rate: 16 000 Hz (enforced)
- Hop length: 320 samples (≈ 20 ms/frame)
- DTW: `dtaidistance` (C-optimized), cosine distance via L2-normalized embeddings, Sakoe-Chiba band (e.g., 25 frames)
