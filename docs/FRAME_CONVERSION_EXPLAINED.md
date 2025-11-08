# Frame-to-Time Conversion Explained

## The Problem: Two Different "Clocks"

Your acoustic retrieval system operates in **two different time domains**:

### Domain 1: Audio Samples (Raw Waveform)
```
Audio file at 16,000 Hz = 16,000 samples per second

Example: 10-second audio file = 160,000 samples
         [sample_0, sample_1, sample_2, ..., sample_159999]
```

### Domain 2: Feature Frames (XLSR Embeddings)
```
XLSR processes audio in chunks (hop_length = 320 samples)

Same 10-second file = 500 feature frames
         [frame_0, frame_1, frame_2, ..., frame_499]

Each frame = 1024-dimensional embedding vector
```

## The Gap

When DTW finds a match, it returns results in **Domain 2 (frames)**:

```python
match = matcher.match(query_emb, corpus_emb)
# MatchResult(start_frame=100, end_frame=150)
```

But users need results in **real-world time (seconds)**:
```
"Your query was found at 2.0s - 3.0s in the audio file"
```

**Without conversion, you can't use the results!**

---

## How Conversion Works

### The Relationship

```
┌─────────────────────────────────────────────────────────┐
│  Raw Audio (16,000 Hz)                                  │
│  [•••••••••••••••••••••••••••••••••••••••••••••••••]    │
│   ↑                                                      │
│   └─ 320 samples (hop_length)                           │
│                                                          │
│  ↓ XLSR Feature Extractor                               │
│                                                          │
│  Feature Frames (1 per 320 samples)                     │
│  [Frame_0] [Frame_1] [Frame_2] ... [Frame_N]            │
│   ↑                                                      │
│   └─ Each frame = 1024-D embedding                      │
└─────────────────────────────────────────────────────────┘

Frame Index → Samples → Seconds
   frame_idx × hop_length / sample_rate = time_seconds
```

### Concrete Example

**Setup:**
- Sample rate: 16,000 Hz
- Hop length: 320 samples
- DTW result: Match at frames 100-150

**Calculation:**

```python
# Frame 100 → Time in seconds
start_time = (100 × 320) / 16000
           = 32000 / 16000
           = 2.0 seconds

# Frame 150 → Time in seconds
end_time = (150 × 320) / 16000
         = 48000 / 16000
         = 3.0 seconds
```

**Result:** "Match found from 2.0s to 3.0s"

---

## Visual Timeline Example

```
Audio File: "podcast.wav" (10 seconds, 160,000 samples)

Raw Samples Timeline (16,000 Hz):
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
0s      1s      2s      3s      4s      5s      6s      7s      8s      9s     10s
0     16000   32000   48000   64000   80000   96000  112000  128000  144000  160000
                ↑       ↑
            sample     sample
            32000      48000


Feature Frames Timeline (hop_length = 320):
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
0       50      100     150     200     250     300     350     400     450     500
                ↑       ↑
              frame    frame
               100      150


DTW Match Result:
                [==================]
                Frame 100 → Frame 150
                   ↓           ↓
                  2.0s        3.0s  ← After conversion!
```

---

## Why This Matters in Your Pipeline

### Without Frame Conversion:
```python
match = matcher.match(query_emb, corpus_emb)
print(f"Match at frames {match.start_frame}-{match.end_frame}")
# Output: "Match at frames 100-150"
# ❌ User has NO IDEA where this is in the actual audio!
```

### With Frame Conversion:
```python
match = matcher.match(query_emb, corpus_emb)

start_sec = frames_to_seconds(match.start_frame,
                               sample_rate=16000,
                               hop_length=320)
end_sec = frames_to_seconds(match.end_frame,
                             sample_rate=16000,
                             hop_length=320)

print(f"Match found at {start_sec:.2f}s - {end_sec:.2f}s")
# Output: "Match found at 2.00s - 3.00s"
# ✅ User can jump to exact timestamp in media player!
```

---

## The Reverse Operation: seconds_to_frames()

Sometimes you need to go the other way (time → frames):

### Use Case 1: Start Search from Specific Time
```python
# "Start searching from 5.5 seconds into the audio"
start_time = 5.5  # seconds

# Convert to frame index
start_frame = seconds_to_frames(5.5, sample_rate=16000, hop_length=320)
# → 275 frames

# Now slice embeddings to only search from that point
corpus_emb_subset = corpus_emb[start_frame:]
```

### Use Case 2: Extract Specific Audio Segment
```python
# "Extract the audio segment from 2.0s to 3.0s"
start_frame = seconds_to_frames(2.0)  # → 100
end_frame = seconds_to_frames(3.0)    # → 150

# Get the embeddings for that time range
segment_emb = corpus_emb[start_frame:end_frame]
```

---

## Key Parameters

### hop_length = 320 (XLSR-specific)

This is **NOT arbitrary** - it's determined by XLSR's architecture:

```
Wav2Vec2/XLSR CNN Encoder:
- Multiple convolutional layers with stride
- Effective temporal stride = 320 samples
- This means: 1 feature vector per 320 audio samples
```

**At 16kHz:**
- 320 samples = 320/16000 = 0.02 seconds = 20ms
- So each frame represents ~20ms of audio
- 50 frames per second

### sample_rate = 16000 (XLSR requirement)

XLSR models are **trained on 16kHz audio only**:
- Must resample all audio to 16kHz before processing
- Using different sample rates will break the model

---

## Common Questions

### Q: Can I change hop_length?

**A:** No, not for XLSR. The hop_length=320 is baked into the model architecture. If you use a different feature extractor (e.g., MFCCs with librosa), then you can set custom hop_length.

### Q: What if my audio is 44.1kHz (CD quality)?

**A:** librosa.load() automatically resamples to 16kHz:
```python
audio, sr = librosa.load('audio.wav', sr=16000)
# audio is now resampled to 16kHz
```

### Q: Is frame conversion lossy?

**A:** Yes, slightly. You lose sub-frame precision:
```
Frame 100 → 2.0s
Frame 101 → 2.02s

You can't represent 2.01s exactly in frame space.
Precision = hop_length / sample_rate = 320/16000 = 0.02s = 20ms
```

For speech, 20ms precision is excellent (phonemes are ~50-200ms).

---

## Code Example: Full Pipeline

```python
from src.features import XLSRFeatureExtractor, load_audio, frames_to_seconds
from src.matching import SubsequenceDTWMatcher

# 1. Extract features
extractor = XLSRFeatureExtractor()
query_audio, sr = load_audio('query.wav')
corpus_audio, sr = load_audio('corpus.wav')

query_emb = extractor.extract(query_audio, sr)
corpus_emb = extractor.extract(corpus_audio, sr)

# 2. Find match (returns FRAME indices)
matcher = SubsequenceDTWMatcher()
match = matcher.match(query_emb, corpus_emb)

print(f"DTW raw result: frames {match.start_frame}-{match.end_frame}")
# Output: "DTW raw result: frames 100-150"
# ❌ Not useful for end user!

# 3. Convert to timestamps (THE CRITICAL STEP)
start_sec = frames_to_seconds(
    match.start_frame,
    sample_rate=sr,
    hop_length=extractor.hop_length  # 320 for XLSR
)
end_sec = frames_to_seconds(
    match.end_frame,
    sample_rate=sr,
    hop_length=extractor.hop_length
)

print(f"User-friendly result: {start_sec:.2f}s - {end_sec:.2f}s")
# Output: "User-friendly result: 2.00s - 3.00s"
# ✅ User can now navigate to this time in any media player!
```

---

## Summary

**frames_to_seconds()** and **seconds_to_frames()** are the **synchronization bridge** between:

1. **Machine representation** (discrete frame indices from DTW)
2. **Human representation** (continuous time in seconds)

**Without these functions:**
- DTW results are unusable (just abstract frame numbers)
- Can't tell users where matches occur
- Can't extract specific time segments
- Can't integrate with media players or UIs

**With these functions:**
- DTW results become actionable timestamps
- Users can jump to exact match locations
- Can build time-based queries ("search from 1:30 to 2:00")
- Can export timestamped results for analysis

**Think of it like:**
- DTW speaks "frame language" (0, 1, 2, 100, 150...)
- Users speak "time language" (0.0s, 2.0s, 3.0s...)
- Frame conversion is the **translator** between them.
