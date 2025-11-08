#!/usr/bin/env python3
"""Debug why identical audio has distance 37 instead of ~0"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.features import CzechWav2Vec2Extractor, load_audio, frames_to_seconds
from src.matching import SubsequenceDTWMatcher

print("="*60)
print("DEBUGGING IDENTICAL AUDIO DISTANCE")
print("="*60)

# Files
query_file = "data/raw_audio/queries/extract_python.wav"
corpus_file = "data/raw_audio/corpus/kancl_1__9_10.wav"

print(f"\nQuery: {query_file} (extracted from corpus at 17.86s)")
print(f"Corpus: {corpus_file}")

# Load
extractor = CzechWav2Vec2Extractor(device='cpu')

print("\n[1] Loading and extracting embeddings...")
query_audio, sr = load_audio(query_file)
corpus_audio, sr = load_audio(corpus_file)

query_emb = extractor.extract(query_audio, sr)
corpus_emb = extractor.extract(corpus_audio, sr)

print(f"Query embeddings: {query_emb.shape}")
print(f"Corpus embeddings: {corpus_emb.shape}")

# Test 1: Self-match (query vs itself)
print("\n[2] Self-match test (query vs itself):")
matcher = SubsequenceDTWMatcher()
self_result = matcher.match(query_emb, query_emb)

print(f"Distance: {self_result.distance:.4f}")
print(f"Segment: {self_result.start_frame} - {self_result.end_frame}")

if self_result.distance < 1.0:
    print("✓ Self-match works (distance near 0)")
else:
    print(f"❌ PROBLEM: Self-match has distance {self_result.distance:.4f}")
    print("   DTW implementation might be broken!")

# Test 2: Extract exact segment from corpus
print("\n[3] Extracting exact segment from corpus embeddings...")
expected_start = 17.86
expected_end = 18.60  # From your output

start_frame = int(expected_start * sr / extractor.hop_length)
end_frame = int(expected_end * sr / extractor.hop_length)

corpus_segment = corpus_emb[start_frame:end_frame]

print(f"Expected frames: {start_frame} - {end_frame}")
print(f"Query: {query_emb.shape}")
print(f"Corpus segment: {corpus_segment.shape}")

# Direct comparison of embeddings
if query_emb.shape == corpus_segment.shape:
    print("\n[4] Direct embedding comparison:")
    diff = np.abs(query_emb - corpus_segment)
    print(f"Mean absolute difference: {diff.mean():.6f}")
    print(f"Max difference: {diff.max():.6f}")

    if diff.mean() < 0.01:
        print("✓ Embeddings are nearly identical!")
    else:
        print(f"❌ Embeddings differ by {diff.mean():.6f}")
        print("   The model is producing different embeddings for the same audio!")
else:
    print(f"\n[4] Shape mismatch:")
    print(f"Query: {query_emb.shape[0]} frames")
    print(f"Corpus segment: {corpus_segment.shape[0]} frames")
    print(f"Difference: {abs(query_emb.shape[0] - corpus_segment.shape[0])} frames")

# Test 3: Small window around expected location
print("\n[5] Testing DTW on small window around expected location...")
window_start = max(0, start_frame - 20)
window_end = min(len(corpus_emb), end_frame + 20)
corpus_window = corpus_emb[window_start:window_end]

window_result = matcher.match(query_emb, corpus_window)
actual_frame = window_start + window_result.start_frame

print(f"Window size: {corpus_window.shape}")
print(f"DTW distance: {window_result.distance:.4f}")
print(f"Found at frame: {actual_frame} (expected: {start_frame})")

# Test 4: Full corpus search (what the demo does)
print("\n[6] Testing full corpus search (what demo does)...")
full_result = matcher.match(query_emb, corpus_emb)

found_time = frames_to_seconds(full_result.start_frame, sr, extractor.hop_length)
print(f"Distance: {full_result.distance:.4f}")
print(f"Found at: {found_time:.2f}s (expected: {expected_start:.2f}s)")

# Compare
print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

if self_result.distance < 1.0:
    print("✓ DTW works (self-match is good)")
else:
    print("❌ DTW is broken")

if query_emb.shape == corpus_segment.shape:
    diff_mean = np.abs(query_emb - corpus_segment).mean()
    if diff_mean < 0.01:
        print("✓ Embeddings are reproducible")
    else:
        print(f"❌ Embeddings differ by {diff_mean:.6f}")
        print("   Model is non-deterministic or extraction has issues")
else:
    print("⚠️  Frame count mismatch - extraction timing issue")

if full_result.distance > 30:
    print(f"❌ Full corpus search gives distance {full_result.distance:.4f}")
    print("\nPossible causes:")
    print("1. Subsequence DTW is matching wrong segment")
    print("2. DTW parameters need tuning")
    print("3. Query is too short for reliable matching")
    print("4. Model produces unstable embeddings")

print(f"\nQuery duration: {len(query_audio)/sr:.2f}s")
if len(query_audio)/sr < 1.0:
    print("⚠️  Query is < 1 second - might be too short for reliable DTW")
