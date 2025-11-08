#!/usr/bin/env python3
"""Diagnose why extracted query has distance 41 instead of ~0"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.features import CzechWav2Vec2Extractor, load_audio, frames_to_seconds
from src.matching import SubsequenceDTWMatcher

print("="*60)
print("DIAGNOSING EXTRACTION ISSUE")
print("="*60)

# Files
query_path = "data/raw_audio/queries/extract.wav"  # Your extracted query
corpus_path = "data/raw_audio/corpus/kancl_1__9_10.wav"  # Source file

print(f"\nQuery (extracted): {query_path}")
print(f"Corpus (original): {corpus_path}")

# Load audio
print("\n[1] Loading raw audio waveforms...")
query_audio, sr_q = load_audio(query_path)
corpus_audio, sr_c = load_audio(corpus_path)

print(f"Query audio: {len(query_audio)} samples, {sr_q} Hz")
print(f"Corpus audio: {len(corpus_audio)} samples, {sr_c} Hz")
print(f"Query duration: {len(query_audio)/sr_q:.2f}s")
print(f"Corpus duration: {len(corpus_audio)/sr_c:.2f}s")

# Extract the "exact" segment from corpus at 17.86s
expected_start_sec = 17.86
expected_duration_sec = 0.88
expected_start_sample = int(expected_start_sec * sr_c)
expected_end_sample = int((expected_start_sec + expected_duration_sec) * sr_c)

corpus_segment = corpus_audio[expected_start_sample:expected_end_sample]

print(f"\n[2] Extracted segment from corpus at {expected_start_sec}s:")
print(f"Corpus segment: {len(corpus_segment)} samples")
print(f"Query: {len(query_audio)} samples")
print(f"Difference: {abs(len(corpus_segment) - len(query_audio))} samples")

# Compare raw waveforms
if len(corpus_segment) == len(query_audio):
    waveform_diff = np.mean(np.abs(corpus_segment - query_audio))
    waveform_corr = np.corrcoef(corpus_segment, query_audio)[0, 1]
    print(f"\nRaw waveform comparison:")
    print(f"  Mean absolute difference: {waveform_diff:.6f}")
    print(f"  Correlation: {waveform_corr:.6f}")

    if waveform_corr > 0.99:
        print("  ✓ Waveforms are nearly identical!")
    elif waveform_corr > 0.9:
        print("  ⚠️  Waveforms are similar but not identical")
    else:
        print("  ❌ Waveforms are different!")
else:
    print("  ⚠️  Different lengths - can't compare directly")

# Extract embeddings
print("\n[3] Extracting embeddings with Czech model...")
extractor = CzechWav2Vec2Extractor(device='cpu')

query_emb = extractor.extract(query_audio, sr_q)
corpus_emb = extractor.extract(corpus_audio, sr_c)

print(f"Query embeddings: {query_emb.shape}")
print(f"Corpus embeddings: {corpus_emb.shape}")

# Extract the embedding segment
expected_start_frame = int(expected_start_sec * sr_c / extractor.hop_length)
expected_end_frame = int((expected_start_sec + expected_duration_sec) * sr_c / extractor.hop_length)

corpus_emb_segment = corpus_emb[expected_start_frame:expected_end_frame]

print(f"\n[4] Comparing embeddings:")
print(f"Query embedding shape: {query_emb.shape}")
print(f"Corpus segment embedding: {corpus_emb_segment.shape}")

# Direct embedding comparison
if query_emb.shape[0] == corpus_emb_segment.shape[0]:
    emb_diff = np.mean(np.abs(query_emb - corpus_emb_segment))
    emb_l2 = np.linalg.norm(query_emb - corpus_emb_segment)

    print(f"  Mean absolute difference: {emb_diff:.6f}")
    print(f"  L2 distance: {emb_l2:.4f}")

    if emb_l2 < 1.0:
        print("  ✓ Embeddings are nearly identical!")
    elif emb_l2 < 10.0:
        print("  ⚠️  Embeddings are similar")
    else:
        print("  ❌ Embeddings are quite different!")
else:
    print(f"  ⚠️  Different frame counts: {query_emb.shape[0]} vs {corpus_emb_segment.shape[0]}")

# Run DTW on the exact segment
print("\n[5] Running DTW on extracted segment...")
matcher = SubsequenceDTWMatcher()

# Create a small window around the expected location
window_start = max(0, expected_start_frame - 50)
window_end = min(len(corpus_emb), expected_end_frame + 50)
corpus_window = corpus_emb[window_start:window_end]

result = matcher.match(query_emb, corpus_window)

actual_frame = window_start + result.start_frame
actual_time = frames_to_seconds(actual_frame, sr_c, extractor.hop_length)

print(f"DTW result:")
print(f"  Distance: {result.distance:.4f}")
print(f"  Found at: {actual_time:.2f}s (expected: {expected_start_sec:.2f}s)")
print(f"  Time error: {abs(actual_time - expected_start_sec):.2f}s")

# Self-match test
print("\n[6] Self-match test (query vs itself):")
self_result = matcher.match(query_emb, query_emb)
print(f"  Distance: {self_result.distance:.4f} (should be ~0)")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

if result.distance < 5:
    print("✓ DTW works fine when directly comparing the segment")
    print("  Problem is likely with the full corpus search")
elif self_result.distance > 1:
    print("❌ Even self-match fails - DTW or embedding issue")
else:
    print("⚠️  Embeddings are different for the same audio")
    print("\nPossible causes:")
    print("1. FFmpeg re-encoded the query during extraction")
    print("2. Different normalization applied")
    print("3. Audio preprocessing differences in librosa")
    print("\nSolution: Extract using Python instead of FFmpeg")
