#!/usr/bin/env python3
"""Debug script to test exact matching"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.features import XLSR53FeatureExtractor, load_audio, frames_to_seconds
from src.matching import SubsequenceDTWMatcher

print("="*60)
print("DEBUG: Testing exact match scenario")
print("="*60)

# Load your actual files (UPDATE THESE PATHS)
query_path = "data/raw_audio/queries/YOUR_QUERY.wav"  # UPDATE THIS
corpus_path = "data/raw_audio/corpus/YOUR_CORPUS.wav"  # UPDATE THIS

print(f"\nQuery: {query_path}")
print(f"Corpus: {corpus_path}")

# Extract features
print("\n[1] Extracting features...")
extractor = XLSR53FeatureExtractor(device='cpu')

query_audio, sr = load_audio(query_path)
corpus_audio, sr = load_audio(corpus_path)

query_emb = extractor.extract(query_audio, sr)
corpus_emb = extractor.extract(corpus_audio, sr)

print(f"Query embeddings: {query_emb.shape}")
print(f"Corpus embeddings: {corpus_emb.shape}")

# Test 1: Single best match
print("\n[2] Testing single best match (original method)...")
matcher = SubsequenceDTWMatcher()
match = matcher.match(query_emb, corpus_emb)

start_sec = frames_to_seconds(match.start_frame, sr, extractor.hop_length)
end_sec = frames_to_seconds(match.end_frame, sr, extractor.hop_length)

print(f"\nBEST MATCH:")
print(f"  Distance: {match.distance:.4f}")
print(f"  Frames: {match.start_frame} - {match.end_frame}")
print(f"  Time: {start_sec:.2f}s - {end_sec:.2f}s")

# Test 2: Top-k matches
print("\n[3] Testing top-k matches...")
top_matches = matcher.match_top_k(query_emb, corpus_emb, k=3)

print(f"\nTOP {len(top_matches)} MATCHES:")
for i, m in enumerate(top_matches, 1):
    s = frames_to_seconds(m.start_frame, sr, extractor.hop_length)
    e = frames_to_seconds(m.end_frame, sr, extractor.hop_length)
    print(f"  #{i}: distance={m.distance:.4f}, time={s:.2f}s-{e:.2f}s, frames={m.start_frame}-{m.end_frame}")

# Test 3: Self-match (should be perfect)
print("\n[4] SANITY CHECK: Query matching itself...")
self_match = matcher.match(query_emb, query_emb)
print(f"  Distance: {self_match.distance:.4f} (should be ~0)")
print(f"  Frames: {self_match.start_frame} - {self_match.end_frame}")

if self_match.distance > 1.0:
    print("  ❌ PROBLEM: Query doesn't even match itself!")
else:
    print("  ✓ Self-match works")

# Test 4: Check if query is actually in corpus
print("\n[5] Where does the query audio appear in the corpus file?")
print("  (You need to tell me this - at what timestamp?)")
expected_start = float(input("  Expected start time (seconds): "))

expected_start_frame = int(expected_start * sr / extractor.hop_length)
print(f"  Expected start frame: ~{expected_start_frame}")
print(f"  Actual best match frame: {match.start_frame}")
print(f"  Difference: {abs(match.start_frame - expected_start_frame)} frames")

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)

if self_match.distance < 1.0:
    print("✓ DTW is working (self-match is good)")
else:
    print("❌ DTW is broken (self-match fails)")

if match.distance < 10:
    print("✓ Best match has good distance")
else:
    print("❌ Best match has poor distance")

if abs(match.start_frame - expected_start_frame) < 50:
    print("✓ Match location is correct")
else:
    print("❌ Match location is WRONG")
