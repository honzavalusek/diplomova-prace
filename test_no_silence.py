#!/usr/bin/env python3
"""Test WITHOUT silence - only different phoneme patterns"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.features import XLSR53FeatureExtractor, frames_to_seconds
from src.matching import SubsequenceDTWMatcher

print("="*60)
print("TEST: No Silence - Only Different Phoneme Patterns")
print("="*60)

def generate_phoneme(freq, duration_ms=150, sr=16000):
    """Generate a phoneme-like pattern"""
    duration_sec = duration_ms / 1000.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    # Add some complexity with harmonics
    wave = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    return wave * np.hamming(len(wave))  # Apply window

# Query pattern: "ABC" (3 distinct phonemes)
print("\n[1] Creating query: A-B-C")
query_parts = [
    generate_phoneme(300),  # A
    generate_phoneme(600),  # B
    generate_phoneme(900),  # C
]
query = np.concatenate(query_parts)

# Corpus: Different patterns with query embedded
print("[2] Creating corpus: X-Y-Z-[ABC]-P-Q-R")
corpus_parts = [
    generate_phoneme(400),  # X (different from query)
    generate_phoneme(700),  # Y
    generate_phoneme(1000), # Z
    # Query starts here (at position 3 phonemes = ~0.45s)
    generate_phoneme(300),  # A (query)
    generate_phoneme(600),  # B (query)
    generate_phoneme(900),  # C (query)
    # Query ends
    generate_phoneme(450),  # P
    generate_phoneme(750),  # Q
    generate_phoneme(1100), # R
]
corpus = np.concatenate(corpus_parts)

expected_time = 0.45  # 3 phonemes * 0.15s each
print(f"\nQuery duration: {len(query)/16000:.2f}s")
print(f"Corpus duration: {len(corpus)/16000:.2f}s")
print(f"Expected match at: ~{expected_time:.2f}s")

# Extract embeddings
print("\n[3] Extracting embeddings...")
extractor = XLSR53FeatureExtractor(device='cpu')

query_emb = extractor.extract(query.astype(np.float32), 16000)
corpus_emb = extractor.extract(corpus.astype(np.float32), 16000)

print(f"Query: {query_emb.shape}")
print(f"Corpus: {corpus_emb.shape}")

# Match
print("\n[4] Running DTW...")
matcher = SubsequenceDTWMatcher()
result = matcher.match(query_emb, corpus_emb)

start_sec = frames_to_seconds(result.start_frame, 16000, extractor.hop_length)
end_sec = frames_to_seconds(result.end_frame, 16000, extractor.hop_length)

print(f"\nMatch found:")
print(f"  Time: {start_sec:.2f}s - {end_sec:.2f}s")
print(f"  Distance: {result.distance:.4f}")
print(f"  Expected: ~{expected_time:.2f}s")

error = abs(start_sec - expected_time)

print("\n[5] Evaluation:")
if error < 0.2:
    print(f"✓ EXCELLENT: Match is {error:.2f}s off (< 0.2s)")
    print("  System works perfectly!")
elif error < 0.5:
    print(f"✓ GOOD: Match is {error:.2f}s off (< 0.5s)")
    print("  System works well enough")
else:
    print(f"❌ FAILED: Match is {error:.2f}s off")

# Top-k
print("\n[6] Top-3 matches:")
top_matches = matcher.match_top_k(query_emb, corpus_emb, k=3)
for i, m in enumerate(top_matches, 1):
    s = frames_to_seconds(m.start_frame, 16000, extractor.hop_length)
    is_correct = abs(s - expected_time) < 0.2
    marker = "✓" if is_correct else ""
    print(f"  #{i}: {s:.2f}s, distance={m.distance:.4f} {marker}")

print("\n" + "="*60)
if error < 0.5:
    print("✓ System WORKS with non-silence audio!")
    print("\nThe Czech problem is likely:")
    print("1. XLSR-53 not trained well on Czech")
    print("2. Try Czech-specific fine-tuned model")
else:
    print("❌ System has deeper issues")
