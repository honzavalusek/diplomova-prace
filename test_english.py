#!/usr/bin/env python3
"""Test with English audio to verify the system works at all"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("ENGLISH AUDIO TEST")
print("="*60)
print()
print("This test will use synthetic English-like audio patterns")
print("to verify the DTW matching works in principle.")
print()

import numpy as np
from src.features import XLSR53FeatureExtractor, frames_to_seconds
from src.matching import SubsequenceDTWMatcher

# Generate synthetic "phoneme-like" patterns
def generate_phoneme(freq, duration_ms=100, sr=16000):
    """Generate a sine wave representing a phoneme"""
    duration_sec = duration_ms / 1000.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    return np.sin(2 * np.pi * freq * t)

# Create a query: phoneme pattern "A-B-C"
print("[1] Creating synthetic query pattern: A-B-C")
phoneme_a = generate_phoneme(300, 200)  # Low freq
phoneme_b = generate_phoneme(600, 200)  # Mid freq
phoneme_c = generate_phoneme(900, 200)  # High freq

query_audio = np.concatenate([phoneme_a, phoneme_b, phoneme_c])
print(f"Query duration: {len(query_audio)/16000:.2f}s")

# Create corpus with query embedded at known location
print("\n[2] Creating corpus with query at 5.0 seconds")
silence_before = np.zeros(int(16000 * 5.0))  # 5 seconds silence
silence_after = np.zeros(int(16000 * 3.0))   # 3 seconds after

# Also add a different pattern as noise
noise_pattern = np.concatenate([
    generate_phoneme(400, 200),
    generate_phoneme(700, 200),
    generate_phoneme(1000, 200),
])

corpus_audio = np.concatenate([
    silence_before,
    query_audio,  # Exact match at 5.0s
    silence_after,
    noise_pattern,  # Different pattern at 8.6s
    np.zeros(int(16000 * 2.0))
])

print(f"Corpus duration: {len(corpus_audio)/16000:.2f}s")
print(f"Query should be found at: 5.00s")

# Extract features
print("\n[3] Extracting XLSR features...")
extractor = XLSR53FeatureExtractor(device='cpu')

query_emb = extractor.extract(query_audio.astype(np.float32), 16000)
corpus_emb = extractor.extract(corpus_audio.astype(np.float32), 16000)

print(f"Query embeddings: {query_emb.shape}")
print(f"Corpus embeddings: {corpus_emb.shape}")

# Match
print("\n[4] Running DTW matching...")
matcher = SubsequenceDTWMatcher()
result = matcher.match(query_emb, corpus_emb)

start_sec = frames_to_seconds(result.start_frame, 16000, extractor.hop_length)
end_sec = frames_to_seconds(result.end_frame, 16000, extractor.hop_length)

print(f"\nBest match found:")
print(f"  Time: {start_sec:.2f}s - {end_sec:.2f}s")
print(f"  Distance: {result.distance:.4f}")
print(f"  Expected: 5.00s")

# Evaluate
print("\n[5] Evaluation:")
time_error = abs(start_sec - 5.0)

if time_error < 0.5:  # Within 500ms
    print(f"✓ EXCELLENT: Match within {time_error:.2f}s of expected")
    print("  The system works correctly!")
elif time_error < 2.0:  # Within 2s
    print(f"⚠️  ACCEPTABLE: Match within {time_error:.2f}s of expected")
    print("  System works but not perfectly")
else:
    print(f"❌ FAILED: Match is {time_error:.2f}s off")
    print("  System has fundamental problems")

if result.distance < 10:
    print(f"✓ Good distance: {result.distance:.4f}")
else:
    print(f"⚠️  High distance: {result.distance:.4f}")

# Test top-k
print("\n[6] Testing top-k matches...")
top_matches = matcher.match_top_k(query_emb, corpus_emb, k=3)

print(f"Top {len(top_matches)} matches:")
for i, m in enumerate(top_matches, 1):
    s = frames_to_seconds(m.start_frame, 16000, extractor.hop_length)
    e = frames_to_seconds(m.end_frame, 16000, extractor.hop_length)
    is_correct = abs(s - 5.0) < 0.5
    marker = "✓ CORRECT!" if is_correct else ""
    print(f"  #{i}: {s:.2f}s-{e:.2f}s, distance={m.distance:.4f} {marker}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)

if time_error < 2.0 and result.distance < 20:
    print("✓ System works! The problem is Czech-specific.")
    print("\nNext steps:")
    print("1. Try Czech fine-tuned model: arampacha/wav2vec2-large-xlsr-czech")
    print("2. Or use XLS-R 1B with --model xls-r-1b --device cuda")
else:
    print("❌ System has fundamental issues even with synthetic audio")
    print("Need to debug the pipeline further")
