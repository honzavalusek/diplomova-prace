#!/usr/bin/env python3
"""Test speaker-independent matching capability"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.features import CzechWav2Vec2Extractor, load_audio
from src.matching import SubsequenceDTWMatcher

print("="*60)
print("CROSS-SPEAKER MATCHING TEST")
print("="*60)

# You need to provide these
speaker1_path = "data/raw_audio/corpus/kancl_1__9_10.wav"  # Original speaker at 17.86s
speaker2_path = "data/raw_audio/corpus/kancl_1__9_10.wav"  # Different speaker at 37.32s

print("\nThis test will compare:")
print("1. Same speaker, same phrase (baseline - should be low distance)")
print("2. Different speaker, same phrase (target - how well does it work?)")

# Extract embeddings
print("\n[1] Loading Czech model...")
extractor = CzechWav2Vec2Extractor(device='cpu')

print("[2] Loading audio...")
corpus_audio, sr = load_audio(speaker1_path)
corpus_emb = extractor.extract(corpus_audio, sr)

# Extract segments
speaker1_start = 17.86  # Your first match
speaker1_end = 18.74
speaker2_start = 37.32  # Second match (different speaker)
speaker2_end = 37.86

speaker1_start_frame = int(speaker1_start * sr / extractor.hop_length)
speaker1_end_frame = int(speaker1_end * sr / extractor.hop_length)
speaker2_start_frame = int(speaker2_start * sr / extractor.hop_length)
speaker2_end_frame = int(speaker2_end * sr / extractor.hop_length)

query_emb = corpus_emb[speaker1_start_frame:speaker1_end_frame]
speaker2_emb = corpus_emb[speaker2_start_frame:speaker2_end_frame]

print(f"\nSpeaker 1 (query): {query_emb.shape}")
print(f"Speaker 2 (different): {speaker2_emb.shape}")

# Compare embeddings
print("\n[3] Embedding similarity analysis:")

# Average embeddings per speaker
query_mean = query_emb.mean(axis=0)
speaker2_mean = speaker2_emb.mean(axis=0)

# Cosine similarity
from src.matching.distance_metrics import cosine_distance
cos_dist = cosine_distance(query_mean, speaker2_mean)
print(f"Cosine distance (speaker 1 vs 2): {cos_dist:.4f}")

# Euclidean distance
euc_dist = np.linalg.norm(query_mean - speaker2_mean)
print(f"Euclidean distance (speaker 1 vs 2): {euc_dist:.4f}")

# DTW distance
print("\n[4] DTW distance:")
matcher = SubsequenceDTWMatcher()

# Self-match (should be ~0)
self_match = matcher.match(query_emb, query_emb)
print(f"Speaker 1 vs itself: {self_match.distance:.4f} (baseline)")

# Cross-speaker match
if len(speaker2_emb) >= len(query_emb):
    cross_match = matcher.match(query_emb, speaker2_emb)
    print(f"Speaker 1 vs Speaker 2: {cross_match.distance:.4f}")

    ratio = cross_match.distance / (self_match.distance + 0.1)
    print(f"\nCross-speaker penalty: {ratio:.1f}x worse")

    if ratio < 3:
        print("✓ Model is relatively speaker-independent!")
    elif ratio < 10:
        print("⚠️  Moderate speaker dependency")
    else:
        print("❌ Highly speaker-dependent")

print("\n[5] Feature analysis:")
# Check variance within vs across speakers
within_var = np.var([query_emb[i] for i in range(len(query_emb))], axis=0).mean()
print(f"Variance within speaker 1: {within_var:.4f}")

print("\n" + "="*60)
print("RECOMMENDATIONS FOR SPEAKER-INDEPENDENT MATCHING:")
print("="*60)
print("\n1. Try larger model (more speaker-invariant):")
print("   --model xls-r-1b --device cuda")
print("\n2. Use phonetic normalization:")
print("   - Extract phoneme-level features")
print("   - Apply speaker normalization techniques")
print("\n3. Adjust DTW parameters:")
print("   - Use cosine distance instead of Euclidean")
print("   - Add feature normalization")
print("\n4. Fine-tune for speaker invariance:")
print("   - Train with speaker-adversarial loss")
print("   - Use triplet loss (same phrase, different speakers)")
print("\n5. Increase threshold:")
print("   - Accept matches up to distance ~80-100 for cross-speaker")
print("   - But this increases false positives")
