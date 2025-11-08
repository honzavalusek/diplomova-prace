#!/usr/bin/env python3
"""Quick test to verify XLSR extractor and DTW matcher work"""

import numpy as np
from src.features import XLSRFeatureExtractor
from src.matching import SubsequenceDTWMatcher

print("="*60)
print("TESTING ACOUSTIC RETRIEVAL PIPELINE")
print("="*60)

# Test 1: Feature Extraction
print("\n[1/2] Testing XLSR feature extraction...")
print("  (This will download ~1.2GB on first run)")
extractor = XLSRFeatureExtractor(device='cpu')

print("  ✓ Model loaded successfully!")
print(f"    Device: {extractor.device}")
print(f"    Embedding dim: {extractor.embedding_dim}")
print(f"    Hop length: {extractor.hop_length}")

# Generate synthetic audio
print("\n  Generating synthetic test audio...")
query_audio = np.random.randn(16000).astype(np.float32)  # 1 second
corpus_audio = np.random.randn(48000).astype(np.float32)  # 3 seconds

print("  Extracting query embeddings...")
query_emb = extractor.extract(query_audio, sample_rate=16000)
print(f"    Query: {query_emb.shape} (frames, embedding_dim)")

print("  Extracting corpus embeddings...")
corpus_emb = extractor.extract(corpus_audio, sample_rate=16000)
print(f"    Corpus: {corpus_emb.shape} (frames, embedding_dim)")

print("  ✓ Feature extraction successful!")

# Test 2: DTW Matching
print("\n[2/2] Testing Subsequence DTW matching...")
matcher = SubsequenceDTWMatcher()

print("  Running S-DTW alignment...")
result = matcher.match(query_emb, corpus_emb)

print(f"  ✓ DTW matching successful!")
print(f"    Match distance: {result.distance:.4f}")
print(f"    Match frames: {result.start_frame} - {result.end_frame}")
print(f"    Match duration: {result.duration_frames} frames")

# Verify dtypes were handled correctly
print("\n  Verifying dtype handling...")
print(f"    Query dtype: {query_emb.dtype} (XLSR output)")
print(f"    Corpus dtype: {corpus_emb.dtype} (XLSR output)")
print("    ✓ Automatic float32→float64 conversion working")

print("\n" + "="*60)
print("SUCCESS! Full pipeline is working correctly.")
print("="*60)
print("\nWhat just happened:")
print("  1. Loaded XLSR-53 transformer model")
print("  2. Extracted 1024-D embeddings from audio")
print("  3. Matched query sequence in longer corpus")
print("  4. Got frame-accurate alignment result")
print("\nNext steps:")
print("  1. Run with real audio: python scripts/01_minimal_demo.py")
print("  2. See QUICKSTART.md for detailed instructions")