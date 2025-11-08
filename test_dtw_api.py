#!/usr/bin/env python3
"""Test dtaidistance API behavior"""

import numpy as np
from dtaidistance.subsequence.dtw import subsequence_alignment

print("Testing dtaidistance subsequence_alignment API...")

# Create synthetic test: query embedded in longer sequence
query = np.random.randn(50, 10).astype(np.float64)

# Create corpus: [noise, exact query copy, more noise]
noise_before = np.random.randn(100, 10).astype(np.float64)
noise_after = np.random.randn(100, 10).astype(np.float64)
corpus = np.vstack([noise_before, query, noise_after])

print(f"Query shape: {query.shape}")
print(f"Corpus shape: {corpus.shape}")
print(f"Query should appear at frame 100-150")

# Test 1: Best match
print("\n[1] Testing best_match()...")
sa = subsequence_alignment(query, corpus)
best = sa.best_match()

print(f"Best match:")
print(f"  Distance: {best.distance:.4f}")
print(f"  Segment: {best.segment}")
print(f"  Expected: (100, 150)")

if best.segment[0] >= 95 and best.segment[0] <= 105:
    print("  ✓ Correct location!")
else:
    print(f"  ❌ Wrong location! Off by {abs(best.segment[0] - 100)} frames")

# Test 2: kbest matches
print("\n[2] Testing kbest_matches(k=3)...")
try:
    kbest = sa.kbest_matches(k=3)
    print(f"Found {len(kbest)} matches:")
    for i, match in enumerate(kbest, 1):
        print(f"  #{i}: distance={match.distance:.4f}, segment={match.segment}")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    print("  kbest_matches might not be available!")
