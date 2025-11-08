#!/usr/bin/env python3
"""Test kbest_matches behavior in detail"""

import numpy as np
from dtaidistance.subsequence.dtw import subsequence_alignment

# Create test case with 3 embedded queries
query = np.random.randn(30, 10).astype(np.float64)

# Corpus: [noise, query copy 1, noise, query copy 2, noise, query copy 3, noise]
parts = [
    np.random.randn(50, 10).astype(np.float64),  # noise
    query.copy(),                                 # match 1 at ~50
    np.random.randn(50, 10).astype(np.float64),  # noise
    query.copy(),                                 # match 2 at ~130
    np.random.randn(50, 10).astype(np.float64),  # noise
    query.copy(),                                 # match 3 at ~210
    np.random.randn(50, 10).astype(np.float64),  # noise
]

corpus = np.vstack(parts)

print(f"Query shape: {query.shape}")
print(f"Corpus shape: {corpus.shape}")
print(f"Expected matches at: ~50, ~130, ~210")

sa = subsequence_alignment(query, corpus)

print("\n[1] Best match:")
best = sa.best_match()
print(f"  {best.segment}, distance={best.distance:.4f}")

print("\n[2] Iterating kbest_matches(k=5):")
kbest_gen = sa.kbest_matches(k=5)
matches = []

for i, match in enumerate(kbest_gen):
    matches.append(match)
    print(f"  #{i+1}: segment={match.segment}, distance={match.distance:.4f}")
    if i >= 4:  # Safety: only take first 5
        break

print(f"\nTotal matches collected: {len(matches)}")

print("\n[3] Testing overlap filtering logic:")
results = []
used_ranges = []
min_distance_frames = 10

for match in matches:
    start_frame = match.segment[0]
    end_frame = match.segment[1]

    # Check overlap
    overlaps = False
    for used_start, used_end in used_ranges:
        if not (end_frame + min_distance_frames < used_start or
                start_frame - min_distance_frames > used_end):
            overlaps = True
            break

    if not overlaps:
        results.append(match)
        used_ranges.append((start_frame, end_frame))
        print(f"  ✓ Kept: {match.segment}, distance={match.distance:.4f}")
    else:
        print(f"  ✗ Filtered (overlap): {match.segment}")

print(f"\nFinal non-overlapping matches: {len(results)}")
