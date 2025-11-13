#!/usr/bin/env python3
"""Test Sakoe-Chiba window constraint in SubsequenceDTWMatcher"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.matching import SubsequenceDTWMatcher

print("=" * 60)
print("Testing Sakoe-Chiba Window Constraint")
print("=" * 60)

# Create synthetic test data
np.random.seed(42)
query = np.random.randn(30, 128).astype(np.float64)
reference = np.random.randn(200, 128).astype(np.float64)

print(f"\nQuery shape: {query.shape}")
print(f"Reference shape: {reference.shape}")

# Test 1: Without window constraint
print("\n[1] Testing WITHOUT window constraint...")
matcher_no_window = SubsequenceDTWMatcher(use_c=True, window=None)
result_no_window = matcher_no_window.match(query, reference)
print(f"    Distance: {result_no_window.distance:.4f}")
print(f"    Match: frames {result_no_window.start_frame}-{result_no_window.end_frame}")

# Test 2: With small window (more constrained)
print("\n[2] Testing WITH window=10 (tight constraint)...")
matcher_small_window = SubsequenceDTWMatcher(use_c=True, window=10)
result_small_window = matcher_small_window.match(query, reference)
print(f"    Distance: {result_small_window.distance:.4f}")
print(f"    Match: frames {result_small_window.start_frame}-{result_small_window.end_frame}")

# Test 3: With larger window
print("\n[3] Testing WITH window=50 (loose constraint)...")
matcher_large_window = SubsequenceDTWMatcher(use_c=True, window=50)
result_large_window = matcher_large_window.match(query, reference)
print(f"    Distance: {result_large_window.distance:.4f}")
print(f"    Match: frames {result_large_window.start_frame}-{result_large_window.end_frame}")

# Test 4: Test match_top_k with window
print("\n[4] Testing match_top_k with window=25...")
matcher_topk = SubsequenceDTWMatcher(use_c=True, window=25)
results_topk = matcher_topk.match_top_k(query, reference, k=3)
print(f"    Found {len(results_topk)} matches:")
for i, match in enumerate(results_topk, 1):
    print(f"      #{i}: distance={match.distance:.4f}, frames={match.start_frame}-{match.end_frame}")

# Test 5: Override window in method call
print("\n[5] Testing method-level window override...")
matcher_default = SubsequenceDTWMatcher(use_c=True, window=10)
result_override = matcher_default.match(query, reference, window=30)
print(f"    Instance window: 10, method window: 30")
print(f"    Distance: {result_override.distance:.4f}")
print(f"    Match: frames {result_override.start_frame}-{result_override.end_frame}")

print("\n" + "=" * 60)
print("✓ All window constraint tests completed!")
print("=" * 60)
print("\nNote: Smaller windows typically result in:")
print("  - Faster computation")
print("  - More constrained alignments (less flexible warping)")
print("  - Potentially higher distances (more restrictive matching)")
