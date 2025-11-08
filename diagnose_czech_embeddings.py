#!/usr/bin/env python3
"""Diagnose why Czech speech matching fails"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.features import XLSR53FeatureExtractor, load_audio
from src.matching import SubsequenceDTWMatcher

print("="*60)
print("DIAGNOSING CZECH SPEECH MATCHING")
print("="*60)

# Your files
query_path = "data/raw_audio/queries/extract.wav"  # "Budeme muset redukovat"
corpus_path = "data/raw_audio/corpus/long.wav"

print(f"\nQuery: 'Budeme muset redukovat' at {query_path}")
print(f"Corpus: {corpus_path}")

# Extract embeddings
print("\n[1] Extracting embeddings...")
extractor = XLSR53FeatureExtractor(device='cpu')

query_audio, sr = load_audio(query_path)
corpus_audio, sr = load_audio(corpus_path)

query_emb = extractor.extract(query_audio, sr)
corpus_emb = extractor.extract(corpus_audio, sr)

print(f"Query: {query_emb.shape}")
print(f"Corpus: {corpus_emb.shape}")

# Compare specific segments
wrong_start_sec = 49.56  # "My dva. Jo tak"
correct_start_sec = 10.0  # Should be "Budeme muset redukovat"

wrong_start_frame = int(wrong_start_sec * sr / extractor.hop_length)
wrong_end_frame = int(51.32 * sr / extractor.hop_length)

correct_start_frame = int(correct_start_sec * sr / extractor.hop_length)
correct_end_frame = int(12.0 * sr / extractor.hop_length)

print(f"\n[2] Comparing segments:")
print(f"Wrong match (49.56s): frames {wrong_start_frame}-{wrong_end_frame}")
print(f"Correct location (10s): frames {correct_start_frame}-{correct_end_frame}")

# Compute DTW distances
matcher = SubsequenceDTWMatcher()

print("\n[3] Computing DTW distances (may take 30s)...")

# Create local windows around each location
wrong_window = corpus_emb[max(0, wrong_start_frame-100):wrong_end_frame+100]
correct_window = corpus_emb[max(0, correct_start_frame-100):correct_end_frame+100]

result_wrong = matcher.match(query_emb, wrong_window)
result_correct = matcher.match(query_emb, correct_window)
result_self = matcher.match(query_emb, query_emb)

print(f"\nDistance to WRONG match (My dva. Jo tak): {result_wrong.distance:.4f}")
print(f"Distance to CORRECT location (10s): {result_correct.distance:.4f}")
print(f"Distance to SELF: {result_self.distance:.4f}")

# Analysis
print("\n[4] Analysis:")
if result_wrong.distance < result_correct.distance:
    print("❌ CRITICAL: Wrong match has LOWER distance!")
    print(f"   Model thinks 'My dva. Jo tak' is MORE similar than the actual match")
    print(f"   Difference: {result_correct.distance - result_wrong.distance:.4f}")
else:
    print("✓ Correct match has lower distance")

# Check embedding variance
print("\n[5] Embedding discriminability:")
query_mean = query_emb.mean(axis=0)
wrong_mean = corpus_emb[wrong_start_frame:wrong_end_frame].mean(axis=0)
correct_mean = corpus_emb[correct_start_frame:correct_end_frame].mean(axis=0)

# Cosine similarity
from src.matching.distance_metrics import cosine_distance
cos_wrong = cosine_distance(query_mean, wrong_mean)
cos_correct = cosine_distance(query_mean, correct_mean)

print(f"Cosine distance to wrong: {cos_wrong:.4f}")
print(f"Cosine distance to correct: {cos_correct:.4f}")

# Euclidean distance
euc_wrong = np.linalg.norm(query_mean - wrong_mean)
euc_correct = np.linalg.norm(query_mean - correct_mean)

print(f"Euclidean distance to wrong: {euc_wrong:.4f}")
print(f"Euclidean distance to correct: {euc_correct:.4f}")

print("\n[6] Full corpus scan (finding all good matches)...")
print("Running full S-DTW on entire corpus...")
print("(This will take several minutes for long files)")

result_full = matcher.match(query_emb, corpus_emb)
matches_top10 = matcher.match_top_k(query_emb, corpus_emb, k=10)

print(f"\nTop 10 matches:")
for i, m in enumerate(matches_top10, 1):
    time_s = m.start_frame * extractor.hop_length / sr
    time_e = m.end_frame * extractor.hop_length / sr
    is_correct = abs(time_s - 10.0) < 2.0  # Within 2 seconds of expected
    marker = "✓ CORRECT!" if is_correct else ""
    print(f"  #{i}: {time_s:.2f}s-{time_e:.2f}s, distance={m.distance:.4f} {marker}")

print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)

if result_wrong.distance < result_correct.distance:
    print("\nThe XLSR-53 model is NOT capturing Czech phonetics well enough.")
    print("\nTry these solutions (in order):")
    print("\n1. Use Czech-specific fine-tuned model:")
    print("   Search Hugging Face for: 'wav2vec2 czech'")
    print("   Example: comodoro/wav2vec2-xls-r-300m-cs-250")
    print()
    print("2. Try larger XLS-R model:")
    print("   python scripts/01_minimal_demo.py --model xls-r-1b --device cuda \\")
    print("       --query extract.wav --corpus long.wav")
    print()
    print("3. Fine-tune XLSR-53 on Czech Common Voice dataset")
    print("   (Advanced - requires training)")
else:
    print("\nEmbeddings CAN distinguish the phrases.")
    print("The issue is likely with the DTW search space.")
    print("Check if the correct match appears in the top 10 above.")