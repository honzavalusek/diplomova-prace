#!/usr/bin/env python3
"""Extract query with EXACT timing from DTW result"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import soundfile as sf
from src.features import load_audio, CzechWav2Vec2Extractor, frames_to_seconds
from src.matching import SubsequenceDTWMatcher

print("Finding EXACT boundaries of the phrase...")

corpus_file = "data/raw_audio/corpus/kancl_1__9_10.wav"
approximate_start = 17.86
window_duration = 2.0  # Search in 2-second window

# Load corpus
corpus_audio, sr = load_audio(corpus_file)

# Extract window
window_start_sample = int((approximate_start - 0.5) * sr)
window_end_sample = int((approximate_start + window_duration) * sr)
window_audio = corpus_audio[window_start_sample:window_end_sample]

# Get embeddings for the window
extractor = CzechWav2Vec2Extractor(device='cpu')
window_emb = extractor.extract(window_audio, sr)

print(f"\nSearching for phrase boundaries...")
# Use first 0.5s as query
query_frames = int(0.5 * sr / extractor.hop_length)
query_emb = window_emb[:query_frames]

# Find in window
matcher = SubsequenceDTWMatcher()
result = matcher.match(query_emb, window_emb)

# Calculate exact timing
exact_start_frame = result.start_frame
exact_end_frame = result.end_frame

exact_start_sec = approximate_start - 0.5 + frames_to_seconds(exact_start_frame, sr, extractor.hop_length)
exact_end_sec = approximate_start - 0.5 + frames_to_seconds(exact_end_frame, sr, extractor.hop_length)
exact_duration = exact_end_sec - exact_start_sec

print(f"\nEXACT phrase boundaries:")
print(f"  Start: {exact_start_sec:.2f}s")
print(f"  End: {exact_end_sec:.2f}s")
print(f"  Duration: {exact_duration:.2f}s")
print(f"  Frames: {exact_end_frame - exact_start_frame}")

# Extract with exact timing
exact_start_sample = int(exact_start_sec * sr)
exact_end_sample = int(exact_end_sec * sr)
exact_query = corpus_audio[exact_start_sample:exact_end_sample]

output_file = "data/raw_audio/queries/extract_exact.wav"
sf.write(output_file, exact_query, sr)

print(f"\nSaved: {output_file}")
print("\nNow test:")
print("python scripts/01_minimal_demo.py \\")
print("    --query data/raw_audio/queries/extract_exact.wav \\")
print("    --corpus data/raw_audio/corpus/kancl_1__9_10.wav \\")
print("    --model czech")
print("\nExpected: Distance should now be < 5")
