#!/usr/bin/env python3
"""Extract query using Python (no FFmpeg) to avoid re-encoding"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import soundfile as sf
from src.features import load_audio

print("Extracting query segment using Python (no FFmpeg)...")

# Source file and timing
corpus_file = "data/raw_audio/corpus/kancl_1__9_10.wav"
start_sec = 17.86
duration_sec = 0.88

# Load full corpus
print(f"Loading: {corpus_file}")
corpus_audio, sr = load_audio(corpus_file)

# Extract segment
start_sample = int(start_sec * sr)
end_sample = int((start_sec + duration_sec) * sr)
query_segment = corpus_audio[start_sample:end_sample]

# Save
output_file = "data/raw_audio/queries/extract_python.wav"
sf.write(output_file, query_segment, sr)

print(f"Saved: {output_file}")
print(f"Duration: {len(query_segment)/sr:.2f}s")
print(f"Sample rate: {sr} Hz")
print("\nNow run:")
print("python scripts/01_minimal_demo.py \\")
print("    --query data/raw_audio/queries/extract_python.wav \\")
print("    --corpus data/raw_audio/corpus/kancl_1__9_10.wav \\")
print("    --model czech \\")
print("    --normalize none")
print("\nExpected: First match should have distance < 5")
