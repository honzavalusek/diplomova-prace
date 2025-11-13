#!/usr/bin/env python3
"""
Minimal End-to-End Demo: Query-by-Example Spoken Term Detection

This script demonstrates the core acoustic retrieval pipeline WITHOUT LSH indexing.
Use this to validate the approach on 2-3 small audio files before scaling up.

Usage:
    python scripts/01_minimal_demo.py --query data/raw_audio/queries/query1.wav \
                                       --corpus data/raw_audio/corpus/*.wav
"""

import sys
import argparse
import logging
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import Wav2Vec2Extractor, load_audio, frames_to_seconds
from src.features.speaker_normalization import apply_normalization
from src.matching import SubsequenceDTWMatcher, MatchResult

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Minimal QbE-STD Demo')
    parser.add_argument('--query', type=str, required=True,
                        help='Path to query audio file')
    parser.add_argument('--corpus', type=str, nargs='+', required=True,
                        help='Paths to corpus audio files')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run on (default: cpu)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top matches to find per corpus file (default: 3)')
    parser.add_argument('--model', type=str, default='xlsr-53',
                        choices=['xlsr-53', 'xls-r-300m', 'xls-r-1b', 'xls-r-2b', 'czech', 'czech2'],
                        help='Feature extraction model (default: xlsr-53)')
    parser.add_argument('--normalize', type=str, default='none',
                        choices=['none', 'mvn', 'cmn'],
                        help='Speaker normalization method (mvn=mean-variance, cmn=cepstral-mean, none=disabled)')

    args = parser.parse_args()

    # Verify files exist
    query_path = Path(args.query)
    if not query_path.exists():
        logger.error(f"Query file not found: {query_path}")
        return 1

    corpus_paths = [Path(p) for p in args.corpus]
    for path in corpus_paths:
        if not path.exists():
            logger.error(f"Corpus file not found: {path}")
            return 1

    logger.info("=" * 60)
    logger.info("MINIMAL QbE-STD DEMO (No LSH)")
    logger.info("=" * 60)

    # Step 1: Initialize feature extractor
    model_map = {
        'xlsr-53': 'facebook/wav2vec2-large-xlsr-53',
        'xls-r-300m': 'facebook/wav2vec2-xls-r-300m',
        'xls-r-1b': 'facebook/wav2vec2-xls-r-1b',
        'xls-r-2b': 'facebook/wav2vec2-xls-r-2b',
        'czech': 'arampacha/wav2vec2-large-xlsr-czech',
        'czech2': 'fav-kky/wav2vec2-base-cs-80k-ClTRUS',
    }

    model_name = model_map[args.model]
    logger.info(f"\n[1/4] Initializing {args.model} feature extractor...")
    logger.info(f"  Model: {model_name}")

    extractor = Wav2Vec2Extractor(model_name=model_name, device=args.device, use_last_x_layers=10, use_half_precision=True)
    logger.info(f"  Embedding dimension: {extractor.embedding_dim}")

    # Step 2: Extract query embeddings
    logger.info(f"\n[2/4] Extracting query features: {query_path.name}")
    start_time = time.time()
    query_audio, sr = load_audio(str(query_path))
    query_embeddings = extractor.extract(query_audio, sr)

    # Apply speaker normalization if requested
    if args.normalize != 'none':
        logger.info(f"  Applying {args.normalize} normalization...")
        query_embeddings = apply_normalization(query_embeddings, method=args.normalize)

    query_time = time.time() - start_time
    logger.info(f"  Query shape: {query_embeddings.shape}")
    logger.info(f"  Extraction time: {query_time:.2f}s")

    # Step 3: Extract corpus embeddings and search
    logger.info(f"\n[3/4] Processing {len(corpus_paths)} corpus file(s)...")
    matcher = SubsequenceDTWMatcher()

    # Store results per file: {file_path: [list of matches]}
    results_by_file = {}

    for corpus_path in corpus_paths:
        logger.info(f"\n  Processing: {corpus_path.name}")

        # Extract embeddings
        start_time = time.time()
        corpus_audio, sr = load_audio(str(corpus_path))
        corpus_embeddings = extractor.extract(corpus_audio, sr)

        # Apply same normalization to corpus
        if args.normalize != 'none':
            corpus_embeddings = apply_normalization(corpus_embeddings, method=args.normalize)

        extract_time = time.time() - start_time

        logger.info(f"    Corpus shape: {corpus_embeddings.shape}")
        logger.info(f"    Extraction time: {extract_time:.2f}s")

        # Perform S-DTW matching - get top k matches in this file
        logger.info(f"    Running S-DTW matching (top {args.top_k})...")
        start_time = time.time()
        top_matches = matcher.match_top_k(query_embeddings, corpus_embeddings, k=args.top_k)
        dtw_time = time.time() - start_time

        logger.info(f"    DTW time: {dtw_time:.2f}s")
        logger.info(f"    Found {len(top_matches)} match(es)")

        for i, match in enumerate(top_matches, 1):
            logger.info(f"      Match {i}: distance={match.distance:.4f}, frames={match.start_frame}-{match.end_frame}")

        # Store results for this file
        results_by_file[corpus_path] = top_matches

    # Step 4: Report results
    logger.info("\n" + "=" * 60)
    logger.info(f"[4/4] FINAL RESULTS - TOP {args.top_k} MATCHES PER FILE")
    logger.info("=" * 60)

    if results_by_file:
        for file_path, matches in results_by_file.items():
            logger.info(f"\n{file_path.name}:")

            if matches:
                for rank, match in enumerate(matches, start=1):
                    # Convert frames to timestamps
                    start_sec = frames_to_seconds(
                        match.start_frame,
                        sample_rate=16000,
                        hop_length=extractor.hop_length
                    )
                    end_sec = frames_to_seconds(
                        match.end_frame,
                        sample_rate=16000,
                        hop_length=extractor.hop_length
                    )

                    logger.info(f"  #{rank} - Distance: {match.distance:.4f}")
                    logger.info(f"       Time: {start_sec:.2f}s - {end_sec:.2f}s (duration: {end_sec - start_sec:.2f}s)")
                    logger.info(f"       Frames: {match.start_frame} - {match.end_frame}")
            else:
                logger.info("  No matches found")

        logger.info("\n" + "=" * 60)
        logger.info("Demo completed successfully!")
        logger.info("=" * 60)
        return 0
    else:
        logger.warning("\nNo matches found!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
