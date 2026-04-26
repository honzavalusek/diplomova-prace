#!/usr/bin/env python3
"""
Single-query QbE-STD search: match one query against one or more corpus files.

Usage:
    python scripts/search.py --query data/raw_audio/queries/query1.wav \
                             --corpus data/raw_audio/corpus/*.wav
"""

import sys
import argparse
import logging
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import SSLSpeechExtractor, load_audio, frames_to_seconds
from src.matching import SubsequenceDTWMatcher, MatchResult

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

MODEL_MAP = {
    'xlsr-53': 'facebook/wav2vec2-large-xlsr-53',
    'xls-r-300m': 'facebook/wav2vec2-xls-r-300m',
    'xls-r-1b': 'facebook/wav2vec2-xls-r-1b',
    'xls-r-2b': 'facebook/wav2vec2-xls-r-2b',
    'wavlm-base': 'microsoft/wavlm-base',
    'wavlm-base-plus': 'microsoft/wavlm-base-plus',
    'wavlm-large': 'microsoft/wavlm-large',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='QbE-STD Single-Query Search')
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
                        choices=list(MODEL_MAP.keys()),
                        help='Feature extraction model (default: xlsr-53)')
    parser.add_argument('--window', type=int, default=None,
                        help='Sakoe-Chiba window constraint for DTW (limits temporal deviation). '
                             'Typical values: 10-50 frames for speech. If not set, no constraint is used.')
    parser.add_argument('--layer-min', type=int, default=None,
                        help='Minimum layer index (0-based). If set without --layer-max, averages from this layer to the last.')
    parser.add_argument('--layer-max', type=int, default=None,
                        help='Maximum layer index (0-based). If set without --layer-min, averages from layer 0 to this layer.')
    return parser.parse_args()


def validate_paths(query: str, corpus: list[str]) -> tuple[Path, list[Path]] | None:
    query_path = Path(query)
    if not query_path.exists():
        logger.error(f"Query file not found: {query_path}")
        return None

    corpus_paths = [Path(p) for p in corpus]
    for path in corpus_paths:
        if not path.exists():
            logger.error(f"Corpus file not found: {path}")
            return None

    return query_path, corpus_paths


def search_corpus(
    query_embeddings,
    corpus_path: Path,
    matcher: SubsequenceDTWMatcher,
    extractor: SSLSpeechExtractor,
    top_k: int,
) -> list[MatchResult]:
    logger.info(f"\n  Processing: {corpus_path.name}")

    start_time = time.time()
    corpus_audio, sr = load_audio(str(corpus_path))
    corpus_embeddings = extractor.extract(corpus_audio, sr)
    extract_time = time.time() - start_time

    logger.info(f"    Corpus shape: {corpus_embeddings.shape}")
    logger.info(f"    Extraction time: {extract_time:.2f}s")

    logger.info(f"    Running S-DTW matching (top {top_k})...")
    start_time = time.time()
    top_matches = matcher.match_top_k(query_embeddings, corpus_embeddings, k=top_k)
    dtw_time = time.time() - start_time

    logger.info(f"    DTW time: {dtw_time:.2f}s")
    logger.info(f"    Found {len(top_matches)} match(es)")

    for i, match in enumerate(top_matches, 1):
        logger.info(f"      Match {i}: distance={match.distance:.4f}, frames={match.start_frame}-{match.end_frame}")

    return top_matches


def report_results(
    results_by_file: dict[Path, list[MatchResult]],
    extractor: SSLSpeechExtractor,
    top_k: int,
) -> None:
    logger.info("\n" + "=" * 60)
    logger.info(f"[4/4] FINAL RESULTS - TOP {top_k} MATCHES PER FILE")
    logger.info("=" * 60)

    for file_path, matches in results_by_file.items():
        logger.info(f"\n{file_path.name}:")

        if matches:
            for rank, match in enumerate(matches, start=1):
                start_sec = frames_to_seconds(
                    match.start_frame,
                    sample_rate=16000,
                    hop_length=extractor.hop_length,
                )
                end_sec = frames_to_seconds(
                    match.end_frame,
                    sample_rate=16000,
                    hop_length=extractor.hop_length,
                )

                logger.info(f"  #{rank} - Distance: {match.distance:.4f}")
                logger.info(f"       Time: {start_sec:.2f}s - {end_sec:.2f}s (duration: {end_sec - start_sec:.2f}s)")
                logger.info(f"       Frames: {match.start_frame} - {match.end_frame}")
        else:
            logger.info("  No matches found")

    logger.info("\n" + "=" * 60)
    logger.info("Demo completed successfully!")
    logger.info("=" * 60)


def main() -> int:
    args = parse_args()

    paths = validate_paths(args.query, args.corpus)
    if paths is None:
        return 1
    query_path, corpus_paths = paths

    logger.info("=" * 60)
    logger.info("QbE-STD SEARCH")
    logger.info("=" * 60)

    model_name = MODEL_MAP[args.model]
    logger.info(f"\n[1/4] Initializing {args.model} feature extractor...")
    logger.info(f"  Model: {model_name}")

    extractor = SSLSpeechExtractor(
        model_name=model_name,
        device=args.device,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
        use_half_precision=True,
    )
    logger.info(f"  Embedding dimension: {extractor.embedding_dim}")

    logger.info(f"\n[2/4] Extracting query features: {query_path.name}")
    start_time = time.time()
    query_audio, sr = load_audio(str(query_path))
    query_embeddings = extractor.extract(query_audio, sr)
    query_time = time.time() - start_time
    logger.info(f"  Query shape: {query_embeddings.shape}")
    logger.info(f"  Extraction time: {query_time:.2f}s")

    logger.info(f"\n[3/4] Processing {len(corpus_paths)} corpus file(s)...")
    matcher = SubsequenceDTWMatcher(window=args.window)

    results_by_file: dict[Path, list[MatchResult]] = {}
    for corpus_path in corpus_paths:
        results_by_file[corpus_path] = search_corpus(
            query_embeddings, corpus_path, matcher, extractor, args.top_k
        )

    report_results(results_by_file, extractor, args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
