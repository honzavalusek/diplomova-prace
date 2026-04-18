#!/usr/bin/env python3
"""
Batch QbE-STD Evaluation Script

Walks data/raw_audio/ recursively, runs match_top_k for every query against
its corpus, and outputs a CSV ready for manual annotation.

Output CSV columns: language, query_file, corpus_file, match_rank, match_distance, match_start, match_end
5 rows per query (ranks 1–5).

Usage:
    python scripts/02_evaluate.py
"""

import sys
import csv
import logging
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import SSLSpeechExtractor, load_audio, frames_to_seconds
from src.matching import SubsequenceDTWMatcher

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw_audio"
MODEL_NAME = "microsoft/wavlm-base"
TOP_K = 5
DEVICE = "cpu"
WINDOW = 25
OUTPUT_PATH = PROJECT_ROOT / "results" / f"top{TOP_K}_{MODEL_NAME.split('/')[-1]}_{DEVICE}_window{WINDOW}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# ---------------------


def discover_jobs(data_dir: Path) -> list[dict]:
    """
    Walk data_dir recursively. For every directory named 'queries/', collect:
      - language: second path component after data_dir (uppercased)
      - corpus_file: single .wav in sibling 'corpus/' directory
      - query_files: all .wav files in 'queries/' directory

    Returns list of dicts with keys: language, corpus_file, query_file
    (one entry per query file).
    """
    jobs = []
    for queries_dir in sorted(data_dir.rglob("queries")):
        if not queries_dir.is_dir():
            continue

        # Language = second path component after data_dir
        rel = queries_dir.relative_to(data_dir)
        language = rel.parts[0].upper()

        # Corpus = single .wav in sibling corpus/
        corpus_dir = queries_dir.parent / "corpus"
        if not corpus_dir.is_dir():
            logger.warning(f"No corpus/ dir next to {queries_dir}, skipping")
            continue

        corpus_wavs = sorted(corpus_dir.glob("*.wav"))
        if len(corpus_wavs) != 1:
            logger.warning(f"Expected 1 corpus wav in {corpus_dir}, found {len(corpus_wavs)}, skipping")
            continue
        corpus_file = corpus_wavs[0]

        # Queries = all .wav files in queries/
        query_files = sorted(queries_dir.glob("*.wav"))
        if not query_files:
            logger.warning(f"No query wav files in {queries_dir}, skipping")
            continue

        for query_file in query_files:
            jobs.append({
                "language": language,
                "corpus_file": corpus_file,
                "query_file": query_file,
            })

    return jobs


def main():
    data_dir = DATA_DIR
    output_path = OUTPUT_PATH

    if not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata.csv if present
    metadata_path = data_dir / "metadata.csv"
    metadata_present = metadata_path.is_file()
    corpus_to_matches: dict[str, list[tuple[float, float]]] = {}  # corpus_norm → [(start, end)]
    same_query_map: dict[tuple[str, str], list[tuple[float, float]]] = {}  # (corpus, query) → [(start, end)]

    if metadata_present:
        logger.info(f"Loading ground-truth metadata from {metadata_path}")
        with open(metadata_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c = row["corpus_file"].strip()
                start, end = float(row["match_start"]), float(row["match_end"])
                corpus_to_matches.setdefault(c, []).append((start, end))
                sq = row["same_query"].strip()
                if sq:
                    same_query_map.setdefault((c, sq), []).append((start, end))
        logger.info(f"Metadata loaded: {len(corpus_to_matches)} corpus files, {len(same_query_map)} same-audio pairs")

    # Discover all (query, corpus) pairs
    logger.info(f"Scanning {data_dir} for query/corpus pairs...")
    jobs = discover_jobs(data_dir)
    logger.info(f"Found {len(jobs)} query files across all topics")

    if not jobs:
        logger.error("No jobs found. Check data directory structure.")
        return 1

    # Initialize model
    logger.info(f"Loading model: {MODEL_NAME} on {DEVICE}")
    extractor = SSLSpeechExtractor(
        model_name=MODEL_NAME,
        device=DEVICE,
        use_half_precision=True
    )
    matcher = SubsequenceDTWMatcher(window=WINDOW)

    # Cache corpus embeddings to avoid re-extracting the same corpus
    import numpy as np
    corpus_cache: dict[Path, tuple[np.ndarray, float]] = {}

    rows = []

    TOL = 0.3

    for job in tqdm(jobs, desc="Evaluating queries"):
        language = job["language"]
        corpus_file = job["corpus_file"]
        query_file = job["query_file"]

        # Extract corpus embeddings (cached)
        if corpus_file not in corpus_cache:
            corpus_audio, sr = load_audio(str(corpus_file))
            corpus_cache[corpus_file] = (extractor.extract(corpus_audio, sr), len(corpus_audio) / sr)
        corpus_embeddings, corpus_length = corpus_cache[corpus_file]

        # Extract query embeddings
        query_audio, sr = load_audio(str(query_file))
        query_embeddings = extractor.extract(query_audio, sr)
        query_length = len(query_audio) / sr

        # Metadata-based annotation (paths relative to data/raw_audio/)
        corpus_norm = str(corpus_file.relative_to(data_dir))
        query_norm = str(query_file.relative_to(data_dir))
        ref_matches = corpus_to_matches.get(corpus_norm) if metadata_present else None
        same_refs = same_query_map.get((corpus_norm, query_norm), []) if metadata_present else []

        # Run top-k matching
        matches = matcher.match_top_k(query_embeddings, corpus_embeddings, k=TOP_K)

        for rank, match in enumerate(matches, start=1):
            pred_start = frames_to_seconds(match.start_frame, sample_rate=16000, hop_length=extractor.hop_length)
            pred_end = frames_to_seconds(match.end_frame, sample_rate=16000, hop_length=extractor.hop_length)
            row = {
                "language": language,
                "query_file": str(query_file.relative_to(PROJECT_ROOT.parent)),
                "query_length": query_length,
                "corpus_file": str(corpus_file.relative_to(PROJECT_ROOT.parent)),
                "corpus_length": corpus_length,
                "match_rank": rank,
                "match_distance": match.distance,
                "match_start": pred_start,
                "match_end": pred_end,
            }
            if metadata_present:
                if ref_matches is not None:
                    is_correct_val = any(
                        abs(pred_start - ref_s) <= TOL and abs(pred_end - ref_e) <= TOL
                        for ref_s, ref_e in ref_matches
                    )
                    is_same_val = any(
                        abs(pred_start - ref_s) <= TOL and abs(pred_end - ref_e) <= TOL
                        for ref_s, ref_e in same_refs
                    )
                else:
                    is_correct_val = ""
                    is_same_val = ""
                row["is_correct (same phrase)"] = is_correct_val
                row["is_same (same audio)"] = is_same_val
            rows.append(row)

        # If fewer than TOP_K matches returned, pad with empty rows so every
        # query always has exactly TOP_K rows (distances left empty)
        for rank in range(len(matches) + 1, TOP_K + 1):
            row = {
                "language": language,
                "query_file": str(query_file.relative_to(PROJECT_ROOT.parent)),
                "query_length": query_length,
                "corpus_file": str(corpus_file.relative_to(PROJECT_ROOT.parent)),
                "corpus_length": corpus_length,
                "match_rank": rank,
                "match_distance": "",
                "match_start": "",
                "match_end": "",
            }
            if metadata_present:
                row["is_correct (same phrase)"] = ""
                row["is_same (same audio)"] = ""
            rows.append(row)

    # Write CSV
    fieldnames = ["language", "query_file", "query_length", "corpus_file", "corpus_length", "match_rank", "match_distance", "match_start", "match_end"]
    if metadata_present:
        fieldnames += ["is_correct (same phrase)", "is_same (same audio)"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} rows to {output_path}")
    logger.info(f"Expected rows: {len(jobs)} queries × {TOP_K} = {len(jobs) * TOP_K}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
