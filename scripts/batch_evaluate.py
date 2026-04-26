#!/usr/bin/env python3
"""
Batch QbE-STD Evaluation Script

Walks the input directory recursively, runs match_top_k for every query against
its corpus, and outputs a CSV ready for manual annotation.

Output CSV columns: language, query_file, query_length, corpus_file,
corpus_length, match_rank, match_distance, match_start, match_end. With a
metadata.csv ground-truth file present, two extra columns are added:
is_correct (same phrase) and is_same (same audio). TOP_K rows per query.

Usage:
    python scripts/batch_evaluate.py
    python scripts/batch_evaluate.py --model wavlm-base-plus --top-k 3 --window 30
"""

import sys
import csv
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    parser = argparse.ArgumentParser(description='QbE-STD Batch Evaluation')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory to walk for query/corpus pairs')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to write the CSV into; filename is auto-generated')
    parser.add_argument('--model', type=str, default='wavlm-base',
                        choices=list(MODEL_MAP.keys()),
                        help='Feature extraction model (default: wavlm-base)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top matches per query (default: 5)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run on (default: cpu)')
    parser.add_argument('--window', type=int, default=25,
                        help='Sakoe-Chiba window constraint for DTW '
                             '(default: 25). Pass a negative value to disable.')
    parser.add_argument('--tolerance', type=float, default=0.3,
                        help='Tolerance in seconds for matching predicted spans against '
                             'metadata.csv ground truth on both start and end (default: 0.3). '
                             'Only applied when metadata.csv is present.')
    return parser.parse_args()


def build_output_path(output_dir: Path, model_name: str, device: str,
                      window: int | None, top_k: int) -> Path:
    model_slug = model_name.split('/')[-1]
    window_part = f"window{window}" if window is not None else "windowNone"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return output_dir / f"top{top_k}_{model_slug}_{device}_{window_part}_{timestamp}.csv"


def load_metadata(metadata_path: Path) -> tuple[
    dict[str, list[tuple[float, float]]],
    dict[tuple[str, str], list[tuple[float, float]]],
]:
    corpus_to_matches: dict[str, list[tuple[float, float]]] = {}
    same_query_map: dict[tuple[str, str], list[tuple[float, float]]] = {}
    with open(metadata_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = row["corpus_file"].strip()
            start, end = float(row["match_start"]), float(row["match_end"])
            corpus_to_matches.setdefault(c, []).append((start, end))
            sq = row["same_query"].strip()
            if sq:
                same_query_map.setdefault((c, sq), []).append((start, end))
    return corpus_to_matches, same_query_map


def discover_jobs(input_dir: Path) -> list[dict]:
    """
    Walk input_dir recursively. For every directory named 'queries/', collect:
      - language: first path component after input_dir (uppercased)
      - corpus_file: single .wav in sibling 'corpus/' directory
      - query_files: all .wav files in 'queries/' directory

    Returns list of dicts with keys: language, corpus_file, query_file
    (one entry per query file).
    """
    jobs = []
    for queries_dir in sorted(input_dir.rglob("queries")):
        if not queries_dir.is_dir():
            continue

        rel = queries_dir.relative_to(input_dir)
        language = rel.parts[0].upper()

        corpus_dir = queries_dir.parent / "corpus"
        if not corpus_dir.is_dir():
            logger.warning(f"No corpus/ dir next to {queries_dir}, skipping")
            continue

        corpus_wavs = sorted(corpus_dir.glob("*.wav"))
        if len(corpus_wavs) != 1:
            logger.warning(f"Expected 1 corpus wav in {corpus_dir}, found {len(corpus_wavs)}, skipping")
            continue
        corpus_file = corpus_wavs[0]

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


def evaluate_job(
    job: dict,
    extractor: SSLSpeechExtractor,
    matcher: SubsequenceDTWMatcher,
    corpus_cache: dict[Path, tuple[np.ndarray, float]],
    top_k: int,
    metadata: dict,
    input_dir: Path,
    tolerance: float,
) -> list[dict]:
    language = job["language"]
    corpus_file = job["corpus_file"]
    query_file = job["query_file"]

    if corpus_file not in corpus_cache:
        corpus_audio, sr = load_audio(str(corpus_file))
        corpus_cache[corpus_file] = (
            extractor.extract(corpus_audio, sr),
            len(corpus_audio) / sr,
        )
    corpus_embeddings, corpus_length = corpus_cache[corpus_file]

    query_audio, sr = load_audio(str(query_file))
    query_embeddings = extractor.extract(query_audio, sr)
    query_length = len(query_audio) / sr

    corpus_norm = str(corpus_file.relative_to(input_dir))
    query_norm = str(query_file.relative_to(input_dir))
    metadata_present = metadata["present"]
    ref_matches = metadata["corpus_to_matches"].get(corpus_norm) if metadata_present else None
    same_refs = metadata["same_query_map"].get((corpus_norm, query_norm), []) if metadata_present else []

    matches = matcher.match_top_k(query_embeddings, corpus_embeddings, k=top_k)

    rows = []
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
                    abs(pred_start - ref_s) <= tolerance and abs(pred_end - ref_e) <= tolerance
                    for ref_s, ref_e in ref_matches
                )
                is_same_val = any(
                    abs(pred_start - ref_s) <= tolerance and abs(pred_end - ref_e) <= tolerance
                    for ref_s, ref_e in same_refs
                )
            else:
                is_correct_val = ""
                is_same_val = ""
            row["is_correct (same phrase)"] = is_correct_val
            row["is_same (same audio)"] = is_same_val
        rows.append(row)

    # Pad with empty rows so every query has exactly top_k rows
    for rank in range(len(matches) + 1, top_k + 1):
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

    return rows


def write_results_csv(rows: list[dict], output_path: Path, metadata_present: bool) -> None:
    fieldnames = [
        "language", "query_file", "query_length", "corpus_file", "corpus_length",
        "match_rank", "match_distance", "match_start", "match_end",
    ]
    if metadata_present:
        fieldnames += ["is_correct (same phrase)", "is_same (same audio)"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_name = MODEL_MAP[args.model]
    window = args.window if args.window is None or args.window >= 0 else None

    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(output_dir, model_name, args.device, window, args.top_k)

    metadata_path = input_dir / "metadata.csv"
    metadata_present = metadata_path.is_file()
    metadata = {"present": metadata_present, "corpus_to_matches": {}, "same_query_map": {}}
    if metadata_present:
        logger.info(f"Loading ground-truth metadata from {metadata_path}")
        metadata["corpus_to_matches"], metadata["same_query_map"] = load_metadata(metadata_path)
        logger.info(
            f"Metadata loaded: {len(metadata['corpus_to_matches'])} corpus files, "
            f"{len(metadata['same_query_map'])} same-audio pairs"
        )

    logger.info(f"Scanning {input_dir} for query/corpus pairs...")
    jobs = discover_jobs(input_dir)
    logger.info(f"Found {len(jobs)} query files across all topics")

    if not jobs:
        logger.error("No jobs found. Check input directory structure.")
        return 1

    logger.info(f"Loading model: {model_name} on {args.device}")
    extractor = SSLSpeechExtractor(
        model_name=model_name,
        device=args.device,
        use_half_precision=True,
    )
    matcher = SubsequenceDTWMatcher(window=window)

    corpus_cache: dict[Path, tuple[np.ndarray, float]] = {}
    rows: list[dict] = []
    for job in tqdm(jobs, desc="Evaluating queries"):
        rows.extend(evaluate_job(job, extractor, matcher, corpus_cache, args.top_k, metadata, input_dir, args.tolerance))

    write_results_csv(rows, output_path, metadata_present)

    logger.info(f"Wrote {len(rows)} rows to {output_path}")
    logger.info(f"Expected rows: {len(jobs)} queries × {args.top_k} = {len(jobs) * args.top_k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
