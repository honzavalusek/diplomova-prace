"""Subsequence DTW matching for Query-by-Example search"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

try:
    from dtaidistance.subsequence.dtw import subsequence_alignment
    DTAIDISTANCE_AVAILABLE = True
except ImportError:
    DTAIDISTANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """
    Result of a subsequence DTW match.

    Attributes:
        distance: DTW distance (lower is better)
        start_frame: Starting frame index in reference sequence
        end_frame: Ending frame index in reference sequence
        path: Optional warping path
    """
    distance: float
    start_frame: int
    end_frame: int
    path: Optional[list] = None

    @property
    def duration_frames(self) -> int:
        """Duration of match in frames"""
        return self.end_frame - self.start_frame

    def __repr__(self) -> str:
        return (f"MatchResult(distance={self.distance:.4f}, "
                f"frames={self.start_frame}-{self.end_frame})")


class SubsequenceDTWMatcher:
    """
    Performs high-precision subsequence DTW matching.

    This is the verification stage in the Filter-and-Verify pipeline,
    providing frame-accurate alignment of query within reference sequences.
    """

    def __init__(self, use_c: bool = True):
        """
        Initialize the subsequence DTW matcher.

        Args:
            use_c: Use C-optimized implementation if available (faster)

        Raises:
            ImportError: If dtaidistance is not installed
        """
        if not DTAIDISTANCE_AVAILABLE:
            raise ImportError(
                "dtaidistance is required for DTW matching. "
                "Install with: pip install dtaidistance"
            )

        self.use_c = use_c
        logger.info("Subsequence DTW matcher initialized")

    def match(
        self,
        query: np.ndarray,
        reference: np.ndarray,
        window: Optional[int] = None
    ) -> MatchResult:
        """
        Find the best match of query within reference using S-DTW.

        Args:
            query: Query embedding sequence, shape (N, embedding_dim)
            reference: Reference embedding sequence, shape (M, embedding_dim)
                      where M >> N (reference is much longer)
            window: Optional Sakoe-Chiba window constraint (reduces computation)

        Returns:
            MatchResult with distance and frame boundaries

        Example:
            >>> matcher = SubsequenceDTWMatcher()
            >>> query_emb = np.random.rand(50, 1024)  # 50 frames
            >>> ref_emb = np.random.rand(500, 1024)   # 500 frames
            >>> result = matcher.match(query_emb, ref_emb)
            >>> print(f"Match at frames {result.start_frame}-{result.end_frame}")
        """
        # Validate inputs
        if query.ndim != 2 or reference.ndim != 2:
            raise ValueError(
                f"Query and reference must be 2D arrays. "
                f"Got query shape: {query.shape}, reference shape: {reference.shape}"
            )

        if query.shape[1] != reference.shape[1]:
            raise ValueError(
                f"Query and reference must have same embedding dimension. "
                f"Got query: {query.shape[1]}, reference: {reference.shape[1]}"
            )

        logger.debug(
            f"Running S-DTW: query shape={query.shape}, "
            f"reference shape={reference.shape}"
        )

        # Convert to float64 (dtaidistance C library requirement)
        query = query.astype(np.float64)
        reference = reference.astype(np.float64)

        # Perform subsequence alignment
        sa = subsequence_alignment(query, reference, use_c=self.use_c)

        # Get best match
        best = sa.best_match()

        # Extract results
        result = MatchResult(
            distance=best.distance,
            start_frame=best.segment[0],
            end_frame=best.segment[1]
        )

        logger.debug(f"S-DTW result: {result}")

        return result

    def match_top_k(
        self,
        query: np.ndarray,
        reference: np.ndarray,
        k: int = 3,
        min_distance_frames: int = 10
    ) -> list:
        """
        Find top-k best matches of query within a single reference sequence.

        Args:
            query: Query embedding sequence, shape (N, embedding_dim)
            reference: Reference embedding sequence, shape (M, embedding_dim)
            k: Number of top matches to return
            min_distance_frames: Minimum distance between matches to avoid overlaps

        Returns:
            List of MatchResult objects, sorted by distance (best first)
        """
        # Validate inputs
        if query.ndim != 2 or reference.ndim != 2:
            raise ValueError(
                f"Query and reference must be 2D arrays. "
                f"Got query shape: {query.shape}, reference shape: {reference.shape}"
            )

        if query.shape[1] != reference.shape[1]:
            raise ValueError(
                f"Query and reference must have same embedding dimension. "
                f"Got query: {query.shape[1]}, reference: {reference.shape[1]}"
            )

        # Convert to float64
        query = query.astype(np.float64)
        reference = reference.astype(np.float64)

        # Get all kbest matches from dtaidistance
        sa = subsequence_alignment(query, reference, use_c=self.use_c)

        # Get kbest matches (dtaidistance supports this)
        kbest_matches = sa.kbest_matches(k=k * 3)  # Get extra to filter overlaps

        results = []
        used_ranges = []

        for match in kbest_matches:
            start_frame = match.segment[0]
            end_frame = match.segment[1]

            # Check if this overlaps with already selected matches
            overlaps = False
            for used_start, used_end in used_ranges:
                # Check if ranges overlap or are too close
                if not (end_frame + min_distance_frames < used_start or
                        start_frame - min_distance_frames > used_end):
                    overlaps = True
                    break

            if not overlaps:
                result = MatchResult(
                    distance=match.distance,
                    start_frame=start_frame,
                    end_frame=end_frame
                )
                results.append(result)
                used_ranges.append((start_frame, end_frame))

                if len(results) >= k:
                    break

        logger.debug(f"Found {len(results)} non-overlapping matches")
        return results

    def match_multiple(
        self,
        query: np.ndarray,
        references: list,
        top_k: int = 1
    ) -> list:
        """
        Match query against multiple reference sequences.

        Args:
            query: Query embedding sequence
            references: List of reference embedding sequences
            top_k: Return top-k best matches

        Returns:
            List of (index, MatchResult) tuples, sorted by distance
        """
        results = []

        for idx, ref in enumerate(references):
            try:
                match = self.match(query, ref)
                results.append((idx, match))
            except Exception as e:
                logger.warning(f"Failed to match reference {idx}: {e}")
                continue

        # Sort by distance (ascending)
        results.sort(key=lambda x: x[1].distance)

        # Return top-k
        return results[:top_k]


def naive_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Simple DTW distance implementation (for testing/fallback).

    Args:
        seq1: First sequence (N, D)
        seq2: Second sequence (M, D)

    Returns:
        DTW distance
    """
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    return dtw_matrix[n, m]
