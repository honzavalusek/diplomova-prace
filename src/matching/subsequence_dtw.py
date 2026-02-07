"""Subsequence DTW matching for Query-by-Example search"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

from dtaidistance.subsequence import SubsequenceAlignment
import logging

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
    Performs high-precision subsequence DTW matching using dtaidistance library.

    This is the verification stage in the Filter-and-Verify pipeline,
    providing frame-accurate alignment of query within reference sequences.

    Uses dtaidistance's subsequence_alignment for optimal subsequence matching.
    """

    def __init__(self, use_c: bool = True, window: Optional[int] = None, use_cosine: bool = True):
        """
        Initialize the subsequence DTW matcher.

        Args:
            use_c: Use C-optimized implementation if available (faster)
            window: Sakoe-Chiba window constraint (limits temporal deviation).
                   Typical values: 10-50 frames for speech.
                   None = no constraint (slower, may be less reliable).
            use_cosine: If True, L2-normalize embeddings before DTW to use cosine
                       distance instead of Euclidean. Recommended for neural embeddings.

        Raises:
            ImportError: If dtaidistance is not installed
        """
        self.use_c = use_c
        self.window = window
        self.use_cosine = use_cosine

        parts = [f"window={window}" if window else "no window", f"use_c={use_c}", f"cosine={use_cosine}"]
        logger.info(f"Subsequence DTW matcher initialized ({', '.join(parts)})")

    def match(
        self,
        query: np.ndarray,
        reference: np.ndarray,
        window: Optional[int] = None
    ) -> MatchResult:
        """
        Find the best match of query within reference using subsequence DTW.

        Args:
            query: Query embedding sequence, shape (N, embedding_dim)
            reference: Reference embedding sequence, shape (M, embedding_dim)
                      where M >> N (reference is much longer)
            window: Optional Sakoe-Chiba window constraint (reduces computation).
                   If None, uses the instance default from __init__.

        Returns:
            MatchResult with distance and frame boundaries

        Example:
            >>> matcher = SubsequenceDTWMatcher(window=25)
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

        # Use method parameter or fall back to instance default
        effective_window = window if window is not None else self.window

        logger.debug(
            f"Running S-DTW: query shape={query.shape}, "
            f"reference shape={reference.shape}, window={effective_window}"
        )

        # Convert to float64 (dtaidistance C library requirement)
        query = query.astype(np.float64)
        reference = reference.astype(np.float64)

        # L2-normalize for cosine distance (Euclidean on unit vectors ∝ cosine distance)
        if self.use_cosine:
            query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
            reference = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-8)

        # Perform subsequence alignment
        sa = SubsequenceAlignment(query, reference, penalty=0.1, use_c=self.use_c, window=effective_window)
        sa.align()

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
        min_distance_frames: int = 10,
        window: Optional[int] = None
    ) -> List[MatchResult]:
        """
        Find top-k best matches of query within a single reference sequence.

        Args:
            query: Query embedding sequence, shape (N, embedding_dim)
            reference: Reference embedding sequence, shape (M, embedding_dim)
            k: Number of top matches to return
            min_distance_frames: Minimum distance between matches to avoid overlaps
            window: Optional Sakoe-Chiba window constraint.
                   If None, uses the instance default from __init__.

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

        # Use method parameter or fall back to instance default
        effective_window = window if window is not None else self.window

        # Convert to float64
        query = query.astype(np.float64)
        reference = reference.astype(np.float64)

        # L2-normalize for cosine distance (Euclidean on unit vectors ∝ cosine distance)
        if self.use_cosine:
            query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
            reference = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-8)

        # Perform subsequence alignment (same parameters as match())
        sa = SubsequenceAlignment(query, reference, penalty=0.1, use_c=self.use_c, window=effective_window)
        sa.align()

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
        top_k: int = 1,
        window: Optional[int] = None
    ) -> list:
        """
        Match query against multiple reference sequences.

        Args:
            query: Query embedding sequence
            references: List of reference embedding sequences
            top_k: Return top-k best matches
            window: Optional Sakoe-Chiba window constraint.
                   If None, uses the instance default from __init__.

        Returns:
            List of (index, MatchResult) tuples, sorted by distance
        """
        results = []

        for idx, ref in enumerate(references):
            try:
                match = self.match(query, ref, window=window)
                results.append((idx, match))
            except Exception as e:
                logger.warning(f"Failed to match reference {idx}: {e}")
                continue

        # Sort by distance (ascending)
        results.sort(key=lambda x: x[1].distance)

        # Return top-k
        return results[:top_k]
