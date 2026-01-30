"""Speaker normalization techniques for speaker-independent matching"""

import numpy as np
from typing import Optional


def mean_variance_normalization(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to zero mean and unit variance.

    Removes speaker-specific baseline and scales, making embeddings
    more speaker-independent.

    Args:
        embeddings: Shape (sequence_length, embedding_dim)

    Returns:
        Normalized embeddings of same shape
    """
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True) + 1e-8

    return (embeddings - mean) / std


def cepstral_mean_normalization(embeddings: np.ndarray) -> np.ndarray:
    """
    Subtract the mean (CMN) - simpler than MVN.

    Args:
        embeddings: Shape (sequence_length, embedding_dim)

    Returns:
        Mean-normalized embeddings
    """
    mean = embeddings.mean(axis=0, keepdims=True)
    return embeddings - mean


def feature_warping(embeddings: np.ndarray, window_size: int = 300) -> np.ndarray:
    """
    Apply feature warping for speaker normalization.

    Ranks features within a sliding window and applies Gaussian CDF.
    More robust to speaker variations than simple normalization.

    Args:
        embeddings: Shape (sequence_length, embedding_dim)
        window_size: Warping window size in frames

    Returns:
        Warped embeddings
    """
    from scipy.stats import norm
    from scipy.ndimage import uniform_filter1d

    seq_len, emb_dim = embeddings.shape

    # For short sequences, fall back to global ranking (simpler, still fast)
    if seq_len <= window_size:
        # Rank each dimension globally
        ranks = np.argsort(np.argsort(embeddings, axis=0), axis=0).astype(np.float64)
        percentiles = (ranks + 1) / (seq_len + 1)  # +1 to avoid 0 and 1
        return norm.ppf(percentiles)

    # For longer sequences, use sliding window approach
    # Vectorized: process all dimensions at once per frame
    warped = np.zeros_like(embeddings, dtype=np.float64)
    half_window = window_size // 2

    for i in range(seq_len):
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        window = embeddings[start:end]
        window_len = end - start

        # Vectorized ranking: compare current frame to all window frames
        # Shape: (window_len, emb_dim) compared to (1, emb_dim) -> (window_len, emb_dim)
        current = embeddings[i:i+1]  # Keep dims for broadcasting
        ranks = (window < current).sum(axis=0)  # Count how many are smaller

        # Convert to percentile, avoiding 0 and 1 for ppf stability
        percentiles = (ranks + 1) / (window_len + 2)
        warped[i] = norm.ppf(percentiles)

    return warped


def apply_normalization(
    embeddings: np.ndarray,
    method: str = "mvn"
) -> np.ndarray:
    """
    Apply speaker normalization to embeddings.

    Args:
        embeddings: Shape (sequence_length, embedding_dim)
        method: Normalization method ('mvn', 'cmn', 'warp', or 'none')

    Returns:
        Normalized embeddings
    """
    if method == "mvn":
        return mean_variance_normalization(embeddings)
    elif method == "cmn":
        return cepstral_mean_normalization(embeddings)
    elif method == "warp":
        return feature_warping(embeddings)
    elif method == "none":
        return embeddings
    else:
        raise ValueError(f"Unknown normalization method: {method}")
