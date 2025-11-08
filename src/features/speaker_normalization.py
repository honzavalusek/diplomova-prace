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

    seq_len, emb_dim = embeddings.shape
    warped = np.zeros_like(embeddings)

    half_window = window_size // 2

    for i in range(seq_len):
        # Define window
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)

        window = embeddings[start:end]

        # Rank within window and apply Gaussian CDF
        for d in range(emb_dim):
            rank = np.searchsorted(np.sort(window[:, d]), embeddings[i, d])
            percentile = rank / len(window)
            warped[i, d] = norm.ppf(percentile + 1e-6)

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
