"""Distance metrics for comparing acoustic embeddings"""

import numpy as np
from typing import Callable


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.

    Cosine distance focuses on direction/orientation rather than magnitude,
    making it ideal for comparing neural embeddings.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine distance (1 - cosine similarity)
    """
    # Normalize vectors
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)

    # Cosine similarity
    cos_sim = np.dot(vec1_norm, vec2_norm)

    # Convert to distance
    return 1 - cos_sim


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Euclidean (L2) distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance
    """
    return np.linalg.norm(vec1 - vec2)


def get_distance_function(metric: str = 'cosine') -> Callable:
    """
    Get distance function by name.

    Args:
        metric: Distance metric name ('cosine' or 'euclidean')

    Returns:
        Distance function

    Raises:
        ValueError: If metric is not recognized
    """
    metrics = {
        'cosine': cosine_distance,
        'euclidean': euclidean_distance
    }

    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}. Choose from {list(metrics.keys())}")

    return metrics[metric]
