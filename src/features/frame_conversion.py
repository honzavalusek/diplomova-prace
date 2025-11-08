"""Frame-to-time conversion utilities"""

import librosa
import numpy as np
from typing import Union


def frames_to_seconds(
    frame_idx: Union[int, np.ndarray],
    sample_rate: int = 16000,
    hop_length: int = 320
) -> Union[float, np.ndarray]:
    """
    Converts feature frame indices to time in seconds.

    This is the critical synchronization bridge between DTW frame indices
    and real-world timestamps in the audio file.

    Args:
        frame_idx: Frame index or array of indices
        sample_rate: Audio sample rate (default: 16000 Hz)
        hop_length: XLSR hop length in samples (default: 320)

    Returns:
        Time in seconds (float or array)

    Example:
        >>> frames_to_seconds(100, sample_rate=16000, hop_length=320)
        2.0  # 100 frames * 320 samples/frame / 16000 samples/sec = 2.0 sec
    """
    return librosa.frames_to_time(
        frame_idx,
        sr=sample_rate,
        hop_length=hop_length
    )


def seconds_to_frames(
    time_sec: Union[float, np.ndarray],
    sample_rate: int = 16000,
    hop_length: int = 320
) -> Union[int, np.ndarray]:
    """
    Converts time in seconds to feature frame indices.

    Args:
        time_sec: Time in seconds
        sample_rate: Audio sample rate (default: 16000 Hz)
        hop_length: XLSR hop length in samples (default: 320)

    Returns:
        Frame index (int or array)
    """
    return librosa.time_to_frames(
        time_sec,
        sr=sample_rate,
        hop_length=hop_length
    )
