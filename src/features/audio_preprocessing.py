"""Audio preprocessing utilities using librosa"""

import librosa
import numpy as np
from typing import Tuple


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Loads audio file and ensures correct sample rate.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000 Hz for XLSR)

    Returns:
        Tuple of (audio_waveform, sample_rate)

    Raises:
        ValueError: If resampling fails
        FileNotFoundError: If audio file doesn't exist
    """
    try:
        audio_waveform, sr = librosa.load(audio_path, sr=target_sr)
    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if sr != target_sr:
        raise ValueError(
            f"Resampling failed. Required SR: {target_sr}, Actual SR: {sr}"
        )

    return audio_waveform, sr


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    duration = librosa.get_duration(path=audio_path)
    return duration
