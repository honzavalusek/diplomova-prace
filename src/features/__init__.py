"""Feature extraction module for XLSR and XLS-R embeddings"""

from .wav2vec2_extractor import Wav2Vec2Extractor
from .audio_preprocessing import load_audio
from .frame_conversion import frames_to_seconds

__all__ = [
    'Wav2Vec2Extractor',
    'load_audio',
    'frames_to_seconds'
]
