"""Feature extraction module for XLSR and XLS-R embeddings"""

from .wav2vec2_wavlm_extractor import Wav2Vec2WavLmExtractor
from .audio_preprocessing import load_audio
from .frame_conversion import frames_to_seconds

__all__ = [
    'Wav2Vec2WavLmExtractor',
    'load_audio',
    'frames_to_seconds'
]
