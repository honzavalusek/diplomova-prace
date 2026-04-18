"""Feature extraction module for self-supervised speech embeddings"""

from .ssl_speech_extractor import SSLSpeechExtractor
from .audio_preprocessing import load_audio
from .frame_conversion import frames_to_seconds

__all__ = [
    'SSLSpeechExtractor',
    'load_audio',
    'frames_to_seconds'
]
