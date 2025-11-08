"""Feature extraction module for XLSR and XLS-R embeddings"""

from .xlsr_53_extractor import XLSR53FeatureExtractor
from .xls_r_extractor import XLSRExtractor
from .audio_preprocessing import load_audio
from .frame_conversion import frames_to_seconds

# Backwards compatibility alias
XLSRFeatureExtractor = XLSR53FeatureExtractor

__all__ = [
    'XLSR53FeatureExtractor',
    'XLSRExtractor',
    'XLSRFeatureExtractor',  # Backwards compatibility
    'load_audio',
    'frames_to_seconds'
]
