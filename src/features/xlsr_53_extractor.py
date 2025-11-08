"""XLSR-53 feature extraction using Hugging Face Transformers"""

import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class XLSR53FeatureExtractor:
    """
    Extracts contextualized acoustic embeddings using XLSR-53.

    XLSR-53 is trained on 53 languages with 300M parameters.
    Output: 1024-dimensional contextualized embeddings.

    Uses the last_hidden_state output rather than raw CNN features
    for higher-quality semantic matching.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-xlsr-53",
        device: Optional[str] = None
    ):
        """
        Initialize the XLSR feature extractor.

        Args:
            model_name: Hugging Face model identifier
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading XLSR model: {model_name} on {self.device}")

        # Load feature extractor and model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        logger.info("XLSR model loaded successfully")

    def extract(
        self,
        audio_waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Extract contextualized embeddings from audio waveform.

        Args:
            audio_waveform: Audio signal as numpy array
            sample_rate: Sample rate of audio (must be 16000 for XLSR)

        Returns:
            Embeddings array of shape (sequence_length, 1024)

        Raises:
            ValueError: If sample rate is not 16000
        """
        if sample_rate != 16000:
            raise ValueError(f"XLSR requires 16000 Hz, got {sample_rate} Hz")

        # Encode audio to model input format
        inputs = self.feature_extractor(
            audio_waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features without gradient computation
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get contextualized embeddings (last_hidden_state)
        # Shape: (batch_size=1, sequence_length, hidden_size=1024)
        embeddings = outputs.last_hidden_state

        # Convert to numpy and remove batch dimension
        # Shape: (sequence_length, 1024)
        embeddings_np = embeddings.squeeze(0).cpu().numpy()

        logger.debug(f"Extracted embeddings shape: {embeddings_np.shape}")

        return embeddings_np

    @property
    def embedding_dim(self) -> int:
        """
        Returns the dimensionality of the embeddings (1024 for XLSR-53).
        """
        return 1024

    @property
    def hop_length(self) -> int:
        """
        Returns the hop length in samples (320 for XLSR-53).
        """
        return 320