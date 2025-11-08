"""XLS-R feature extraction using Hugging Face Transformers"""

import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class XLSRExtractor:
    """
    Extracts contextualized acoustic embeddings using XLS-R models.

    XLS-R (Cross-Lingual Speech Representations) are improved models
    with more parameters and training data than XLSR-53.

    Available models:
    - facebook/wav2vec2-xls-r-300m (300M params, 1024-D embeddings)
    - facebook/wav2vec2-xls-r-1b (1B params, 1280-D embeddings)
    - facebook/wav2vec2-xls-r-2b (2B params, 1920-D embeddings)

    Uses the last_hidden_state output for contextualized embeddings.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        device: Optional[str] = None
    ):
        """
        Initialize the XLS-R feature extractor.

        Args:
            model_name: Hugging Face model identifier
                       Default: "facebook/wav2vec2-xls-r-300m"
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading XLS-R model: {model_name} on {self.device}")

        # Load feature extractor and model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Auto-detect embedding dimension from model config
        self._embedding_dim = self.model.config.hidden_size

        logger.info(f"XLS-R model loaded successfully (embedding_dim={self._embedding_dim})")

    def extract(
        self,
        audio_waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Extract contextualized embeddings from audio waveform.

        Args:
            audio_waveform: Audio signal as numpy array
            sample_rate: Sample rate of audio (must be 16000 for XLS-R)

        Returns:
            Embeddings array of shape (sequence_length, embedding_dim)
            - 300M model: (sequence_length, 1024)
            - 1B model: (sequence_length, 1280)
            - 2B model: (sequence_length, 1920)

        Raises:
            ValueError: If sample rate is not 16000
        """
        if sample_rate != 16000:
            raise ValueError(f"XLS-R requires 16000 Hz, got {sample_rate} Hz")

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
        # Shape: (batch_size=1, sequence_length, hidden_size)
        embeddings = outputs.last_hidden_state

        # Convert to numpy and remove batch dimension
        # Shape: (sequence_length, hidden_size)
        embeddings_np = embeddings.squeeze(0).cpu().numpy()

        logger.debug(f"Extracted embeddings shape: {embeddings_np.shape}")

        return embeddings_np

    @property
    def embedding_dim(self) -> int:
        """
        Returns the dimensionality of the embeddings.

        - 300M model: 1024
        - 1B model: 1280
        - 2B model: 1920
        """
        return self._embedding_dim

    @property
    def hop_length(self) -> int:
        """
        Returns the hop length in samples (320 for all XLS-R models).
        """
        return 320
