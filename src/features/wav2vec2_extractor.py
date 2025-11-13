"""XLS-R feature extraction using Hugging Face Transformers"""

import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class Wav2Vec2Extractor:
    """
    Extracts contextualized acoustic embeddings using selected model,
    with support for averaging features across a specified number of
    the final hidden layers.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        use_last_x_layers: Optional[int] = None,
        use_half_precision: bool = True
    ):
        """
        Initialize the Wav2Vec2 feature extractor.

        Args:
            model_name: Hugging Face model identifier
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
            use_last_x_layers: Number of final hidden layers to average.
                               If None or 1, only the last layer is used.
            use_half_precision: Use torch.float16 for weights and computation
                                (recommended for large models).
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_last_x_layers = use_last_x_layers
        self.use_half_precision = use_half_precision
        self.dtype = torch.float16 if self.use_half_precision and self.device == 'cuda' else torch.float32

        logger.info(f"Loading model: {model_name} on {self.device}")
        logger.info(f"Using precision: {self.dtype}")

        # Load feature extractor and model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        # Load model weights, potentially in half-precision
        self.model = Wav2Vec2Model.from_pretrained(
            model_name,
            torch_dtype=self.dtype
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # --- ATTRIBUTES ---
        self._embedding_dim = self.model.config.hidden_size
        self._num_layers = self.model.config.num_hidden_layers
        # --- END ATTRIBUTES ---

        logger.info(f"Model loaded successfully (layers={self._num_layers}, dim={self._embedding_dim})")

        if self.use_last_x_layers is not None and self.use_last_x_layers > 1:
            if not (1 <= self.use_last_x_layers <= self._num_layers):
                raise ValueError(
                    f"Invalid layer count: {self.use_last_x_layers}. "
                    f"Must be between 1 and {self._num_layers} (total layers)."
                )
            logger.info(f"Features will be averaged across the last {self.use_last_x_layers} layers.")


    def extract(
        self,
        audio_waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Extract contextualized embeddings from audio waveform, optionally
        averaging across a subset of the final transformer layers.

        Args:
            audio_waveform: Audio signal as numpy array
            sample_rate: Sample rate of audio (must be 16000 for XLS-R)

        Returns:
            Embeddings array of shape (sequence_length, embedding_dim)

        Raises:
            ValueError: If sample rate is not 16000
        """
        if sample_rate!= 16000:
            raise ValueError(f"Wav2Vec models require 16000 Hz, got {sample_rate} Hz")

        # Encode audio to model input format
        inputs = self.feature_extractor(
            audio_waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Use torch.autocast for mixed precision if enabled
        context_manager = torch.autocast(device_type=self.device, dtype=self.dtype)

        # Extract features without gradient computation
        with context_manager, torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        if self.use_last_x_layers is None or self.use_last_x_layers <= 1:
            # Default: Use the last hidden state (final layer output)
            embeddings = outputs.last_hidden_state
        else:
            # Advanced: Average across the last X internal layers
            # hidden_states contains the initial embedding + num_layers outputs.
            # We skip the first element (initial embedding)
            hidden_states = outputs.hidden_states[1:]

            # Calculate slice: e.g., if num_layers=24 and X=10, we slice from index 14 onwards.
            slice_start_index = self._num_layers - self.use_last_x_layers

            # Select the layers to average
            selected_layers = hidden_states[slice_start_index:]

            # Stack tensors along a new dimension (layer index), then take the mean
            stacked_tensors = torch.stack(selected_layers, dim=0)

            # Compute the mean across the layer dimension
            embeddings = torch.mean(stacked_tensors, dim=0)


        # Convert to numpy and remove batch dimension
        embeddings_np = embeddings.squeeze(0).cpu().numpy()

        logger.debug(f"Extracted embeddings shape: {embeddings_np.shape}")

        return embeddings_np

    @property
    def embedding_dim(self) -> int:
        """
        Returns the dimensionality of the embeddings.
        """
        return self._embedding_dim

    @property
    def hop_length(self) -> int:
        """
        Returns the hop length in samples (320 for all models).
        """
        return 320

    @property
    def num_layers(self) -> int:
        """
        Returns the total number of transformer layers in the model.
        """
        return self._num_layers