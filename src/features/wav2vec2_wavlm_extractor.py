"""XLS-R feature extraction using Hugging Face Transformers"""

import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class Wav2Vec2WavLmExtractor:
    """
    Extracts contextualized acoustic embeddings using selected model,
    with support for averaging features across a specified range of
    transformer layers.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        layer_min: Optional[int] = None,
        layer_max: Optional[int] = None,
        use_half_precision: bool = True
    ):
        """
        Initialize the Wav2Vec2 feature extractor.

        Args:
            model_name: Hugging Face model identifier
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
            layer_min: Minimum layer index (0-based, inclusive). If None with
                       layer_max set, defaults to 0.
            layer_max: Maximum layer index (0-based, inclusive). If None with
                       layer_min set, defaults to last layer.
            use_half_precision: Use torch.float16 for weights and computation
                                (recommended for large models).

        Layer selection behavior:
            - Both None: Average all layers [0, num_layers-1] (default)
            - layer_min=M, layer_max=None: Average layers [M, num_layers-1]
            - layer_min=None, layer_max=X: Average layers [0, X]
            - layer_min=M, layer_max=X: Average layers [M, X]
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.layer_min = layer_min
        self.layer_max = layer_max
        self.use_half_precision = use_half_precision
        self.dtype = torch.float16 if self.use_half_precision and self.device == 'cuda' else torch.float32

        logger.info(f"Loading model: {model_name} on {self.device}")
        logger.info(f"Using precision: {self.dtype}")

        # Load feature extractor and model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        # Load model weights, potentially in half-precision
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=self.dtype
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # --- ATTRIBUTES ---
        self._embedding_dim = self.model.config.hidden_size
        self._num_layers = self.model.config.num_hidden_layers
        # --- END ATTRIBUTES ---

        logger.info(f"Model loaded successfully (layers={self._num_layers}, dim={self._embedding_dim})")

        logger.info(f"Model has {self._num_layers} transformer layers.")

        # Validate layer indices
        max_layer_idx = self._num_layers - 1
        if self.layer_min is not None:
            if not (0 <= self.layer_min <= max_layer_idx):
                raise ValueError(
                    f"Invalid layer_min: {self.layer_min}. "
                    f"Must be between 0 and {max_layer_idx}."
                )
        if self.layer_max is not None:
            if not (0 <= self.layer_max <= max_layer_idx):
                raise ValueError(
                    f"Invalid layer_max: {self.layer_max}. "
                    f"Must be between 0 and {max_layer_idx}."
                )
        if self.layer_min is not None and self.layer_max is not None:
            if self.layer_min > self.layer_max:
                raise ValueError(
                    f"layer_min ({self.layer_min}) must be <= layer_max ({self.layer_max})."
                )

        # Log layer selection
        effective_min = self.layer_min if self.layer_min is not None else 0
        effective_max = self.layer_max if self.layer_max is not None else max_layer_idx
        logger.info(f"Features will be averaged across layers {effective_min} to {effective_max}.")


    def extract(
        self,
        audio_waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Extract contextualized embeddings from audio waveform, averaging
        across the specified range of transformer layers.

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

        # Average across specified layer range (default: all layers)
        # hidden_states contains the initial embedding + num_layers outputs.
        # We skip the first element (initial embedding), so indices become 0..num_layers-1
        hidden_states = outputs.hidden_states[1:]

        effective_min = self.layer_min if self.layer_min is not None else 0
        effective_max = self.layer_max if self.layer_max is not None else self._num_layers - 1

        # Slice is inclusive on both ends, so we need max+1 for Python slicing
        selected_layers = hidden_states[effective_min:effective_max + 1]

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