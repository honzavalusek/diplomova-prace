"""Self-supervised speech embedding extraction using Hugging Face Transformers"""

import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class SSLSpeechExtractor:
    """
    Extracts contextualized acoustic embeddings from self-supervised speech
    models (wav2vec2, XLS-R, XLSR-53, WavLM), with support for averaging
    features across a specified range of transformer layers.
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
        Initialize the SSL speech feature extractor.

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

        # Cache the resolved layer range so we don't recompute it on every call
        self._layer_min_eff = self.layer_min if self.layer_min is not None else 0
        self._layer_max_eff = self.layer_max if self.layer_max is not None else max_layer_idx
        self._num_selected = self._layer_max_eff - self._layer_min_eff + 1
        logger.info(f"Features will be averaged across layers {self._layer_min_eff} to {self._layer_max_eff}.")


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
            sample_rate: Sample rate of audio (must be 16000)

        Returns:
            Embeddings array of shape (sequence_length, embedding_dim)

        Raises:
            ValueError: If sample rate is not 16000
        """
        if sample_rate != 16000:
            raise ValueError(f"SSL speech models require 16000 Hz, got {sample_rate} Hz")

        inputs = self.feature_extractor(
            audio_waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        embeddings = self._forward_with_layer_mean(inputs)

        embeddings_np = embeddings.squeeze(0).cpu().numpy()
        logger.debug(f"Extracted embeddings shape: {embeddings_np.shape}")
        return embeddings_np

    def _forward_with_layer_mean(self, inputs: dict) -> torch.Tensor:
        """
        Run a forward pass and return the per-frame mean over the selected
        layers, accumulating in a single tensor instead of holding every
        layer's output via output_hidden_states=True.
        """
        acc: Optional[torch.Tensor] = None

        def hook(_module, _input, output):
            nonlocal acc
            hs = output[0] if isinstance(output, tuple) else output
            acc = hs if acc is None else acc + hs

        layers = self.model.encoder.layers
        handles = [
            layers[i].register_forward_hook(hook)
            for i in range(self._layer_min_eff, self._layer_max_eff + 1)
        ]
        try:
            with torch.autocast(device_type=self.device, dtype=self.dtype), torch.no_grad():
                self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        return acc / self._num_selected

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