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
        use_half_precision: bool = True,
        chunk_seconds: Optional[float] = 30.0,
        chunk_overlap_seconds: float = 2.0,
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
            chunk_seconds: Chunk length in seconds for sliding-window inference
                           on long inputs. None or 0 disables chunking.
            chunk_overlap_seconds: Overlap on each side of a chunk; trimmed
                                   from the stitched output to hide boundary
                                   effects. Must satisfy
                                   0 < overlap < chunk_seconds / 2.

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

        # Chunked-inference configuration
        self.chunk_seconds = chunk_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds
        self._chunking_enabled = bool(chunk_seconds and chunk_seconds > 0)
        if self._chunking_enabled and not (0 < chunk_overlap_seconds < chunk_seconds / 2):
            raise ValueError(
                f"chunk_overlap_seconds ({chunk_overlap_seconds}) must satisfy "
                f"0 < overlap < chunk_seconds/2 ({chunk_seconds / 2})."
            )


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

        if not self._chunking_enabled:
            return self._extract_single(audio_waveform, sample_rate)

        chunk_samples = int(self.chunk_seconds * sample_rate)
        overlap_samples = int(self.chunk_overlap_seconds * sample_rate)
        if len(audio_waveform) <= chunk_samples + 2 * overlap_samples:
            return self._extract_single(audio_waveform, sample_rate)

        return self._extract_chunked(audio_waveform, sample_rate, chunk_samples, overlap_samples)

    def _preprocess(self, audio_waveform: np.ndarray, sample_rate: int) -> dict:
        inputs = self.feature_extractor(
            audio_waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

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

    def _extract_single(self, audio_waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        inputs = self._preprocess(audio_waveform, sample_rate)
        embeddings = self._forward_with_layer_mean(inputs)
        embeddings_np = embeddings.squeeze(0).cpu().numpy()
        logger.debug(f"Extracted embeddings shape: {embeddings_np.shape}")
        return embeddings_np

    def _extract_chunked(
        self,
        audio_waveform: np.ndarray,
        sample_rate: int,
        chunk_samples: int,
        overlap_samples: int,
    ) -> np.ndarray:
        # Whole-utterance normalization once up front so per-chunk forward
        # passes see the same statistics a single-pass forward would.
        do_normalize = getattr(self.feature_extractor, "do_normalize", False)
        if do_normalize:
            mean = audio_waveform.mean()
            std = np.sqrt(audio_waveform.var() + 1e-7)
            audio_waveform = (audio_waveform - mean) / std

        overlap_frames = overlap_samples // self.hop_length
        n = len(audio_waveform)
        chunk_embeds: list[torch.Tensor] = []

        try:
            if do_normalize:
                self.feature_extractor.do_normalize = False

            for s in range(0, n, chunk_samples):
                a = max(0, s - overlap_samples)
                b = min(n, s + chunk_samples + overlap_samples)
                is_first = s == 0
                is_last = s + chunk_samples >= n

                inputs = self._preprocess(audio_waveform[a:b], sample_rate)
                emb = self._forward_with_layer_mean(inputs)  # (1, T, D)

                T = emb.shape[1]
                start_trim = 0 if is_first else overlap_frames
                end_trim = T if is_last else T - overlap_frames
                chunk_embeds.append(emb[:, start_trim:end_trim, :])
        finally:
            if do_normalize:
                self.feature_extractor.do_normalize = True

        full = torch.cat(chunk_embeds, dim=1)
        embeddings_np = full.squeeze(0).cpu().numpy()
        logger.debug(f"Extracted embeddings shape (chunked, {len(chunk_embeds)} chunks): {embeddings_np.shape}")
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