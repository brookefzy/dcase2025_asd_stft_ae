"""Feature extraction using the BEATs pre-trained model.

This module provides a thin wrapper around the BEATs model from the
``transformers`` library.  It extracts embeddings from an input waveform and
applies mean-pooling over the temporal dimension while keeping the spectral
structure.  The resulting feature vector has shape ``[F * C]`` where ``F`` is
the number of spectral patches and ``C`` is the hidden size of the model.

The implementation follows the pipeline described in the project
requirements:

1. Run a forward pass of the BEATs model to obtain patch embeddings with
   shape ``[B, T*F, C]``.
2. Reshape the sequence dimension into ``[T, F]`` to recover the temporal and
   spectral dimensions.
3. Apply mean pooling over the temporal dimension only, obtaining
   ``[B, F, C]``.
4. Flatten the pooled embedding into ``[B, F*C]`` which can be stored in a
   memory bank.

The code relies on ``transformers`` and ``torch``.  Importing this module will
not instantiate the model until :class:`BEATsFeatureExtractor` is constructed,
so environments without these dependencies can still import the file without
immediate failures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import BEATsModel, BEATsProcessor
except Exception as err:  # pragma: no cover - handle missing deps gracefully
    torch = None  # type: ignore
    BEATsModel = None  # type: ignore
    BEATsProcessor = None  # type: ignore
    _IMPORT_ERROR = err
else:  # pragma: no cover - no error
    _IMPORT_ERROR = None


@dataclass
class BEATsFeatureExtractor:  # pragma: no cover - small wrapper
    """Extract features from waveforms using a pre-trained BEATs model.

    Parameters
    ----------
    model_name:
        Name of the pre-trained BEATs checkpoint hosted on Hugging Face.
    device:
        Torch device on which the model will run.  If ``None`` the device is
        chosen automatically.
    """

    model_name: str = "microsoft/beats-base"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if BEATsModel is None or BEATsProcessor is None:
            raise ImportError(
                "BEATsFeatureExtractor requires `torch` and `transformers` to be installed."
            ) from _IMPORT_ERROR

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BEATsProcessor.from_pretrained(self.model_name)
        self.model = BEATsModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        config = self.model.config
        # Estimate the number of spectral patches produced by the model.  This
        # depends on the number of mel bins and the frequency patch size.
        num_mels = getattr(config, "num_mel_bins", 128)
        patch_size = getattr(config, "patch_size", 16)
        if isinstance(patch_size, (list, tuple)):
            patch_freq = patch_size[-1]
        else:
            patch_freq = patch_size
        self.freq_patches = max(1, num_mels // patch_freq)

    @torch.no_grad()
    def __call__(self, waveform: "torch.Tensor", sample_rate: int) -> "torch.Tensor":
        """Extract pooled BEATs embeddings from an input waveform.

        Parameters
        ----------
        waveform:
            A tensor containing the audio waveform with shape ``[n_samples]`` or
            ``[channel, n_samples]``.  The audio is assumed to be mono or the
            first channel is used.
        sample_rate:
            Sampling rate of the waveform.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``[F * C]`` containing the pooled embedding.
        """

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        inputs = self.processor(
            waveform, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state  # [B, T*F, C]
        bsz, seq_len, hidden = embeddings.shape
        freq = self.freq_patches
        time = seq_len // freq
        embeddings = embeddings.view(bsz, time, freq, hidden)  # [B, T, F, C]
        pooled = embeddings.mean(dim=1)  # [B, F, C]
        return pooled.reshape(bsz, freq * hidden)
