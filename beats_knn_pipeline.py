"""High-level pipeline tying BEATs feature extraction with memory-based kNN.

The module exposes convenience functions to build source/target memory banks
and to compute anomaly scores for new samples.  It does not depend on any
specific dataset structure; callers are expected to provide raw waveforms.
"""
from __future__ import annotations

from typing import Iterable, Tuple

try:  # pragma: no cover - optional dependency
    import torch
except Exception as err:  # pragma: no cover
    torch = None  # type: ignore
    _IMPORT_ERROR = err
else:  # pragma: no cover
    _IMPORT_ERROR = None

from beats_feature_extractor import BEATsFeatureExtractor
from memory_utils import memory_mixup, knn_anomaly_score


def _check_torch() -> None:  # pragma: no cover - helper
    if torch is None:
        raise ImportError("BEATs pipeline requires `torch` to be installed") from _IMPORT_ERROR


def build_memory_banks(
    source_waveforms: Iterable["torch.Tensor"],
    target_waveforms: Iterable["torch.Tensor"],
    sample_rate: int,
    *,
    extractor: BEATsFeatureExtractor | None = None,
    k_mixup: int = 5,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Create source and (augmented) target memory banks.

    Parameters
    ----------
    source_waveforms, target_waveforms:
        Iterables of waveform tensors.
    sample_rate:
        Sampling rate shared by all waveforms.
    extractor:
        Optional :class:`BEATsFeatureExtractor` instance.  A new extractor is
        created if not provided.
    k_mixup:
        Number of neighbours used when augmenting the target memory bank via
        memory mixup.
    """
    _check_torch()
    extractor = extractor or BEATsFeatureExtractor()

    src_feats = [extractor(w, sample_rate) for w in source_waveforms]
    tgt_feats = [extractor(w, sample_rate) for w in target_waveforms]
    src_bank = torch.vstack(src_feats)
    tgt_bank = torch.vstack(tgt_feats)
    tgt_bank = memory_mixup(tgt_bank, src_bank, k=k_mixup)
    return src_bank, tgt_bank


def score_sample(
    waveform: "torch.Tensor",
    sample_rate: int,
    source_bank: "torch.Tensor",
    target_bank: "torch.Tensor",
    *,
    extractor: BEATsFeatureExtractor | None = None,
    k: int = 5,
) -> Tuple[float, float, float]:
    """Compute the anomaly score for ``waveform``.

    Parameters
    ----------
    waveform:
        Waveform tensor representing the test sample.
    sample_rate:
        Sampling rate of the waveform.
    source_bank, target_bank:
        Memory banks built with :func:`build_memory_banks`.
    extractor:
        Optional feature extractor instance.
    k:
        Number of neighbours used for kNN scoring.
    """
    _check_torch()
    extractor = extractor or BEATsFeatureExtractor()
    feat = extractor(waveform, sample_rate)
    # ``feat`` has shape [1, D]; squeeze to 1-D before scoring.
    return knn_anomaly_score(feat.squeeze(0), source_bank, target_bank, k=k)
