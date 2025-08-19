"""Utility functions for feature memory banks used in anomaly detection.

This module implements two core components of the BEATs-based pipeline:

1. **Memory Mixup** – Augments target-domain features when the amount of
   available target data is limited.  Each target feature is interpolated with
   its nearest ``K`` source-domain features, creating synthetic examples that
   mitigate domain shift.
2. **kNN Anomaly Scoring** – Computes the k-nearest-neighbour distance of a
   test sample against both source and target memory banks.  The minimum of the
   two distances serves as the final anomaly score, while the individual
   distances are also returned for inspection.

The functions operate on ``torch.Tensor`` inputs but are written to be
self-contained and easily testable.
"""
from __future__ import annotations

from typing import Tuple

try:  # pragma: no cover - optional dependency
    import torch
except Exception as err:  # pragma: no cover
    torch = None  # type: ignore
    _IMPORT_ERROR = err
else:  # pragma: no cover
    _IMPORT_ERROR = None


def _check_torch() -> None:  # pragma: no cover - helper
    if torch is None:
        raise ImportError(
            "memory_utils requires `torch` to be installed"
        ) from _IMPORT_ERROR


def memory_mixup(
    target_feats: "torch.Tensor", source_feats: "torch.Tensor", k: int = 5, seed: int | None = None
) -> "torch.Tensor":
    """Augment target-domain features by interpolating with source features.

    Parameters
    ----------
    target_feats:
        Tensor of shape ``[N_t, D]`` representing target-domain features.
    source_feats:
        Tensor of shape ``[N_s, D]`` representing source-domain features.
    k:
        Number of nearest neighbours from the source domain to mix with each
        target feature.
    seed:
        Optional random seed to make the augmentation deterministic.

    Returns
    -------
    torch.Tensor
        Augmented target features containing the original target features and
        the synthetic ones generated via interpolation.
    """
    _check_torch()

    if target_feats.ndim != 2 or source_feats.ndim != 2:
        raise ValueError("Input feature matrices must be 2-D")
    if target_feats.size(1) != source_feats.size(1):
        raise ValueError("Source and target features must have the same dimension")

    if seed is not None:
        torch.manual_seed(seed)

    augmented = [target_feats]
    for t in target_feats:
        dists = torch.cdist(t.unsqueeze(0), source_feats)[0]
        k_small = min(k, len(dists))
        knn_idx = dists.topk(k_small, largest=False).indices
        for idx in knn_idx:
            alpha = torch.rand(1, device=target_feats.device)
            mixed = alpha * t + (1.0 - alpha) * source_feats[idx]
            augmented.append(mixed.unsqueeze(0))

    return torch.vstack(augmented)


def _knn_distance(sample: "torch.Tensor", bank: "torch.Tensor", k: int = 5) -> "torch.Tensor":
    """Compute the average distance to the ``k`` nearest neighbours."""
    dists = torch.cdist(sample.unsqueeze(0), bank)[0]
    k_small = min(k, len(dists))
    return dists.topk(k_small, largest=False).values.mean()


def knn_anomaly_score(
    sample: "torch.Tensor", source_bank: "torch.Tensor", target_bank: "torch.Tensor", k: int = 5
) -> Tuple[float, float, float]:
    """Compute kNN-based anomaly scores for ``sample``.

    Parameters
    ----------
    sample:
        Feature vector representing the test sample with shape ``[D]``.
    source_bank:
        Memory bank built from source-domain features with shape ``[N_s, D]``.
    target_bank:
        Memory bank built from (augmented) target-domain features with shape
        ``[N_t, D]``.
    k:
        Number of neighbours used to compute the kNN distance.

    Returns
    -------
    Tuple[float, float, float]
        ``(score, dist_source, dist_target)`` where ``score`` is the minimum of
        the two distances.  Higher scores indicate that the sample is more
        anomalous.
    """
    _check_torch()

    if sample.ndim != 1:
        raise ValueError("Sample must be a 1-D feature vector")

    d_s = _knn_distance(sample, source_bank, k)
    d_t = _knn_distance(sample, target_bank, k)
    score = torch.minimum(d_s, d_t)
    return float(score), float(d_s), float(d_t)
