"""Exact GPU (torch/CUDA) 2-NN SIFT descriptor matching.

Replaces the approximate FLANN kd-tree matcher when CUDA is gpu_matcher_available: a
brute-force torch.cdist top-2. This is EXACT nearest-neighbour search (FLANN
at trees=5/checks=64 is approximate and can miss true neighbours), it is
deterministic, and a 6000x6000 descriptor pair takes ~5 ms on an RTX 4090
versus ~100 ms on CPU. Per-frame descriptor tensors are cached in VRAM for
the duration of the matching stage (~3 MB per frame).

Distances use torch.cdist's matrix-multiply expansion (float32): relative
error ~1e-6 on SIFT-scale distances, far below the 0.75 ratio-test margin.
"""

# Torch is an optional GPU-only dependency.
# pyright: reportMissingImports=false

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]

try:
    import torch

    _TORCH = True
except Exception:  # torch not installed
    _TORCH = False


def gpu_matcher_available() -> bool:
    return _TORCH and torch.cuda.is_available()


class GpuMatcher:
    def __init__(self, device: str = "cuda") -> None:
        self.dev = torch.device(device)
        self._cache: dict[int, "torch.Tensor"] = {}

    def _tens(self, key: int, des: FloatArray) -> "torch.Tensor":
        cached_descriptors = self._cache.get(key)
        if cached_descriptors is None:
            cached_descriptors = torch.from_numpy(np.ascontiguousarray(des)).to(self.dev)
            self._cache[key] = cached_descriptors
        return cached_descriptors

    def knn2(
        self, key_q: int, des_q: FloatArray, key_t: int, des_t: FloatArray
    ) -> tuple[FloatArray, FloatArray, IntArray]:
        """Top-2 L2 neighbours of every query descriptor in the train set.
        Returns (d1, d2, nn1) numpy arrays; d2 is +inf when the train set has
        a single descriptor (ratio test then rejects, same as FLANN k=2)."""
        query_descriptors = self._tens(key_q, des_q)
        train_descriptors = self._tens(key_t, des_t)
        pairwise_distances = torch.cdist(query_descriptors[None], train_descriptors[None])[0]
        neighbor_count = min(2, train_descriptors.shape[0])
        nearest_distances, nearest_indices = torch.topk(
            pairwise_distances, k=neighbor_count, dim=1, largest=False
        )
        if neighbor_count == 1:
            first_neighbor_distances = nearest_distances[:, 0].cpu().numpy()
            second_neighbor_distances = np.full_like(first_neighbor_distances, np.inf)
            first_neighbor_indices = nearest_indices[:, 0].cpu().numpy()
            return (
                first_neighbor_distances,
                second_neighbor_distances,
                first_neighbor_indices,
            )
        return (
            nearest_distances[:, 0].cpu().numpy(),
            nearest_distances[:, 1].cpu().numpy(),
            nearest_indices[:, 0].cpu().numpy(),
        )

    def clear(self) -> None:
        self._cache.clear()
        if _TORCH:
            torch.cuda.empty_cache()
