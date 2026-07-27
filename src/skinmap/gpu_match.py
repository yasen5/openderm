"""Exact GPU (torch/CUDA) 2-NN SIFT descriptor matching.

Replaces the approximate FLANN kd-tree matcher when CUDA is available: a
brute-force torch.cdist top-2. This is EXACT nearest-neighbour search (FLANN
at trees=5/checks=64 is approximate and can miss true neighbours), it is
deterministic, and a 6000x6000 descriptor pair takes ~5 ms on an RTX 4090
versus ~100 ms on CPU. Per-frame descriptor tensors are cached in VRAM for
the duration of the matching stage (~3 MB per frame).

Distances use torch.cdist's matrix-multiply expansion (float32): relative
error ~1e-6 on SIFT-scale distances, far below the 0.75 ratio-test margin.
"""

from __future__ import annotations

import numpy as np

try:
    import torch

    _TORCH = True
except Exception:  # torch not installed
    _TORCH = False


def available() -> bool:
    return _TORCH and torch.cuda.is_available()


class GpuMatcher:
    def __init__(self, device: str = "cuda"):
        self.dev = torch.device(device)
        self._cache: dict[int, "torch.Tensor"] = {}

    def _tens(self, key: int, des: np.ndarray):
        t = self._cache.get(key)
        if t is None:
            t = torch.from_numpy(np.ascontiguousarray(des)).to(self.dev)
            self._cache[key] = t
        return t

    def knn2(self, key_q: int, des_q: np.ndarray, key_t: int, des_t: np.ndarray):
        """Top-2 L2 neighbours of every query descriptor in the train set.
        Returns (d1, d2, nn1) numpy arrays; d2 is +inf when the train set has
        a single descriptor (ratio test then rejects, same as FLANN k=2)."""
        q = self._tens(key_q, des_q)
        t = self._tens(key_t, des_t)
        d = torch.cdist(q[None], t[None])[0]
        k = min(2, t.shape[0])
        vals, idxs = torch.topk(d, k=k, dim=1, largest=False)
        if k == 1:
            d1 = vals[:, 0].cpu().numpy()
            return d1, np.full_like(d1, np.inf), idxs[:, 0].cpu().numpy()
        return (vals[:, 0].cpu().numpy(), vals[:, 1].cpu().numpy(), idxs[:, 0].cpu().numpy())

    def clear(self):
        self._cache.clear()
        if _TORCH:
            torch.cuda.empty_cache()
