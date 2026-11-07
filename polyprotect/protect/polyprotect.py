"""PolyProtect V→P mapping (Hahn & Marcel, Eqs. 1–3)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def sample_user_params(
    m: int,
    rng: np.random.RandomState,
    c_low: int = -50,
    c_high: int = 50,
    e_low: int = 1,
    e_high: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample unique non-zero integer coefficients and unique positive exponents."""
    coeffs: List[int] = []
    while len(coeffs) < m:
        c = int(rng.randint(c_low, c_high + 1))
        if c == 0 or c in coeffs:
            continue
        coeffs.append(c)
    exps: List[int] = []
    while len(exps) < m:
        e = int(rng.randint(e_low, e_high + 1))
        if e in exps:
            continue
        exps.append(e)
    # if m > (e_high-e_low+1), allow repeats for exponents
    if len(exps) < m:
        while len(exps) < m:
            exps.append(int(rng.randint(e_low, e_high + 1)))
    return np.asarray(coeffs, dtype=np.int64), np.asarray(exps[:m], dtype=np.int64)


class PolyProtect:
    """Map n-D embeddings to protected templates with user-specific polynomials."""

    def __init__(self, m: int = 5, overlap: int = 4):
        if not (0 <= overlap < m):
            raise ValueError("overlap must be in [0, m)")
        self.m = m
        self.overlap = overlap
        self.stride = m - overlap

    def _starts(self, n: int) -> List[int]:
        # Stop once every element of V is used (Sec. 3); only the last set may need padding.
        starts: List[int] = [0]
        while starts[-1] + self.m < n:
            starts.append(starts[-1] + self.stride)
        return starts

    def transform_numpy(self, v: np.ndarray, coeffs: np.ndarray, exps: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        n = int(v.shape[0])
        m = self.m
        starts = self._starts(n)
        need = starts[-1] + m
        if need > n:
            v = np.concatenate([v, np.zeros(need - n, dtype=np.float64)])
        out = []
        for st in starts:
            window = v[st : st + m]
            acc = 0.0
            for i in range(m):
                x = float(window[i])
                e = int(exps[i])
                c = float(coeffs[i])
                acc += c * x**e
            out.append(acc)
        return np.asarray(out, dtype=np.float64)

    def transform_torch(self, emb: torch.Tensor, coeffs: np.ndarray, exps: np.ndarray) -> torch.Tensor:
        """Differentiable single-vector transform. emb: (D,)."""
        v = emb.reshape(-1)
        n = int(v.numel())
        m = self.m
        starts = self._starts(n)
        need = starts[-1] + m
        if need > n:
            v = torch.cat([v, torch.zeros(need - n, device=v.device, dtype=v.dtype)])
        c = torch.as_tensor(coeffs, device=v.device, dtype=v.dtype)
        e = torch.as_tensor(exps, device=v.device)
        outs = []
        for st in starts:
            window = v[st : st + m]
            outs.append(torch.sum(c * torch.pow(window, e)))
        out = torch.stack(outs)
        return F.normalize(out, p=2, dim=0)

    def transform(self, emb: torch.Tensor, coeffs: np.ndarray, exps: np.ndarray) -> torch.Tensor:
        """emb: (D,) or (N,D) → L2-normalized protected templates."""
        if emb.ndim == 1:
            return self.transform_torch(emb, coeffs, exps)
        return torch.stack([self.transform_torch(emb[i], coeffs, exps) for i in range(emb.shape[0])])
