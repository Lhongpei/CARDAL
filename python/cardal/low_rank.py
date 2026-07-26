# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

"""Structured symmetric matrix inputs for CARDAL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class LowRank:
    """A symmetric matrix represented as ``U diag(weights) U.T``.

    ``core`` may be supplied instead of ``weights`` to represent ``U D U.T``.
    The small symmetric matrix ``D`` is diagonalized once during construction.
    Signed weights are supported, so the represented matrix need not be PSD.
    """

    factors: np.ndarray
    weights: np.ndarray

    def __init__(
        self,
        factors: Any,
        weights: Optional[Any] = None,
        *,
        core: Optional[Any] = None,
        tolerance: float = 0.0,
    ) -> None:
        U = np.ascontiguousarray(factors, dtype=np.float64)
        if U.ndim != 2:
            raise ValueError("factors must be a two-dimensional array")
        if not np.all(np.isfinite(U)):
            raise ValueError("factors must contain only finite values")
        if core is not None and weights is not None:
            raise ValueError("provide either weights or core, not both")

        if core is not None:
            D = np.ascontiguousarray(core, dtype=np.float64)
            rank = U.shape[1]
            if D.shape != (rank, rank):
                raise ValueError(
                    f"core shape {D.shape} does not match ({rank}, {rank})"
                )
            if not np.all(np.isfinite(D)):
                raise ValueError("core must contain only finite values")
            if not np.allclose(D, D.T, rtol=1e-12, atol=1e-14):
                raise ValueError("core must be symmetric")
            cutoff = float(tolerance)
            if not np.isfinite(cutoff) or cutoff < 0.0:
                raise ValueError("tolerance must be finite and nonnegative")
            eigvals, eigvecs = np.linalg.eigh(D)
            keep = np.abs(eigvals) > cutoff
            U = np.ascontiguousarray(U @ eigvecs[:, keep], dtype=np.float64)
            d = np.ascontiguousarray(eigvals[keep], dtype=np.float64)
        else:
            cutoff = float(tolerance)
            if not np.isfinite(cutoff) or cutoff < 0.0:
                raise ValueError("tolerance must be finite and nonnegative")
            if weights is None:
                d = np.ones(U.shape[1], dtype=np.float64)
            else:
                d = np.ascontiguousarray(weights, dtype=np.float64)
                if d.ndim != 1 or d.size != U.shape[1]:
                    raise ValueError(
                        "weights must be one-dimensional with one entry per "
                        "factor column"
                    )
                if not np.all(np.isfinite(d)):
                    raise ValueError("weights must contain only finite values")
            if cutoff > 0.0:
                keep = np.abs(d) > cutoff
                U = np.ascontiguousarray(U[:, keep], dtype=np.float64)
                d = np.ascontiguousarray(d[keep], dtype=np.float64)

        object.__setattr__(self, "factors", U)
        object.__setattr__(self, "weights", d)

    @property
    def rank(self) -> int:
        return int(self.factors.shape[1])


@dataclass(frozen=True)
class SparseLowRank:
    """A symmetric matrix represented as sparse/dense data plus ``LowRank``."""

    sparse: Any
    low_rank: LowRank

    def __init__(
        self,
        sparse: Any,
        factors: Any,
        weights: Optional[Any] = None,
        *,
        core: Optional[Any] = None,
        tolerance: float = 0.0,
    ) -> None:
        object.__setattr__(self, "sparse", sparse)
        object.__setattr__(
            self,
            "low_rank",
            LowRank(
                factors,
                weights,
                core=core,
                tolerance=tolerance,
            ),
        )
