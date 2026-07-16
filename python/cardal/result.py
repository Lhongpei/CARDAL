# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

"""Frozen ``Result`` dataclass produced by ``cardal.solve*``.

Design goals (contra PDHCG-II ``Model``):
    * Immutable — cannot go stale as the solver state changes.
    * Picklable / dataclass-friendly — round-trips through ``dataclasses.asdict``.
    * Zero-copy numpy views over the underlying C buffers when possible.
    * Owns the underlying C ``_core.Result`` for its lifetime so the numpy
      views remain valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .enums import TerminationReason


@dataclass(frozen=True)
class Result:
    """Solver output. All numpy arrays are read-only views over the C buffers
    owned by the underlying handle; ``Result`` keeps that handle alive.

    Attributes
    ----------
    status
        Termination reason (:class:`TerminationReason`).
    primal_objective, dual_objective
        Objective values at the final iterate.
    objective_gap
        Absolute primal-dual gap.
    rel_primal_residual, rel_dual_residual, rel_objective_gap
        Relative primal / dual residual and duality gap (the three quantities
        compared against ``eps_primal_relative``, ``eps_dual_relative``, and
        ``eps_optimal_relative``, respectively).
    runtime_sec
        Solver wall-clock, excluding setup and post-solve unscaling.
    outer_iters, inner_iters
        Outer ALM iteration count and total inner-LBFGS iterations.
    num_cones, num_variables, num_constraints
        Problem-shape metadata.
    total_rank
        Sum of per-cone Burer-Monteiro ranks at termination.
    primal_factor
        Flattened low-rank primal factor. Length = sum(dim_c * rank_c).
    dual
        Dual solution y. Length = num_constraints.
    rank_list
        Per-cone final BM rank.

    The final field ``_handle`` is a Pythonic strong reference to the
    underlying C ``_core.Result``; users should not touch it, but it must
    live as long as any of the numpy views above are in use.
    """

    status: TerminationReason
    primal_objective: float
    dual_objective: float
    objective_gap: float
    rel_primal_residual: float
    rel_dual_residual: float
    rel_objective_gap: float
    runtime_sec: float
    outer_iters: int
    inner_iters: int
    num_cones: int
    num_variables: int
    num_constraints: int
    total_rank: int
    primal_factor: np.ndarray
    dual: np.ndarray
    rank_list: np.ndarray
    _handle: object = field(repr=False)

    # ---- pretty printing ------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Result(status={self.status.name}, "
            f"primal_obj={self.primal_objective:.6e}, "
            f"gap={self.rel_objective_gap:.2e}, "
            f"pres={self.rel_primal_residual:.2e}, "
            f"dres={self.rel_dual_residual:.2e}, "
            f"rank={self.total_rank}, "
            f"outer={self.outer_iters}, "
            f"inner={self.inner_iters}, "
            f"time={self.runtime_sec:.2f}s)"
        )

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        return (
            f"CARDAL result\n"
            f"  status              : {self.status.name}\n"
            f"  runtime (sec)       : {self.runtime_sec:.2f}\n"
            f"  outer / inner iters : {self.outer_iters} / {self.inner_iters}\n"
            f"  primal objective    : {self.primal_objective:.6e}\n"
            f"  dual   objective    : {self.dual_objective:.6e}\n"
            f"  rel primal residual : {self.rel_primal_residual:.2e}\n"
            f"  rel dual   residual : {self.rel_dual_residual:.2e}\n"
            f"  rel objective gap   : {self.rel_objective_gap:.2e}\n"
            f"  total BM rank       : {self.total_rank}\n"
        )


def _make_result(core_result: object) -> Result:
    """Build a frozen :class:`Result` from a ``cardal._core.Result`` handle.

    Zero-copy numpy views are pulled off the handle; the handle itself is
    stashed on the dataclass as ``_handle`` to keep its underlying buffers
    alive.
    """
    # The numpy arrays returned by the pybind11 bindings already carry the
    # core_result as their `base`, so they are safe on their own — but we
    # still stash a strong reference to belt-and-suspenders it against future
    # changes to the binding.
    return Result(
        status=TerminationReason(int(core_result.status)),
        primal_objective=float(core_result.primal_objective),
        dual_objective=float(core_result.dual_objective),
        objective_gap=float(core_result.objective_gap),
        rel_primal_residual=float(core_result.rel_primal_residual),
        rel_dual_residual=float(core_result.rel_dual_residual),
        rel_objective_gap=float(core_result.rel_objective_gap),
        runtime_sec=float(core_result.runtime_sec),
        outer_iters=int(core_result.outer_iters),
        inner_iters=int(core_result.inner_iters),
        num_cones=int(core_result.num_cones),
        num_variables=int(core_result.num_variables),
        num_constraints=int(core_result.num_constraints),
        total_rank=int(core_result.total_rank),
        primal_factor=np.asarray(core_result.primal_factor()),
        dual=np.asarray(core_result.dual()),
        rank_list=np.asarray(core_result.rank_list()),
        _handle=core_result,
    )
