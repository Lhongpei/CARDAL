# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

"""CARDAL — a curvature-aware rank-adaptive distributed ALM solver for SDPs.

A tiny, user-facing wrapper over the C/CUDA solver.

Quick start:

    >>> import cardal
    >>> m = cardal.Model()
    >>> m.read_file("fe4s4_sos.dat-s")
    >>> result = m.solve(time_sec_limit=5.0)
    >>> print(result.status, result.primal_objective)
"""

from __future__ import annotations

from .enums import TerminationReason
from .model import Model
from .result import Result

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "Model",
    "Result",
    "TerminationReason",
    # Convenience re-exports of the most common status codes at package top:
    "OPTIMAL",
    "TIME_LIMIT",
    "ITERATION_LIMIT",
    "USER_INTERRUPT",
    "UNSPECIFIED",
]

OPTIMAL = TerminationReason.OPTIMAL
TIME_LIMIT = TerminationReason.TIME_LIMIT
ITERATION_LIMIT = TerminationReason.ITERATION_LIMIT
USER_INTERRUPT = TerminationReason.USER_INTERRUPT
UNSPECIFIED = TerminationReason.UNSPECIFIED
