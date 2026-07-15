# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

"""Public enum types for CARDAL results."""

from __future__ import annotations

from enum import IntEnum

from . import _core


class TerminationReason(IntEnum):
    """Why the solver stopped. Mirror of the C `cardal_status` enum.

    Members are re-exported at the package top level for convenience:
    ``cardal.OPTIMAL``, ``cardal.TIME_LIMIT``, etc.

    ``USER_INTERRUPT`` is returned by the C solver when the cooperative
    cancel flag is set (e.g. via Ctrl-C from the Python binding, which
    additionally re-raises ``KeyboardInterrupt`` — so users typically do
    not observe this value directly in Python).
    """

    UNSPECIFIED = int(_core.Status.UNSPECIFIED)
    OPTIMAL = int(_core.Status.OPTIMAL)
    TIME_LIMIT = int(_core.Status.TIME_LIMIT)
    ITERATION_LIMIT = int(_core.Status.ITERATION_LIMIT)
    USER_INTERRUPT = int(_core.Status.USER_INTERRUPT)

    def __str__(self) -> str:
        return self.name
