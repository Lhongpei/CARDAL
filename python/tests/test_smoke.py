# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

"""End-to-end smoke: optionally load fe4s4_sos.dat-s and run a tiny solve.

Set CARDAL_FE4S4_FIXTURE to the absolute path of fe4s4_sos.dat-s to enable
the fixture-backed tests. They skip gracefully otherwise.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

import cardal
from cardal import Model


FE4S4 = os.environ.get("CARDAL_FE4S4_FIXTURE", "")
FE4S4_SKIP_REASON = (
    "set CARDAL_FE4S4_FIXTURE to an fe4s4_sos.dat-s path to enable this test"
    if not FE4S4
    else f"benchmark fixture missing: {FE4S4}"
)


@pytest.mark.skipif(not os.path.exists(FE4S4),
                    reason=FE4S4_SKIP_REASON)
def test_solve_fe4s4():
    m = Model()
    m.read_file(FE4S4)
    result = m.solve(
        time_sec_limit=5.0,
        eps_optimal_relative=1e-2,
        eps_primal_relative=1e-2,
        eps_dual_relative=1e-2,
        verbose=0,
    )
    assert isinstance(result, cardal.Result)
    assert result.status in (
        cardal.OPTIMAL, cardal.TIME_LIMIT, cardal.ITERATION_LIMIT
    )
    assert result.num_cones == 3           # fe4s4 has 3 PSD blocks
    assert result.num_constraints > 0
    assert result.rank_list.dtype == np.int32
    assert result.rank_list.shape == (3,)
    assert result.dual.dtype == np.float64
    assert result.primal_factor.dtype == np.float64
    # Total rank should equal sum(rank_list) since fe4s4 has no LP block.
    assert result.total_rank == int(result.rank_list.sum())


def test_default_params_shape():
    p = Model.default_params()
    for key in ("eps_optimal_relative", "eps_primal_relative",
                "eps_dual_relative",
                "time_sec_limit", "iteration_limit", "augmentation_mode",
                "l_inf_ruiz_iterations", "pock_chambolle_rescaling",
                "pock_chambolle_alpha", "bound_objective_rescaling",
                "psd_scale_mode",
                "verbose"):
        assert key in p
    assert p["eps_primal_relative"] == 1e-4
    assert p["eps_dual_relative"] == 1e-4
    assert p["penalty_factor"] == 3.3
    assert p["augmentation_mode"] == "random"
    assert p["l_inf_ruiz_iterations"] == 10
    assert p["pock_chambolle_rescaling"] is True
    assert p["pock_chambolle_alpha"] == 1.0
    assert p["bound_objective_rescaling"] is True
    assert p["psd_scale_mode"] == "per-element"


def test_solve_before_read_file_raises():
    m = Model()
    with pytest.raises(RuntimeError, match="no problem loaded"):
        m.solve()


def test_unknown_param_raises():
    if not os.path.exists(FE4S4):
        pytest.skip(FE4S4_SKIP_REASON)
    m = Model()
    m.read_file(FE4S4)
    with pytest.raises(TypeError, match="unknown parameter"):
        m.solve(definitely_not_a_real_parameter=42)


@pytest.mark.skipif(not os.path.exists(FE4S4),
                    reason=FE4S4_SKIP_REASON)
def test_cancel_raises_keyboard_interrupt():
    """Set the cancel flag from a helper thread mid-solve; solve() should
    return promptly with KeyboardInterrupt (mirroring what Ctrl-C would do
    at the SIGINT handler level)."""
    import threading
    import time

    from cardal import _core

    m = Model()
    m.read_file(FE4S4)

    def request_cancel_after_delay():
        time.sleep(0.5)
        _core.request_cancel()

    canceller = threading.Thread(target=request_cancel_after_delay,
                                  daemon=True)
    canceller.start()

    t0 = time.time()
    with pytest.raises(KeyboardInterrupt):
        m.solve(
            time_sec_limit=60.0,   # long budget so cancel is what stops us
            eps_optimal_relative=1e-12,
            verbose=0,
        )
    elapsed = time.time() - t0
    canceller.join(timeout=2.0)

    # Should have terminated well before the 60s budget.
    assert elapsed < 15.0, f"cancel took too long: {elapsed:.1f}s"
    assert not _core.cancel_requested(), "cancel flag should be cleared after solve"
