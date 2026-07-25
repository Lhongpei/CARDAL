# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

"""Test Model.set_problem / set_problem_coo with numpy + scipy inputs.

The reference problem is the smallest-eigenvalue SDP for a symmetric matrix H:

    min   <H, X>       s.t.   trace(X) = 1,  X in S^n_+

whose optimal value equals lambda_min(H). We use a 3x3 diagonal H so the
analytical answer is trivial to check without depending on a solver-specific
tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

import cardal
from cardal import Model


def _smallest_eig_sdp(H):
    """Return (block_dims, C, A, b) for min <H, X> s.t. trace(X)=1, X psd."""
    n = H.shape[0]
    block_dims = [n]
    C = [H]                        # objective
    # Single constraint: trace(X) = 1
    A = [[np.eye(n)]]               # one constraint × one block
    b = [1.0]
    return block_dims, C, A, b


def test_set_problem_dense_smallest_eig():
    H = np.diag([3.0, 1.0, 2.0])   # lambda_min = 1.0
    block_dims, C, A, b = _smallest_eig_sdp(H)

    m = Model()
    m.set_problem(block_dims=block_dims, C=C, A=A, b=b)
    assert m.num_cones == 1
    assert m.num_constraints == 1
    assert m.block_dims == [3]

    result = m.solve(
        time_sec_limit=30.0,
        eps_optimal_relative=1e-6,
        eps_primal_relative=1e-6,
        eps_dual_relative=1e-6,
        l_inf_ruiz_iterations=0,
        pock_chambolle_rescaling=False,
        pock_chambolle_alpha=0.5,
        bound_objective_rescaling=False,
        psd_scale_mode="per-cone",
        verbose=0,
    )
    assert result.status is cardal.OPTIMAL, f"got {result.status}"
    # Primal objective should equal lambda_min(H) = 1.0
    assert abs(result.primal_objective - 1.0) < 1e-3, \
        f"got primal={result.primal_objective:.6f}"


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("l_inf_ruiz_iterations", -1, "must be nonnegative"),
        ("pock_chambolle_alpha", -1.0, "must be nonnegative"),
        ("psd_scale_mode", "diagonal", "per-element.*per-cone"),
    ],
)
def test_invalid_scaling_parameter_raises(parameter, value, message):
    m = Model()
    m.set_problem(
        block_dims=[2],
        C=[np.eye(2)],
        A=[[np.eye(2)]],
        b=[1.0],
    )
    with pytest.raises(ValueError, match=message):
        m.solve(**{parameter: value})


def test_set_problem_sparse_smallest_eig():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    H = scipy_sparse.diags([3.0, 1.0, 2.0], 0, format="csr")
    block_dims = [3]
    C = [H]
    A = [[scipy_sparse.eye(3, format="csr")]]
    b = [1.0]

    m = Model()
    m.set_problem(block_dims=block_dims, C=C, A=A, b=b)
    result = m.solve(
        time_sec_limit=30.0,
        eps_optimal_relative=1e-6,
        eps_primal_relative=1e-6,
        eps_dual_relative=1e-6,
        verbose=0,
    )
    assert result.status is cardal.OPTIMAL, f"got {result.status}"
    assert abs(result.primal_objective - 1.0) < 1e-3


def test_set_problem_coo_smallest_eig():
    # Same problem via the low-level COO entry.
    # H = diag(3, 1, 2); constraint: trace(X) = 1.
    m = Model()
    m.set_problem_coo(
        block_dims=[3],
        b=[1.0],
        # C: lower triangle only. Diagonal only.
        C=(
            np.array([0, 0, 0], dtype=np.int32),          # cone_ind
            np.array([0, 1, 2], dtype=np.int32),          # row_ind
            np.array([0, 1, 2], dtype=np.int32),          # col_ind
            np.array([3.0, 1.0, 2.0], dtype=np.float64),  # val
        ),
        # A: one constraint (index 0), block 0, diagonal ones.
        A=(
            np.array([0, 0, 0], dtype=np.int32),          # constr_ind
            np.array([0, 0, 0], dtype=np.int32),          # cone_ind
            np.array([0, 1, 2], dtype=np.int32),          # row_ind
            np.array([0, 1, 2], dtype=np.int32),          # col_ind
            np.array([1.0, 1.0, 1.0], dtype=np.float64),  # val
        ),
    )
    result = m.solve(
        time_sec_limit=30.0,
        eps_optimal_relative=1e-6,
        eps_primal_relative=1e-6,
        eps_dual_relative=1e-6,
        verbose=0,
    )
    assert result.status is cardal.OPTIMAL, f"got {result.status}"
    assert abs(result.primal_objective - 1.0) < 1e-3


def test_native_free_variables():
    # min x_0 - 2 x_1, subject to 2 x_0 = 2 and 0.5 x_1 = -1.
    # The unique solution is x = (1, -2), with objective 5.
    m = Model()
    m.set_problem(
        block_dims=[],
        C=[],
        A=[[], []],
        b=[2.0, -1.0],
        free_dim=2,
        free_obj=[1.0, -2.0],
        A_free=np.array([[2.0, 0.0], [0.0, 0.5]]),
    )
    assert m.num_cones == 0
    assert m.lp_dim == 0
    assert m.free_dim == 2

    result = m.solve(
        time_sec_limit=10.0,
        inner_iterations_limit=1000,
        eps_optimal_relative=1e-7,
        eps_primal_relative=1e-7,
        eps_dual_relative=1e-7,
        verbose=0,
    )
    assert result.status is cardal.OPTIMAL
    np.testing.assert_allclose(result.free_primal, [1.0, -2.0], atol=1e-6)
    assert result.primal_factor.size == 0
    assert result.lp_primal.size == 0
    assert abs(result.primal_objective - 5.0) < 1e-6
    assert result.rel_primal_residual < 1e-7
    assert result.rel_dual_residual < 1e-7


def test_mixed_psd_lp_and_free_variables():
    # min <diag(1, 2), X> + x_lp + 3 x_free
    # s.t. trace(X)=1, x_lp=2, x_free=-1.
    # The optimum is diag(1, 0), x_lp=2, x_free=-1, with objective 0.
    m = Model()
    m.set_problem(
        block_dims=[2],
        C=[np.diag([1.0, 2.0])],
        A=[
            [np.eye(2)],
            [None],
            [None],
        ],
        b=[1.0, 2.0, -1.0],
        lp_dim=1,
        lp_obj=[1.0],
        A_lp=np.array([[0.0], [1.0], [0.0]]),
        free_dim=1,
        free_obj=[3.0],
        A_free=np.array([[0.0], [0.0], [1.0]]),
    )

    result = m.solve(
        time_sec_limit=20.0,
        eps_optimal_relative=1e-6,
        eps_primal_relative=1e-6,
        eps_dual_relative=1e-6,
        verbose=0,
    )
    assert result.status is cardal.OPTIMAL
    assert result.primal_factor.size == 2 * int(result.rank_list[0])
    np.testing.assert_allclose(result.lp_primal, [2.0], atol=1e-5)
    np.testing.assert_allclose(result.free_primal, [-1.0], atol=1e-5)
    assert abs(result.primal_objective) < 1e-4


def test_free_objective_is_required():
    m = Model()
    with pytest.raises(ValueError, match="free_obj required"):
        m.set_problem(
            block_dims=[],
            C=[],
            A=[[]],
            b=[0.0],
            free_dim=1,
            A_free=np.ones((1, 1)),
        )


def test_set_problem_shape_mismatch_raises():
    m = Model()
    with pytest.raises(ValueError, match="len\\(C\\)="):
        m.set_problem(
            block_dims=[3],
            C=[np.eye(3), np.eye(2)],   # 2 blocks but only 1 declared
            A=[[np.eye(3)]],
            b=[1.0],
        )


def test_set_problem_solve_and_reuse():
    """Verify that set_problem is idempotent and swappable with read_file."""
    H1 = np.diag([2.0, 5.0])   # lambda_min = 2
    H2 = np.diag([7.0, 4.0])   # lambda_min = 4

    m = Model()
    m.set_problem(block_dims=[2], C=[H1], A=[[np.eye(2)]], b=[1.0])
    r1 = m.solve(time_sec_limit=15.0, eps_optimal_relative=1e-6,
                 eps_primal_relative=1e-6, eps_dual_relative=1e-6,
                 verbose=0)
    assert r1.status is cardal.OPTIMAL
    assert abs(r1.primal_objective - 2.0) < 1e-3

    # Swap in a different problem.
    m.set_problem(block_dims=[2], C=[H2], A=[[np.eye(2)]], b=[1.0])
    r2 = m.solve(time_sec_limit=15.0, eps_optimal_relative=1e-6,
                 eps_primal_relative=1e-6, eps_dual_relative=1e-6,
                 verbose=0)
    assert r2.status is cardal.OPTIMAL
    assert abs(r2.primal_objective - 4.0) < 1e-3
    # Original result is untouched (frozen dataclass).
    assert abs(r1.primal_objective - 2.0) < 1e-3
