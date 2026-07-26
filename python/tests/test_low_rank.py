# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

"""Regression tests for signed low-rank objective and constraint data."""

from __future__ import annotations

import numpy as np
import pytest

import cardal


SOLVE_PARAMS = {
    "time_sec_limit": 15.0,
    "iteration_limit": 2000,
    "initial_rank": 3,
    "max_rank": 3,
    "eps_optimal_relative": 1e-5,
    "eps_primal_relative": 1e-6,
    "eps_dual_relative": 1e-5,
    "verbose": 0,
}


def _solve(cost, constraint, *, scaling):
    model = cardal.Model()
    model.set_problem(
        block_dims=[3],
        b=[1.0],
        C=[cost],
        A=[[constraint]],
    )
    params = dict(SOLVE_PARAMS)
    if not scaling:
        params.update(
            l_inf_ruiz_iterations=0,
            pock_chambolle_rescaling=False,
            bound_objective_rescaling=False,
        )
    return model.solve(**params)


@pytest.mark.parametrize("scaling", [False, True])
def test_signed_low_rank_objective_matches_explicit_matrix(scaling):
    u = np.array([[1.0], [2.0], [-1.0]])
    explicit = -(u @ u.T)
    dense = _solve(explicit, np.eye(3), scaling=scaling)
    structured = _solve(
        cardal.LowRank(u, weights=[-1.0]),
        np.eye(3),
        scaling=scaling,
    )

    assert dense.status is cardal.OPTIMAL
    assert structured.status is cardal.OPTIMAL
    assert structured.primal_objective == pytest.approx(
        dense.primal_objective, abs=1e-4
    )
    assert structured.rel_primal_residual < 1e-5


def test_low_rank_constraint_matches_explicit_matrix():
    u = np.array([[1.0], [2.0], [-1.0]])
    cost = -(u @ u.T)
    dense = _solve(cost, np.eye(3), scaling=False)
    structured = _solve(cost, cardal.LowRank(np.eye(3)), scaling=False)

    assert structured.status is cardal.OPTIMAL
    assert structured.primal_objective == pytest.approx(
        dense.primal_objective, abs=1e-4
    )
    assert structured.rel_primal_residual < 1e-5


def test_sparse_plus_indefinite_core_matches_explicit_matrix():
    factors = np.array([[1.0, 0.0], [2.0, 1.0], [-1.0, 2.0]])
    core = np.array([[-1.2, 0.35], [0.35, 0.4]])
    sparse = np.diag([0.2, 0.4, 0.6])
    explicit = sparse + factors @ core @ factors.T
    structured_cost = cardal.SparseLowRank(sparse, factors, core=core, tolerance=1e-14)

    dense = _solve(explicit, np.eye(3), scaling=True)
    structured = _solve(structured_cost, cardal.LowRank(np.eye(3)), scaling=True)

    assert structured.status is cardal.OPTIMAL
    assert structured.primal_objective == pytest.approx(
        dense.primal_objective, abs=1e-4
    )


def test_low_rank_validation():
    with pytest.raises(ValueError, match="two-dimensional"):
        cardal.LowRank([1.0, 2.0])
    with pytest.raises(ValueError, match="symmetric"):
        cardal.LowRank(np.eye(2), core=[[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        cardal.LowRank([[1.0], [np.inf]])
    with pytest.raises(ValueError, match="finite"):
        cardal.LowRank(np.eye(2), weights=[1.0, np.nan])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        cardal.LowRank(np.eye(2), tolerance=np.inf)


def test_low_rank_tolerance_drops_columns():
    low_rank = cardal.LowRank(
        np.eye(3),
        weights=[1.0, 1e-10, -2.0],
        tolerance=1e-8,
    )
    assert low_rank.rank == 2
    np.testing.assert_allclose(low_rank.weights, [1.0, -2.0])


def test_batched_low_rank_cones_match_explicit_matrices():
    num_cones = 16  # CARDAL_MIN_BATCH_SIZE
    dim = 3
    weights = np.array([-1.0, 0.2, 0.4])
    identity = np.eye(dim)

    dense_model = cardal.Model()
    dense_model.set_problem(
        block_dims=[dim] * num_cones,
        b=np.ones(num_cones),
        C=[np.diag(weights) for _ in range(num_cones)],
        A=[
            [identity if i == cone else None for cone in range(num_cones)]
            for i in range(num_cones)
        ],
    )

    structured_model = cardal.Model()
    structured_model.set_problem(
        block_dims=[dim] * num_cones,
        b=np.ones(num_cones),
        C=[cardal.LowRank(identity, weights=weights) for _ in range(num_cones)],
        A=[
            [
                cardal.LowRank(identity) if i == cone else None
                for cone in range(num_cones)
            ]
            for i in range(num_cones)
        ],
    )

    dense = dense_model.solve(**SOLVE_PARAMS)
    structured = structured_model.solve(**SOLVE_PARAMS)

    assert dense.status is cardal.OPTIMAL
    assert structured.status is cardal.OPTIMAL
    assert structured.primal_objective == pytest.approx(
        dense.primal_objective, abs=2e-4
    )
    assert structured.rel_primal_residual < 1e-5


def test_multiple_percone_pure_low_rank_cones():
    num_cones = 2  # Below CARDAL_MIN_BATCH_SIZE.
    identity = np.eye(3)
    model = cardal.Model()
    model.set_problem(
        block_dims=[3] * num_cones,
        b=np.ones(num_cones),
        C=[
            cardal.LowRank(identity, weights=[-1.0, 0.2, 0.4]) for _ in range(num_cones)
        ],
        A=[
            [
                cardal.LowRank(identity) if row == cone else None
                for cone in range(num_cones)
            ]
            for row in range(num_cones)
        ],
    )

    result = model.solve(**SOLVE_PARAMS)

    assert result.status is cardal.OPTIMAL
    assert result.primal_objective == pytest.approx(-2.0, abs=2e-4)
    assert result.rel_primal_residual < 1e-5


def test_batched_fixed_rank_curvature_path():
    num_cones = 32
    identity = np.eye(3)
    model = cardal.Model()
    model.set_problem(
        block_dims=[3] * num_cones,
        b=[float(num_cones)],
        C=[
            cardal.LowRank(identity, weights=[-1.0, 0.2, 0.4]) for _ in range(num_cones)
        ],
        A=[[cardal.LowRank(identity) for _ in range(num_cones)]],
    )
    params = dict(SOLVE_PARAMS)
    params.update(
        iteration_limit=30,
        inner_iterations_limit=20,
        initial_rank=1,
        max_rank=1,
        l_inf_ruiz_iterations=0,
        pock_chambolle_rescaling=False,
        bound_objective_rescaling=False,
    )

    result = model.solve(**params)

    assert result.status is cardal.OPTIMAL
    assert result.primal_objective == pytest.approx(-num_cones, abs=5e-3)
    assert result.rel_primal_residual < 1e-4
