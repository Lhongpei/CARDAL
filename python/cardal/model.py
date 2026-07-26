# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

"""CARDAL Model — the user-facing entry point.

Typical usage::

    import cardal
    m = cardal.Model.read_file("problem.dat-s")
    result = m.solve(time_sec_limit=60.0, eps_primal_relative=1e-4,
                     eps_dual_relative=1e-4, eps_optimal_relative=1e-4)
    print(result.summary())
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union
import os

import numpy as np

from . import _core
from .low_rank import LowRank, SparseLowRank
from .result import Result, _make_result


def _hstack_ints(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(chunks).astype(np.int32, copy=False)


def _hstack_floats(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64, copy=False)


def _unpack_coo(t: Tuple, expected_arity: int, name: str):
    if len(t) != expected_arity:
        raise ValueError(
            f"{name} must be a tuple of {expected_arity} arrays, got {len(t)}"
        )
    arrs = []
    for i, a in enumerate(t):
        dtype = np.float64 if i == expected_arity - 1 else np.int32
        arrs.append(np.ascontiguousarray(a, dtype=dtype))
    sizes = [a.size for a in arrs]
    if len(set(sizes)) > 1:
        raise ValueError(f"{name}: parallel arrays have inconsistent lengths {sizes}")
    return arrs


def _matrix_to_lower_coo(mat, dim: int, label: str):
    """Convert a symmetric dense/sparse matrix to lower-triangular COO.

    Returns (row_ind, col_ind, val) as int32/int32/float64 arrays.
    Off-diagonal entries above the diagonal are dropped (the matrix is
    assumed symmetric).
    """
    # scipy.sparse duck-typing without importing scipy at module level.
    if hasattr(mat, "tocoo"):
        coo = mat.tocoo()
        rows = np.asarray(coo.row, dtype=np.int32)
        cols = np.asarray(coo.col, dtype=np.int32)
        vals = np.asarray(coo.data, dtype=np.float64)
        if coo.shape != (dim, dim):
            raise ValueError(
                f"{label}: sparse matrix shape {coo.shape} != ({dim}, {dim})"
            )
    else:
        arr = np.ascontiguousarray(mat, dtype=np.float64)
        if arr.shape != (dim, dim):
            raise ValueError(
                f"{label}: dense matrix shape {arr.shape} != ({dim}, {dim})"
            )
        rows_all, cols_all = np.nonzero(arr)
        rows = rows_all.astype(np.int32)
        cols = cols_all.astype(np.int32)
        vals = arr[rows_all, cols_all].astype(np.float64)
    # Keep only the lower triangle (row >= col).
    mask = rows >= cols
    return rows[mask], cols[mask], vals[mask]


def _linear_matrix_to_coo(mat, num_constraints: int, variable_dim: int, label: str):
    if hasattr(mat, "tocoo"):
        coo = mat.tocoo()
        if coo.shape != (num_constraints, variable_dim):
            raise ValueError(
                f"{label} shape {coo.shape} != ({num_constraints}, {variable_dim})"
            )
        return (
            np.asarray(coo.row, dtype=np.int32),
            np.asarray(coo.col, dtype=np.int32),
            np.asarray(coo.data, dtype=np.float64),
        )
    arr = np.ascontiguousarray(mat, dtype=np.float64)
    if arr.shape != (num_constraints, variable_dim):
        raise ValueError(
            f"{label} shape {arr.shape} != ({num_constraints}, {variable_dim})"
        )
    rows, cols = np.nonzero(arr)
    return (
        rows.astype(np.int32),
        cols.astype(np.int32),
        arr[rows, cols].astype(np.float64),
    )


def _split_matrix_input(mat, dim: int, label: str):
    sparse = mat
    low_rank = None
    if isinstance(mat, SparseLowRank):
        sparse = mat.sparse
        low_rank = mat.low_rank
    elif isinstance(mat, LowRank):
        sparse = None
        low_rank = mat

    if low_rank is not None and low_rank.factors.shape[0] != dim:
        raise ValueError(
            f"{label}: low-rank factor has {low_rank.factors.shape[0]} rows, "
            f"expected {dim}"
        )
    if sparse is None:
        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)
        vals = np.empty(0, dtype=np.float64)
    else:
        rows, cols, vals = _matrix_to_lower_coo(sparse, dim, label)
    return rows, cols, vals, low_rank


class Model:
    """A CARDAL SDP model.

    A ``Model`` is an empty shell after construction. Use :meth:`read_file`
    to construct a model from a problem file, or :meth:`load_file` to replace
    the problem held by an existing model.

    Results are returned as immutable :class:`cardal.Result` objects and
    are NOT stored on the ``Model``.
    """

    __slots__ = ("_problem",)

    def __init__(self) -> None:
        self._problem: Optional["_core.Problem"] = None

    @classmethod
    def read_file(cls, path: Union[str, os.PathLike]) -> "Model":
        """Construct a model from an SDPA / MATLAB / PDSDP file.

        Format is auto-detected from the file header.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file cannot be parsed.
        """
        model = cls()
        model.load_file(path)
        return model

    def load_file(self, path: Union[str, os.PathLike]) -> None:
        """Replace this model's problem with one loaded from a file.

        Format is auto-detected from the file header. Any previously loaded
        problem is discarded when the old underlying C handle is released.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file cannot be parsed.
        """
        path = os.fspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self._problem = _core.read_sdpa(path)

    def set_problem_coo(
        self,
        *,
        block_dims: Sequence[int],
        b,
        C: Tuple,
        A: Tuple,
        C_low_rank: Optional[Tuple] = None,
        A_low_rank: Optional[Tuple] = None,
        lp_dim: int = 0,
        lp_obj=None,
        A_lp: Optional[Tuple] = None,
        free_dim: int = 0,
        free_obj=None,
        A_free: Optional[Tuple] = None,
    ) -> None:
        """Set the problem directly from COO triplet arrays.

        This is the low-level entry that maps 1:1 to the C ABI. Prefer
        :meth:`set_problem` for the list-of-matrices form.

        Parameters
        ----------
        block_dims : sequence of int, length ``p``
            PSD block dimensions.
        b : array-like, length ``m``
            Right-hand-side vector.
        C : tuple ``(cone_ind, row_ind, col_ind, val)``
            COO triplets of the primal cost. Provide only one triangle per
            block (typically lower: ``row >= col``); off-diagonal entries
            are NOT doubled internally.
        A : tuple ``(constr_ind, cone_ind, row_ind, col_ind, val)``
            COO triplets of the constraint matrices. Same triangle rule as
            ``C``.
        C_low_rank : tuple ``(cone_ind, rank, factors, weights)`` or None
            Signed low-rank objective terms. Factor matrices are concatenated
            term by term in column-major order; weights are concatenated in
            the same order.
        A_low_rank : tuple ``(constr_ind, cone_ind, rank, factors, weights)``
            Signed low-rank constraint terms, packed like ``C_low_rank``.
        lp_dim : int
            Dimension of the optional nonnegative LP block. Default 0.
        lp_obj : array-like, length ``lp_dim``, or None
            LP-block objective. Required iff ``lp_dim > 0``.
        A_lp : tuple ``(constr_ind, col_ind, val)`` or None
            LP-block constraint triplets. Pass None for no LP constraints.
        free_dim : int
            Dimension of the optional unrestricted real block. Default 0.
        free_obj : array-like, length ``free_dim``, or None
            Free-block objective. Required iff ``free_dim > 0``.
        A_free : tuple ``(constr_ind, col_ind, val)`` or None
            Free-block constraint triplets.

        Raises
        ------
        ValueError
            If any array length is inconsistent with the declared sizes.
        """
        block_dims_arr = np.ascontiguousarray(block_dims, dtype=np.int32)
        b_arr = np.ascontiguousarray(b, dtype=np.float64)
        num_cones = int(block_dims_arr.size)
        num_constraints = int(b_arr.size)

        c_cone, c_row, c_col, c_val = _unpack_coo(C, 4, "C")
        a_constr, a_cone, a_row, a_col, a_val = _unpack_coo(A, 5, "A")
        if C_low_rank is None:
            c_lr_cone = np.empty(0, dtype=np.int32)
            c_lr_rank = np.empty(0, dtype=np.int32)
            c_lr_factors = np.empty(0, dtype=np.float64)
            c_lr_weights = np.empty(0, dtype=np.float64)
        else:
            if len(C_low_rank) != 4:
                raise ValueError(
                    "C_low_rank must be (cone_ind, rank, factors, weights)"
                )
            c_lr_cone = np.ascontiguousarray(C_low_rank[0], dtype=np.int32)
            c_lr_rank = np.ascontiguousarray(C_low_rank[1], dtype=np.int32)
            c_lr_factors = np.ascontiguousarray(C_low_rank[2], dtype=np.float64)
            c_lr_weights = np.ascontiguousarray(C_low_rank[3], dtype=np.float64)
        if A_low_rank is None:
            a_lr_constr = np.empty(0, dtype=np.int32)
            a_lr_cone = np.empty(0, dtype=np.int32)
            a_lr_rank = np.empty(0, dtype=np.int32)
            a_lr_factors = np.empty(0, dtype=np.float64)
            a_lr_weights = np.empty(0, dtype=np.float64)
        else:
            if len(A_low_rank) != 5:
                raise ValueError(
                    "A_low_rank must be (constr_ind, cone_ind, rank, factors, weights)"
                )
            a_lr_constr = np.ascontiguousarray(A_low_rank[0], dtype=np.int32)
            a_lr_cone = np.ascontiguousarray(A_low_rank[1], dtype=np.int32)
            a_lr_rank = np.ascontiguousarray(A_low_rank[2], dtype=np.int32)
            a_lr_factors = np.ascontiguousarray(A_low_rank[3], dtype=np.float64)
            a_lr_weights = np.ascontiguousarray(A_low_rank[4], dtype=np.float64)

        if lp_dim > 0:
            if lp_obj is None:
                raise ValueError("lp_obj required when lp_dim > 0")
            lp_obj_arr = np.ascontiguousarray(lp_obj, dtype=np.float64)
            if lp_obj_arr.size != lp_dim:
                raise ValueError(f"lp_obj length {lp_obj_arr.size} != lp_dim {lp_dim}")
        else:
            lp_obj_arr = np.empty(0, dtype=np.float64)

        if A_lp is not None:
            lp_constr, lp_col, lp_val = _unpack_coo(A_lp, 3, "A_lp")
        else:
            lp_constr = np.empty(0, dtype=np.int32)
            lp_col = np.empty(0, dtype=np.int32)
            lp_val = np.empty(0, dtype=np.float64)

        if free_dim > 0:
            if free_obj is None:
                raise ValueError("free_obj required when free_dim > 0")
            free_obj_arr = np.ascontiguousarray(free_obj, dtype=np.float64)
            if free_obj_arr.size != free_dim:
                raise ValueError(
                    f"free_obj length {free_obj_arr.size} != free_dim {free_dim}"
                )
        else:
            free_obj_arr = np.empty(0, dtype=np.float64)

        if A_free is not None:
            free_constr, free_col, free_val = _unpack_coo(A_free, 3, "A_free")
        else:
            free_constr = np.empty(0, dtype=np.int32)
            free_col = np.empty(0, dtype=np.int32)
            free_val = np.empty(0, dtype=np.float64)

        self._problem = _core.build_problem(
            num_constraints=num_constraints,
            num_cones=num_cones,
            lp_dim=int(lp_dim),
            free_dim=int(free_dim),
            blk_dims=block_dims_arr,
            c_cone_ind=c_cone,
            c_row_ind=c_row,
            c_col_ind=c_col,
            c_val=c_val,
            a_constr_ind=a_constr,
            a_cone_ind=a_cone,
            a_row_ind=a_row,
            a_col_ind=a_col,
            a_val=a_val,
            c_lr_cone_ind=c_lr_cone,
            c_lr_rank=c_lr_rank,
            c_lr_factors=c_lr_factors,
            c_lr_weights=c_lr_weights,
            a_lr_constr_ind=a_lr_constr,
            a_lr_cone_ind=a_lr_cone,
            a_lr_rank=a_lr_rank,
            a_lr_factors=a_lr_factors,
            a_lr_weights=a_lr_weights,
            lp_obj=lp_obj_arr,
            lp_constr_ind=lp_constr,
            lp_col_ind=lp_col,
            lp_val=lp_val,
            free_obj=free_obj_arr,
            free_constr_ind=free_constr,
            free_col_ind=free_col,
            free_val=free_val,
            b=b_arr,
        )

    def set_problem(
        self,
        *,
        block_dims: Sequence[int],
        b,
        C: Sequence,
        A: Sequence[Sequence],
        lp_dim: int = 0,
        lp_obj=None,
        A_lp=None,
        free_dim: int = 0,
        free_obj=None,
        A_free=None,
    ) -> None:
        """Set the problem from a list of numpy/scipy matrices.

        Convenient for small to medium problems. Internally converts every
        input matrix to COO triplets, then calls :meth:`set_problem_coo`.

        Parameters
        ----------
        block_dims : sequence of int, length ``p``
            PSD block dimensions.
        b : array-like, length ``m``
            Right-hand-side vector.
        C : sequence of ``p`` matrices
            Per-block primal cost matrices. Each entry may be a dense
            ``numpy.ndarray``, any ``scipy.sparse`` matrix,
            :class:`cardal.LowRank`, or :class:`cardal.SparseLowRank`.
            Explicit matrix data is assumed symmetric; only the lower
            triangle (``row >= col``) is stored.
        A : sequence of ``m`` rows of ``p`` matrices, i.e. ``A[i][k]`` is
            the ``k``-th block of the ``i``-th constraint. Each entry may
            use any form accepted by ``C`` or be ``None`` (treated as zero).
        lp_dim : int
            Dimension of the optional nonnegative LP block. Default 0.
        lp_obj : array-like, length ``lp_dim``, or None
            LP-block objective.
        A_lp : scipy.sparse matrix or ndarray of shape ``(m, lp_dim)`` or None
            LP-block constraint matrix. Rows index constraints, columns
            index LP variables.
        free_dim : int
            Dimension of the optional unrestricted real block. Default 0.
        free_obj : array-like, length ``free_dim``, or None
            Free-block objective.
        A_free : scipy.sparse matrix or ndarray of shape ``(m, free_dim)``
            Free-block constraint matrix.
        """
        block_dims = list(block_dims)
        num_cones = len(block_dims)
        b_arr = np.ascontiguousarray(b, dtype=np.float64)
        num_constraints = int(b_arr.size)

        if len(C) != num_cones:
            raise ValueError(
                f"len(C)={len(C)} does not match len(block_dims)={num_cones}"
            )
        if len(A) != num_constraints:
            raise ValueError(f"len(A)={len(A)} does not match len(b)={num_constraints}")

        # Build C-side COO by iterating over blocks.
        c_cone, c_row, c_col, c_val = [], [], [], []
        c_lr_cone, c_lr_rank, c_lr_factors, c_lr_weights = [], [], [], []
        for k, Ck in enumerate(C):
            rows, cols, vals, lr = _split_matrix_input(Ck, block_dims[k], f"C[{k}]")
            c_cone.append(np.full(rows.size, k, dtype=np.int32))
            c_row.append(rows)
            c_col.append(cols)
            c_val.append(vals)
            if lr is not None and lr.rank > 0:
                c_lr_cone.append(k)
                c_lr_rank.append(lr.rank)
                c_lr_factors.append(lr.factors.ravel(order="F"))
                c_lr_weights.append(lr.weights)
        c_cone = _hstack_ints(c_cone)
        c_row = _hstack_ints(c_row)
        c_col = _hstack_ints(c_col)
        c_val = _hstack_floats(c_val)

        # Build A-side COO by iterating over constraints and blocks.
        a_constr, a_cone, a_row, a_col, a_val = [], [], [], [], []
        a_lr_constr, a_lr_cone, a_lr_rank = [], [], []
        a_lr_factors, a_lr_weights = [], []
        for i, row in enumerate(A):
            if len(row) != num_cones:
                raise ValueError(f"A[{i}] has length {len(row)}, expected {num_cones}")
            for k, Aik in enumerate(row):
                if Aik is None:
                    continue
                rows, cols, vals, lr = _split_matrix_input(
                    Aik, block_dims[k], f"A[{i}][{k}]"
                )
                if rows.size > 0:
                    a_constr.append(np.full(rows.size, i, dtype=np.int32))
                    a_cone.append(np.full(rows.size, k, dtype=np.int32))
                    a_row.append(rows)
                    a_col.append(cols)
                    a_val.append(vals)
                if lr is not None and lr.rank > 0:
                    a_lr_constr.append(i)
                    a_lr_cone.append(k)
                    a_lr_rank.append(lr.rank)
                    a_lr_factors.append(lr.factors.ravel(order="F"))
                    a_lr_weights.append(lr.weights)
        a_constr = _hstack_ints(a_constr)
        a_cone = _hstack_ints(a_cone)
        a_row = _hstack_ints(a_row)
        a_col = _hstack_ints(a_col)
        a_val = _hstack_floats(a_val)

        # LP.
        A_lp_coo = None
        if lp_dim > 0 and A_lp is not None:
            lp_constr, lp_col, lp_val = _linear_matrix_to_coo(
                A_lp, num_constraints, lp_dim, "A_lp"
            )
            A_lp_coo = (lp_constr, lp_col, lp_val)
        A_free_coo = None
        if free_dim > 0 and A_free is not None:
            free_constr, free_col, free_val = _linear_matrix_to_coo(
                A_free, num_constraints, free_dim, "A_free"
            )
            A_free_coo = (free_constr, free_col, free_val)

        self.set_problem_coo(
            block_dims=block_dims,
            b=b_arr,
            C=(c_cone, c_row, c_col, c_val),
            A=(a_constr, a_cone, a_row, a_col, a_val),
            C_low_rank=(
                np.asarray(c_lr_cone, dtype=np.int32),
                np.asarray(c_lr_rank, dtype=np.int32),
                _hstack_floats(c_lr_factors),
                _hstack_floats(c_lr_weights),
            ),
            A_low_rank=(
                np.asarray(a_lr_constr, dtype=np.int32),
                np.asarray(a_lr_cone, dtype=np.int32),
                np.asarray(a_lr_rank, dtype=np.int32),
                _hstack_floats(a_lr_factors),
                _hstack_floats(a_lr_weights),
            ),
            lp_dim=lp_dim,
            lp_obj=lp_obj,
            A_lp=A_lp_coo,
            free_dim=free_dim,
            free_obj=free_obj,
            A_free=A_free_coo,
        )

    def solve(self, **params: Any) -> Result:
        """Solve the loaded problem.

        Any keyword argument is passed straight through to the solver as a
        parameter override; unknown keys raise ``TypeError`` (contra
        Gurobi-style silent-accept). See :meth:`default_params` for the
        supported parameter keys and their defaults.

        A running solve can be interrupted with Ctrl-C; the SIGINT handler
        installed by the C binding flips a cooperative cancel flag, the
        outer loop returns at its next boundary, and this method re-raises
        ``KeyboardInterrupt`` in Python.

        Raises
        ------
        RuntimeError
            If no problem has been loaded or constructed.
        TypeError
            If ``params`` contains a key not returned by
            :meth:`default_params`.
        KeyboardInterrupt
            If the solve is cancelled via SIGINT or
            ``cardal._core.request_cancel``.

        Returns
        -------
        Result
            An immutable :class:`cardal.Result` snapshot.
        """
        if self._problem is None:
            raise RuntimeError("no problem loaded; load or construct one first")
        core_result = _core.solve(self._problem, params)
        return _make_result(core_result)

    @classmethod
    def default_params(cls) -> dict:
        """Return the CARDAL default parameter values as a fresh dict.

        Keys correspond to the accepted keyword arguments of :meth:`solve`.
        Values are copied on each call; the returned dict is safe to mutate.
        """
        return _core.default_params()

    # ----- Problem metadata ------------------------------------------------
    #
    # Read-only pass-through properties. Each raises RuntimeError if
    # No problem has been loaded yet (loud misuse beats silent None).

    @property
    def num_cones(self) -> int:
        """Number of PSD cone blocks in the loaded problem."""
        return self._require_problem().num_cones

    @property
    def num_constraints(self) -> int:
        """Number of equality constraints ``m``."""
        return self._require_problem().num_constraints

    @property
    def num_variables(self) -> int:
        """Total active variable count across PSD, LP, and free blocks."""
        return self._require_problem().num_variables

    @property
    def lp_dim(self) -> int:
        """Dimension of the nonnegative LP block (``0`` for pure SDP)."""
        return self._require_problem().lp_dim

    @property
    def free_dim(self) -> int:
        """Dimension of the unrestricted real block."""
        return self._require_problem().free_dim

    @property
    def block_dims(self) -> List[int]:
        """Per-cone block dimensions ``[n_1, ..., n_p]``."""
        return list(self._require_problem().block_dims)

    def _require_problem(self) -> "_core.Problem":
        p = self._problem
        if p is None:
            raise RuntimeError("no problem loaded; load or construct one first")
        return p

    def __repr__(self) -> str:
        if self._problem is None:
            return "<cardal.Model (empty)>"
        return (
            f"<cardal.Model num_cones={self.num_cones} "
            f"num_constraints={self.num_constraints} "
            f"block_dims={self.block_dims}>"
        )
