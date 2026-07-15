# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

"""CARDAL Model — the user-facing entry point.

Typical usage::

    import cardal
    m = cardal.Model()
    m.read_file("problem.dat-s")
    result = m.solve(time_sec_limit=60.0, eps_optimal_relative=1e-4)
    print(result.summary())
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union
import os

import numpy as np

from . import _core
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


def _lp_matrix_to_coo(mat, num_constraints: int, lp_dim: int):
    if hasattr(mat, "tocoo"):
        coo = mat.tocoo()
        if coo.shape != (num_constraints, lp_dim):
            raise ValueError(
                f"A_lp shape {coo.shape} != ({num_constraints}, {lp_dim})"
            )
        return (
            np.asarray(coo.row, dtype=np.int32),
            np.asarray(coo.col, dtype=np.int32),
            np.asarray(coo.data, dtype=np.float64),
        )
    arr = np.ascontiguousarray(mat, dtype=np.float64)
    if arr.shape != (num_constraints, lp_dim):
        raise ValueError(
            f"A_lp shape {arr.shape} != ({num_constraints}, {lp_dim})"
        )
    rows, cols = np.nonzero(arr)
    return (
        rows.astype(np.int32),
        cols.astype(np.int32),
        arr[rows, cols].astype(np.float64),
    )


class Model:
    """A CARDAL SDP model.

    A ``Model`` is an empty shell after construction. Call :meth:`read_file`
    to load a problem, then :meth:`solve` to run the solver. The same
    ``Model`` can be reused across problems by calling :meth:`read_file`
    again; each call replaces the internal handle.

    Results are returned as immutable :class:`cardal.Result` objects and
    are NOT stored on the ``Model``.
    """

    __slots__ = ("_problem",)

    def __init__(self) -> None:
        self._problem: Optional["_core.Problem"] = None

    def read_file(self, path: Union[str, os.PathLike]) -> None:
        """Load an SDPA / MATLAB / PDSDP file.

        Format is auto-detected from the file header. Any previously loaded
        problem is discarded (the underlying C handle is dropped when the
        old reference goes out of scope).

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
        lp_dim: int = 0,
        lp_obj=None,
        A_lp: Optional[Tuple] = None,
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
        lp_dim : int
            Dimension of the optional nonnegative LP block. Default 0.
        lp_obj : array-like, length ``lp_dim``, or None
            LP-block objective. Required iff ``lp_dim > 0``.
        A_lp : tuple ``(constr_ind, col_ind, val)`` or None
            LP-block constraint triplets. Pass None for no LP constraints.

        Raises
        ------
        ValueError
            If any array length is inconsistent with the declared sizes.
        """
        block_dims_arr = np.ascontiguousarray(block_dims, dtype=np.int32)
        b_arr          = np.ascontiguousarray(b,          dtype=np.float64)
        num_cones = int(block_dims_arr.size)
        num_constraints = int(b_arr.size)

        c_cone, c_row, c_col, c_val = _unpack_coo(C, 4, "C")
        a_constr, a_cone, a_row, a_col, a_val = _unpack_coo(A, 5, "A")

        if lp_dim > 0:
            if lp_obj is None:
                raise ValueError("lp_obj required when lp_dim > 0")
            lp_obj_arr = np.ascontiguousarray(lp_obj, dtype=np.float64)
            if lp_obj_arr.size != lp_dim:
                raise ValueError(
                    f"lp_obj length {lp_obj_arr.size} != lp_dim {lp_dim}"
                )
        else:
            lp_obj_arr = np.empty(0, dtype=np.float64)

        if A_lp is not None:
            lp_constr, lp_col, lp_val = _unpack_coo(A_lp, 3, "A_lp")
        else:
            lp_constr = np.empty(0, dtype=np.int32)
            lp_col    = np.empty(0, dtype=np.int32)
            lp_val    = np.empty(0, dtype=np.float64)

        self._problem = _core.build_problem(
            num_constraints=num_constraints,
            num_cones=num_cones,
            lp_dim=int(lp_dim),
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
            lp_obj=lp_obj_arr,
            lp_constr_ind=lp_constr,
            lp_col_ind=lp_col,
            lp_val=lp_val,
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
            ``numpy.ndarray`` or any ``scipy.sparse`` matrix. Assumed
            symmetric; only the lower triangle (``row >= col``) is stored.
        A : sequence of ``m`` rows of ``p`` matrices, i.e. ``A[i][k]`` is
            the ``k``-th block of the ``i``-th constraint. Each entry may
            be a dense ndarray, a scipy.sparse matrix, or ``None`` (treated
            as zero). Assumed symmetric per block.
        lp_dim : int
            Dimension of the optional nonnegative LP block. Default 0.
        lp_obj : array-like, length ``lp_dim``, or None
            LP-block objective.
        A_lp : scipy.sparse matrix or ndarray of shape ``(m, lp_dim)`` or None
            LP-block constraint matrix. Rows index constraints, columns
            index LP variables.
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
            raise ValueError(
                f"len(A)={len(A)} does not match len(b)={num_constraints}"
            )

        # Build C-side COO by iterating over blocks.
        c_cone, c_row, c_col, c_val = [], [], [], []
        for k, Ck in enumerate(C):
            rows, cols, vals = _matrix_to_lower_coo(Ck, block_dims[k], f"C[{k}]")
            c_cone.append(np.full(rows.size, k, dtype=np.int32))
            c_row.append(rows); c_col.append(cols); c_val.append(vals)
        c_cone = _hstack_ints(c_cone)
        c_row  = _hstack_ints(c_row)
        c_col  = _hstack_ints(c_col)
        c_val  = _hstack_floats(c_val)

        # Build A-side COO by iterating over constraints and blocks.
        a_constr, a_cone, a_row, a_col, a_val = [], [], [], [], []
        for i, row in enumerate(A):
            if len(row) != num_cones:
                raise ValueError(
                    f"A[{i}] has length {len(row)}, expected {num_cones}"
                )
            for k, Aik in enumerate(row):
                if Aik is None:
                    continue
                rows, cols, vals = _matrix_to_lower_coo(
                    Aik, block_dims[k], f"A[{i}][{k}]"
                )
                if rows.size == 0:
                    continue
                a_constr.append(np.full(rows.size, i, dtype=np.int32))
                a_cone.append(np.full(rows.size, k, dtype=np.int32))
                a_row.append(rows); a_col.append(cols); a_val.append(vals)
        a_constr = _hstack_ints(a_constr)
        a_cone   = _hstack_ints(a_cone)
        a_row    = _hstack_ints(a_row)
        a_col    = _hstack_ints(a_col)
        a_val    = _hstack_floats(a_val)

        # LP.
        A_lp_coo = None
        if lp_dim > 0 and A_lp is not None:
            lp_constr, lp_col, lp_val = _lp_matrix_to_coo(A_lp, num_constraints, lp_dim)
            A_lp_coo = (lp_constr, lp_col, lp_val)

        self.set_problem_coo(
            block_dims=block_dims,
            b=b_arr,
            C=(c_cone, c_row, c_col, c_val),
            A=(a_constr, a_cone, a_row, a_col, a_val),
            lp_dim=lp_dim,
            lp_obj=lp_obj,
            A_lp=A_lp_coo,
        )

    def solve(self, **params: Any) -> Result:
        """Solve the loaded problem.

        Any keyword argument is passed straight through to the solver as a
        parameter override; unknown keys raise ``TypeError`` (contra
        Gurobi-style silent-accept). See :meth:`default_params` for the
        full list of recognized keys and their defaults.

        A running solve can be interrupted with Ctrl-C; the SIGINT handler
        installed by the C binding flips a cooperative cancel flag, the
        outer loop returns at its next boundary, and this method re-raises
        ``KeyboardInterrupt`` in Python.

        Raises
        ------
        RuntimeError
            If :meth:`read_file` has not been called.
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
            raise RuntimeError("no problem loaded; call read_file() first")
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
    # read_file() has not been called yet (loud misuse beats silent None).

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
        """Total variable count (sum of block-triangular sizes + LP block)."""
        return self._require_problem().num_variables

    @property
    def lp_dim(self) -> int:
        """Dimension of the nonnegative LP block (``0`` for pure SDP)."""
        return self._require_problem().lp_dim

    @property
    def block_dims(self) -> List[int]:
        """Per-cone block dimensions ``[n_1, ..., n_p]``."""
        return list(self._require_problem().block_dims)

    def _require_problem(self) -> "_core.Problem":
        p = self._problem
        if p is None:
            raise RuntimeError("no problem loaded; call read_file() first")
        return p

    def __repr__(self) -> str:
        if self._problem is None:
            return "<cardal.Model (empty)>"
        return (
            f"<cardal.Model num_cones={self.num_cones} "
            f"num_constraints={self.num_constraints} "
            f"block_dims={self.block_dims}>"
        )
