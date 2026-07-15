# CARDAL &mdash; Python Interface

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](../pyproject.toml)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

Python bindings for **CARDAL**, a GPU-accelerated Augmented Lagrangian solver for large-scale semidefinite programs on a Burer-Monteiro low-rank factorization. For the CLI, distributed builds, citation, and the top-level project overview see the [main README](../README.md).

The interface is intentionally small: one class (`Model`), one frozen dataclass (`Result`), one `IntEnum` (`TerminationReason`), and a solver-knob dictionary discovered via `Model.default_params()`. The Python build is **single-GPU only** &mdash; for multi-GPU runs use the C CLI under `mpirun`.

## Installation

### Requirements

- Python 3.9+
- `numpy >= 1.21`
- Optional: `scipy >= 1.8` (sparse-matrix input), `cvxpy >= 1.4` (cross-checking small instances)
- A working CUDA 12.x toolchain (`nvcc`) matched to your host compiler; see the [main README](../README.md) for the full hardware/toolchain matrix
- An NVIDIA GPU with compute capability 7.0-9.0 (Volta through Hopper)

### Install from source

CARDAL is not yet on PyPI; install from a checkout:

```bash
git clone https://github.com/Lhongpei/CARDAL.git
cd CARDAL
pip install .
```

Optional extras and a developer install:

```bash
pip install ".[scipy]"      # scipy >= 1.8, for scipy.sparse inputs
pip install ".[cvxpy]"      # cvxpy >= 1.4
pip install -e ".[dev]"     # pytest, editable install
pytest python/tests
```

The wheel is built with `scikit-build-core` and compiles the CUDA extension in place. If the system `nvcc` is too old, point scikit-build at a newer toolchain via `CUDACXX` before invoking `pip install`:

```bash
CUDACXX=/usr/local/cuda-12.6/bin/nvcc pip install .
```

The CMake project also honours `SKBUILD_CMAKE_ARGS` for pass-through configuration (e.g. `SKBUILD_CMAKE_ARGS="-DENABLE_MATIO=OFF"`).

## Quick Start

### From a file

CARDAL reads SDPA (`.dat-s`, `.dat-s.gz`), MATLAB (`.mat`), and PDSDP (`.npz`) files; the format is auto-detected from the header.

```python
import cardal

m = cardal.Model()
m.read_file("problem.dat-s")                       # or .dat-s.gz / .mat / .npz

result = m.solve(
    time_sec_limit=60.0,
    eps_optimal_relative=1e-4,
    eps_feasible_relative=1e-4,
    verbose=1,
)

print(f"Status:              {result.status}")
print(f"Primal objective:    {result.primal_objective:.6e}")
print(f"Rel. objective gap:  {result.rel_objective_gap:.2e}")
print(f"Outer / inner iters: {result.outer_iters} / {result.inner_iters}")

if result.status is cardal.OPTIMAL:
    print("Converged to the requested tolerances.")

print(result.summary())                            # multi-line human-readable report
```

The same `Model` is reusable: calling `m.read_file(...)` again swaps the underlying problem, and every `m.solve(...)` returns an independent immutable `Result`. SDPLIB instances such as `fe4s4_sos.dat-s` are available at [github.com/vsdp/SDPLIB](https://github.com/vsdp/SDPLIB). The random-problem generators (`maxcut`, `snl`, `quantum_order`) are exposed only through the C CLI's `-g` flag &mdash; there is no `load_generator` in Python; produce a `.dat-s` file with the CLI first and read it back.

### From NumPy arrays

`Model.set_problem()` accepts a list of dense `numpy` or `scipy.sparse` matrices per PSD block. As a worked example, the smallest-eigenvalue SDP for a symmetric matrix $H$ &mdash; $\min\langle H, X\rangle$ subject to $\operatorname{tr}(X)=1$, $X\succeq 0$ &mdash; whose optimum equals $\lambda_{\min}(H)$:

```python
import numpy as np
import cardal

H = np.diag([3.0, 1.0, 2.0])                        # lambda_min(H) = 1.0

m = cardal.Model()
m.set_problem(
    block_dims=[3],                                 # one 3x3 PSD block
    C=[H],                                          # objective, one matrix per block
    A=[[np.eye(3)]],                                # 1 constraint x 1 block: trace(X) = 1
    b=[1.0],
)
result = m.solve(eps_optimal_relative=1e-6, verbose=0)
assert abs(result.primal_objective - 1.0) < 1e-3

# Post-solve metadata
print(m.num_cones, m.num_constraints, m.num_variables, m.block_dims, m.lp_dim)
```

Matrices may be dense (`numpy.ndarray`) or any `scipy.sparse` matrix (CSR, CSC, and COO are all accepted). Each is assumed symmetric, and **only the lower triangle is read**: entries with `row < col` are ignored, and off-diagonals are **not** implicitly doubled. `A[i][k]` may be `None` to mean "constraint *i* touches nothing in cone *k*".

### From `scipy.sparse`

The lower-triangle convention applies verbatim to sparse inputs. Pre-mask with `scipy.sparse.tril` if your generator produces both triangles:

```python
import numpy as np, scipy.sparse as sp, cardal

n = 500
# Random symmetric objective, lower triangle only (nnz counted once):
C = sp.tril((sp.random(n, n, density=1e-3, format="csr", random_state=0)
             + sp.random(n, n, density=1e-3, format="csr", random_state=1).T))
I_diag = sp.eye(n, format="csr")                    # diagonal identity is already lower-triangular

m = cardal.Model()
m.set_problem(
    block_dims=[n],
    C=[C],                                          # sparse objective
    A=[[I_diag]],                                   # one constraint: trace(X) = 1
    b=[1.0],
)
result = m.solve(time_sec_limit=30.0, verbose=1)
```

### Mixed SDP + LP

An optional nonnegative LP tail $x_{\text{LP}}\ge 0$ is appended after the PSD blocks. Pass `lp_dim`, the LP objective `lp_obj`, and the LP-block constraint columns `A_lp` (dense, `scipy.sparse`, or COO 3-tuple `(constr_ind, col_ind, val)` for `set_problem_coo`). `lp_obj` is required whenever `lp_dim > 0`:

```python
m.set_problem(
    block_dims=[3],
    C=[np.diag([3.0, 1.0, 2.0])],
    A=[[np.eye(3)]],
    b=[1.0],
    lp_dim=2,
    lp_obj=np.array([0.5, -0.5]),                   # linear cost on x_LP
    A_lp=np.array([[1.0, 1.0]]),                    # 1 constraint x 2 LP vars
)
```

### From raw COO triplets

For very large problems it is often faster to skip the matrix layer entirely. `set_problem_coo` is a 1:1 mirror of the C ABI:

```python
m.set_problem_coo(
    block_dims=[3],
    b=[1.0],
    # C: (cone_ind, row_ind, col_ind, val) --- diag(3, 1, 2)
    C=(np.zeros(3, np.int32),
       np.arange(3, dtype=np.int32),
       np.arange(3, dtype=np.int32),
       np.array([3.0, 1.0, 2.0])),
    # A: (constr_ind, cone_ind, row_ind, col_ind, val) --- I as one constraint
    A=(np.zeros(3, np.int32),
       np.zeros(3, np.int32),
       np.arange(3, dtype=np.int32),
       np.arange(3, dtype=np.int32),
       np.ones(3)),
)
```

The COO path uses the same lower-triangle-only convention; LP-block triplets go through `A_lp=(constr_ind, col_ind, val)` alongside `lp_dim` and `lp_obj`.

## Cone Constraints

CARDAL supports two cone kinds, both provided implicitly through `block_dims` and `lp_dim` rather than a separate cone-descriptor argument:

- **PSD cones.** Each entry of `block_dims` is the side length $n_c$ of a symmetric block $X_c \succeq 0$. Stored as Burer-Monteiro factors $V_c \in \mathbb{R}^{n_c \times r_c}$.
- **Nonnegative LP tail.** A single nonnegative block $x_{\text{LP}} \in \mathbb{R}^{\text{lp\_dim}}$ appended after the PSD blocks. Set `lp_dim=0` (default) for pure SDPs.

Second-order cones, exponential cones, and generic bounded cones are **not** supported. Nonnegative LP variables are the only non-PSD cone.

## `cardal.Model`

| Symbol                            | Purpose                                                                                                    |
|:----------------------------------|:-----------------------------------------------------------------------------------------------------------|
| `cardal.Model()`                  | Construct an empty model. Reusable across problems.                                                        |
| `m.read_file(path)`               | Parse an SDPA/MATLAB/PDSDP file (auto-detected) and replace the internal handle. Raises `FileNotFoundError` if the path is missing and `ValueError` on parse failure. |
| `m.set_problem(**kwargs)`         | Build from lists of dense `numpy` or `scipy.sparse` matrices per block.                                    |
| `m.set_problem_coo(**kwargs)`     | Low-level entry: build from raw COO triplet arrays (5 arrays for the constraints, 4 for the objective, plus `b`). |
| `m.solve(**params)`               | Solve the loaded problem. Returns a frozen `cardal.Result`. Raises `RuntimeError` if no problem was loaded and `TypeError` on unknown kwargs. |
| `Model.default_params()`          | `@classmethod` returning a fresh, mutable dict of every recognized parameter and its default.              |

Read-only properties (each raises `RuntimeError` before a problem is loaded):

| Property             | Description                                                        |
|:---------------------|:-------------------------------------------------------------------|
| `m.num_cones`        | Number of PSD blocks $p$.                                          |
| `m.num_constraints`  | Number of equality constraints $m$.                                |
| `m.num_variables`    | Total variable count (sum of triangular block sizes + LP block).   |
| `m.lp_dim`           | Size of the nonnegative LP tail (`0` for pure SDP).                |
| `m.block_dims`       | Per-cone side lengths $[n_1,\dots,n_p]$, as a fresh list.          |

`repr(m)` yields `"<cardal.Model (empty)>"` before load and `"<cardal.Model num_cones=... num_constraints=... block_dims=...>"` afterwards. **No silent typos.** Unlike Gurobi, `solve()` does not accept unknown keyword arguments; any typo raises `TypeError("unknown parameter '<name>'")`. Cross-check against `cardal.Model.default_params().keys()`.

## Parameters

Every parameter is passed as a keyword argument to `Model.solve`. `Model.default_params()` is the **canonical source** of every recognized key and its default value &mdash; the table below documents the commonly tuned subset. These defaults come from `set_default_parameters()` in `src/utils.cu` and are shared between the CLI and the Python API.

| Parameter                    | Type   | Default        | Description                                                                     |
|:-----------------------------|:-------|:---------------|:--------------------------------------------------------------------------------|
| `eps_optimal_relative`       | float  | `1e-4`         | Relative objective-gap tolerance.                                               |
| `eps_feasible_relative`      | float  | `1e-4`         | Relative primal/dual residual tolerance.                                        |
| `time_sec_limit`             | float  | `3600.0`       | Wall-clock budget in seconds (`0.0` disables).                                  |
| `iteration_limit`            | int    | `20000000`     | Total outer ALM iteration cap.                                                  |
| `initial_rank`               | int    | `-1`           | Starting per-cone BM rank; `-1` uses $\lceil 2 \log m \rceil$.                  |
| `max_rank`                   | int    | `-1`           | Rank cap per cone; `-1` uses the Pataki bound $\lceil(\sqrt{8m+1}-1)/2\rceil$.  |
| `lbfgs_history_size`         | int    | `5`            | L-BFGS memory length used by the inner solver.                                  |
| `penalty_factor`             | float  | `1.2`          | Multiplicative ALM penalty update per outer step.                               |
| `initial_penalty_coef`       | float  | `-1.0`         | Starting ALM penalty $\beta$; `-1.0` uses $2/\sqrt{N}$.                         |
| `max_penalty_coef`           | float  | `5e5`          | Upper cap on $\beta$.                                                           |
| `inner_iterations_limit`     | int    | `30000`        | Inner L-BFGS iteration cap per outer step.                                      |
| `verbose`                    | int    | `2`            | `0` silent; `1` banner + summary; `2` + iter table; `3` debug.                  |

Additional preconditioner/scaling switches are exposed only through the C CLI (`--l_inf_ruiz_iter`, `--pock_chambolle_alpha`, `--psd_scale_mode`, `--no_scaling`, `--no_pock_chambolle`, `--no_bound_obj_rescaling`) &mdash; leave them at their defaults unless retuning. Discover every accepted Python kwarg programmatically:

```python
defaults = cardal.Model.default_params()
print(sorted(defaults))                             # every accepted parameter name
print(defaults["eps_optimal_relative"])             # its default value
```

## Solution Attributes

`Model.solve(...)` returns a frozen `cardal.Result` dataclass. All numeric fields are populated on every return.

### Attribute Reference

| Field                  | Type                | Description                                                              |
|:-----------------------|:--------------------|:-------------------------------------------------------------------------|
| `status`               | `TerminationReason` | Termination code (see below).                                            |
| `primal_objective`     | `float`             | Final primal objective $\langle C, X\rangle$.                            |
| `dual_objective`       | `float`             | Final dual objective.                                                    |
| `objective_gap`        | `float`             | Absolute primal-dual gap.                                                |
| `rel_primal_residual`  | `float`             | Relative primal feasibility residual.                                    |
| `rel_dual_residual`    | `float`             | Relative dual feasibility residual.                                      |
| `rel_objective_gap`    | `float`             | Relative primal-dual gap (checked against `eps_optimal_relative`).       |
| `runtime_sec`          | `float`             | Solver wall-clock (excludes parse and post-solve unscaling).             |
| `outer_iters`          | `int`               | Outer ALM iteration count.                                               |
| `inner_iters`          | `int`               | Total L-BFGS inner iteration count.                                      |
| `num_cones`            | `int`               | Number of cone blocks.                                                   |
| `num_variables`        | `int`               | Total variable count.                                                    |
| `num_constraints`      | `int`               | Number of equality constraints.                                          |
| `total_rank`           | `int`               | Sum of per-cone BM ranks at termination.                                 |
| `primal_factor`        | `numpy.ndarray`     | Flattened Burer-Monteiro factor $V$, length $\sum_c d_c r_c$.            |
| `dual`                 | `numpy.ndarray`     | Dual multipliers $y$, length `num_constraints`.                          |
| `rank_list`            | `numpy.ndarray`     | Per-cone final BM rank, length `num_cones`.                              |

The three NumPy arrays are **zero-copy read-only views** over C-side buffers; the underlying storage lives as long as the `Result` object (a private `_handle` field pins it). `repr(result)` gives a one-line status; `result.summary()` returns a formatted multi-line report &mdash; the same shape the CLI writes into `<instance>_summary.txt` under `-O`.

`TerminationReason` is an `IntEnum` whose members are also re-exported at module scope:

| Constant                  | Meaning                                                                                                    |
|:--------------------------|:-----------------------------------------------------------------------------------------------------------|
| `cardal.UNSPECIFIED`      | Solver did not set a status (should not occur on normal returns).                                          |
| `cardal.OPTIMAL`          | All three relative tolerances satisfied.                                                                   |
| `cardal.TIME_LIMIT`       | Exceeded `time_sec_limit`.                                                                                 |
| `cardal.ITERATION_LIMIT`  | Exceeded `iteration_limit`.                                                                                |
| `cardal.USER_INTERRUPT`   | Cooperative cancel flag was set. From Python, Ctrl-C raises `KeyboardInterrupt` instead of returning this. |

Use identity comparison (`result.status is cardal.OPTIMAL`) or the enum directly.

## Cooperative Cancellation

Pressing **Ctrl-C** during a running `m.solve(...)` cooperatively cancels the solver and re-raises `KeyboardInterrupt` in Python. Mechanically:

1. The pybind11 binding installs a `SIGINT` handler that **flips a cooperative cancel flag** (no `longjmp`, no `abort`).
2. The outer ALM loop polls this flag at each iteration boundary. When set, the loop returns the partial state (C-side status would be `USER_INTERRUPT`).
3. The Python wrapper detects the cancel and re-raises `KeyboardInterrupt` on the calling thread, so idiomatic `try/except KeyboardInterrupt` works. No `Result` is returned in that case.
4. The flag is auto-cleared before the next `solve()`, so successive calls don't inherit stale cancels.

Because the wrapper re-raises `KeyboardInterrupt`, users almost never observe `result.status == cardal.USER_INTERRUPT` from Python. That code exists for callers who trigger the cancel programmatically:

```python
import threading, time
import cardal
from cardal import _core                             # private, but stable

def _cancel_after(delay):
    time.sleep(delay)
    _core.request_cancel()

m = cardal.Model()
m.read_file("big.dat-s")
threading.Thread(target=_cancel_after, args=(5.0,), daemon=True).start()

try:
    result = m.solve(verbose=1)
except KeyboardInterrupt:
    print("cancelled from another thread")
```

> The cancel flag is process-wide. Two `solve()` calls running concurrently in the same process both terminate on a single cancel; callers needing per-solve cancellation must serialize.

## Troubleshooting

- **`TypeError: unknown parameter '<name>'` from `m.solve()`.** CARDAL does not silently accept unknown kwargs. Cross-check against `cardal.Model.default_params().keys()`.
- **`RuntimeError: no problem loaded` from `m.solve()` or a property.** Call `m.read_file(...)`, `m.set_problem(...)`, or `m.set_problem_coo(...)` first.
- **`FileNotFoundError` from `m.read_file(...)`.** Verify absolute vs relative path, and that `.dat-s.gz` files are actually gzip-compressed.
- **`ValueError` from `m.read_file(...)`.** The parser could not identify the header; check the file is SDPA sparse `.dat-s` / `.dat-s.gz`, MATLAB `.mat`, or PDSDP `.npz`.
- **Off-diagonals look under-weighted.** CARDAL reads only the lower triangle (`row >= col`) and does *not* implicitly double off-diagonals. Provide exactly one triangle; do not pre-symmetrize or pre-scale.
- **Ctrl-C during `solve()` returns no result.** By design: `SIGINT` flips the cancel flag, the solver breaks at the next outer boundary, and the binding re-raises `KeyboardInterrupt`.
- **`nvcc: command not found` or CUDA too old during `pip install`.** Export `CUDACXX=/usr/local/cuda-12.6/bin/nvcc` (or your actual path) before invoking `pip install`.
- **Sparse matrix rejected.** Install the `scipy` extra: `pip install ".[scipy]"`.
- **Wanted multi-GPU from Python.** The Python front-end is single-GPU only. Use the C CLI under `mpirun`; see the [main README](../README.md).
