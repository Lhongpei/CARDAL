# Results and Outputs

CARDAL reports objectives, residuals, iteration counts, and the final low-rank
primal factors.

## Termination status

| Status | Meaning |
|:--|:--|
| `OPTIMAL` | Requested primal, dual, and objective-gap tolerances were met |
| `TIME_LIMIT` | The wall-clock budget was reached |
| `ITERATION_LIMIT` | The outer iteration limit was reached |
| `USER_INTERRUPT` | Cooperative cancellation was requested |
| `UNSPECIFIED` | No status was assigned; not expected for a normal return |

Python raises `KeyboardInterrupt` for Ctrl-C cancellation instead of returning
a `Result` with `USER_INTERRUPT`.

## Python Result

`Model.solve` returns an immutable `cardal.Result`:

| Field | Type | Description |
|:--|:--|:--|
| `status` | `TerminationReason` | Final solver status |
| `primal_objective` | `float` | Final \(\langle C,X\rangle\) |
| `dual_objective` | `float` | Final dual objective |
| `objective_gap` | `float` | Absolute primal-dual gap |
| `rel_primal_residual` | `float` | Relative primal feasibility residual |
| `rel_dual_residual` | `float` | Relative dual feasibility residual |
| `rel_objective_gap` | `float` | Relative primal-dual gap |
| `runtime_sec` | `float` | Solver runtime excluding parse and post-solve unscaling |
| `outer_iters` | `int` | Outer ALM iterations |
| `inner_iters` | `int` | Total inner L-BFGS iterations |
| `total_rank` | `int` | Sum of final PSD block ranks |
| `rank_list` | `numpy.ndarray` | Final rank of each PSD block |
| `primal_factor` | `numpy.ndarray` | Flattened PSD factors |
| `dual` | `numpy.ndarray` | Dual multipliers |

Use `result.summary()` for a formatted report.

## Reconstruct PSD blocks

Factors are concatenated by cone and stored column-major within each cone:

```python
offset = 0
factors = []

for dim, rank in zip(model.block_dims, result.rank_list):
    size = dim * int(rank)
    factor = result.primal_factor[offset : offset + size].reshape(
        (dim, int(rank)),
        order="F",
    )
    factors.append(factor)
    offset += size

X_blocks = [V @ V.T for V in factors]
```

`primal_factor`, `dual`, and `rank_list` are read-only zero-copy NumPy views.
Their storage remains alive with the `Result`.

## C result buffers

C result accessors return borrowed pointers:

```c
int factor_length = 0;
const double *factor =
    cardal_result_primal_factor(result, &factor_length);

int dual_length = 0;
const double *dual =
    cardal_result_dual(result, &dual_length);
```

Do not free these arrays. They become invalid when
`cardal_result_free(result)` is called.

## CLI output files

The CLI always prints a final summary. Passing `--output-dir` additionally
writes:

```text
<output-dir>/<instance>_summary.txt
<output-dir>/<instance>_primal_factor.txt
<output-dir>/<instance>_rank_list.txt
<output-dir>/<instance>_dual_solution.txt
```

The summary includes the status, objectives, relative residuals, gap, rank,
iteration counts, runtime, and rescaling time when applicable.

Each solution file contains one value per line. The primal factor uses the same
layout as the Python and C result buffers: PSD cones are stored consecutively,
with each \(n_c \times r_c\) factor in column-major order. The corresponding
entry in `rank_list` gives \(r_c\). If an LP tail is present, it follows the PSD
factor data. The dual file contains one multiplier per constraint.

For example:

```python
import numpy as np

factor = np.loadtxt("results/problem_primal_factor.txt")
ranks = np.loadtxt("results/problem_rank_list.txt", dtype=np.int32)
dual = np.loadtxt("results/problem_dual_solution.txt")
```
