# Python Interface

The Python package is a single-GPU interface built around `Model`, `Result`,
and `TerminationReason`.

## Install

```bash
python -m pip install cardal
```

For SciPy sparse matrices:

```bash
python -m pip install "cardal[scipy]"
```

## Read a problem file

```python
import cardal

model = cardal.Model()
model.read_file("problem.dat-s")
result = model.solve(verbose=1)

print(result.summary())
```

`read_file` replaces any problem previously held by the model.

## Construct from block matrices

`set_problem` accepts dense NumPy arrays or SciPy sparse matrices:

```python
import numpy as np
import cardal

H = np.diag([3.0, 1.0, 2.0])

model = cardal.Model()
model.set_problem(
    block_dims=[3],
    C=[H],
    A=[[np.eye(3)]],
    b=[1.0],
)

result = model.solve(eps_optimal_relative=1e-6, verbose=0)
```

The example solves
\(\min\langle H,X\rangle\) subject to
\(\mathrm{tr}(X)=1\) and \(X\succeq0\).

`A[i][c]` is the block for constraint \(i\), cone \(c\). It may be `None` to
represent a zero block. Only the lower triangle of each matrix is read.

## Construct from COO arrays

For large sparse inputs, `set_problem_coo` avoids creating a matrix object for
every constraint block:

```python
import numpy as np
import cardal

model = cardal.Model()
model.set_problem_coo(
    block_dims=[3],
    b=np.array([1.0]),
    C=(
        np.array([0, 0, 0], dtype=np.int32),     # cone
        np.array([0, 1, 2], dtype=np.int32),     # row
        np.array([0, 1, 2], dtype=np.int32),     # col
        np.array([3.0, 1.0, 2.0]),               # value
    ),
    A=(
        np.array([0, 0, 0], dtype=np.int32),     # constraint
        np.array([0, 0, 0], dtype=np.int32),     # cone
        np.array([0, 1, 2], dtype=np.int32),     # row
        np.array([0, 1, 2], dtype=np.int32),     # col
        np.ones(3),                              # value
    ),
)
```

All coordinate arrays are zero-indexed. CARDAL converts index arrays to
contiguous `int32` storage and values to contiguous `float64` storage.

## Add a nonnegative LP tail

```python
model.set_problem(
    block_dims=[3],
    C=[np.diag([3.0, 1.0, 2.0])],
    A=[[np.eye(3)]],
    b=[1.0],
    lp_dim=2,
    lp_obj=np.array([0.5, -0.5]),
    A_lp=np.array([[1.0, 1.0]]),
)
```

`A_lp` has shape `(num_constraints, lp_dim)`. It may be a NumPy array or a
SciPy sparse matrix.

## Solver parameters

Pass parameter overrides as keyword arguments:

```python
result = model.solve(
    time_sec_limit=300.0,
    initial_rank=16,
    augmentation_mode="qp",
    l_inf_ruiz_iterations=10,
    pock_chambolle_rescaling=True,
    psd_scale_mode="per-element",
)
```

Unknown names raise `TypeError`; invalid enumerated values raise `ValueError`.
The complete cross-interface table is on the [Parameters](../parameters.md)
page. The live defaults are available programmatically:

```python
defaults = cardal.Model.default_params()
print(defaults)
```

## Model reference

| Member | Description |
|:--|:--|
| `Model()` | Create an empty reusable model |
| `read_file(path)` | Load a supported problem file |
| `set_problem(...)` | Build from dense or sparse block matrices |
| `set_problem_coo(...)` | Build from raw COO arrays |
| `solve(**params)` | Solve and return an immutable `Result` |
| `Model.default_params()` | Return all recognized parameters and defaults |
| `num_cones` | Number of PSD blocks |
| `num_constraints` | Number of equality constraints |
| `num_variables` | Total triangular PSD plus LP variable count |
| `block_dims` | PSD block dimensions |
| `lp_dim` | Size of the nonnegative LP tail |

## Cancellation

Pressing Ctrl-C during `solve` requests cooperative cancellation and raises
`KeyboardInterrupt`. No `Result` is returned for that call. The cancellation
flag is cleared before the next solve.

See [Results and Outputs](results.md) for the `Result` fields and status codes.

