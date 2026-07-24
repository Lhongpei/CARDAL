# Frequently Asked Questions

## Which interface should I use?

Use Python for NumPy/SciPy model construction and single-GPU workflows. Use the
C API to embed CARDAL in a native application. Use the CLI for files,
generators, reproducible shell scripts, and distributed runs.

## Why does `pip install cardal` invoke a compiler?

The package contains a CUDA extension and is built against the CUDA toolkit on
the installation machine. A working `nvcc` and compatible host compiler are
required.

## Is SciPy required?

No. NumPy and file input work without SciPy. Install `cardal[scipy]` only when
passing `scipy.sparse` matrices to `Model.set_problem`.

## Can a `Model` be reused?

Yes. Calling `read_file`, `set_problem`, or `set_problem_coo` replaces the
current problem. Previously returned `Result` objects remain valid and
independent.

## Are Python result arrays copied?

No. `primal_factor`, `dual`, and `rank_list` are read-only zero-copy views over
the C result buffers. The `Result` object keeps those buffers alive.

## Why does CARDAL use one matrix triangle?

PSD objective and constraint blocks are symmetric. Storing one triangle avoids
duplicate coordinates and unnecessary memory traffic. Python and C in-memory
input use lower-triangular coordinates with `row >= col`.

## How do I solve a QUBO?

Use the dedicated `cardal_qubo` executable. It builds an SDP relaxation and
performs GPU random-hyperplane rounding:

```bash
./build/cardal_qubo -f model.qubo
```

The main `cardal` executable intentionally handles SDP input only.

## What happens when I press Ctrl-C in Python?

CARDAL requests cooperative cancellation and Python raises
`KeyboardInterrupt`. The cancellation state is cleared before the next solve.

## Can I contribute?

Yes. Pull requests are welcome. Keep changes focused, include a test when
behavior changes, and describe the CUDA and MPI configurations used for
validation.

