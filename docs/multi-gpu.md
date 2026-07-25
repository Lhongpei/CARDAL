# Multi-GPU

The native CARDAL executable supports distributed execution with MPI for
process control and NCCL for GPU collectives. The PyPI Python interface is
single-GPU.

## Build

MPI support is enabled by default in the source build:

```bash
cmake -S . -B build
cmake --build build -j
```

The build requires an MPI 3.1 implementation and NCCL 2.18 or newer. To select
a CUDA toolkit explicitly:

```bash
CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
  cmake -S . -B build
cmake --build build -j
```

## Launch

Use one MPI process per GPU:

```bash
mpirun -n 4 ./build/cardal -f problem.dat-s
```

Restrict the visible devices when the node has additional GPUs:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
  mpirun -n 4 ./build/cardal -f problem.dat-s
```

CARDAL detects the MPI environment and selects the distributed solver
automatically.

## Process grid

The optional grid is written as:

```text
row,rank,cone
```

Its dimensions must multiply to the MPI process count.

| Axis | Distributed quantity |
|:--|:--|
| `row` | Equality constraints |
| `rank` | Columns of the Burer-Monteiro factors |
| `cone` | PSD cone blocks |

Examples:

```bash
# Four processes split across row and rank axes
mpirun -n 4 ./build/cardal \
  -f problem.dat-s \
  --grid-size 2,2,1

# Eight processes split across cone ownership
mpirun -n 8 ./build/cardal \
  -f problem.dat-s \
  --grid-size 1,1,8
```

When `--grid-size` is omitted, CARDAL selects a valid topology from the problem
structure and process count.

## Constraint ordering

Distributed builds can reorder constraints before partitioning:

```bash
mpirun -n 4 ./build/cardal \
  -f problem.dat-s \
  --shuffle col
```

Available modes are:

| Mode | Meaning |
|:--|:--|
| `none` | Preserve input order |
| `uniform` | Uniform redistribution |
| `block` | Block-oriented ordering |
| `col` | Column-locality ordering |

The default is `col`.

## Record results

`--output-dir` works identically in distributed runs:

```bash
mpirun -n 8 ./build/cardal \
  -f problem.dat-s \
  --grid-size 1,2,4 \
  --output-dir results
```

Only the root process writes the consolidated summary, PSD factor, optional
LP/free primal vectors, rank list, and dual solution.
