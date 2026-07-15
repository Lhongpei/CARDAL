# CARDAL: A Curvature-Aware Rank-Adaptive Distributed Augmented-Lagrangian Solver for Large-Scale SDPs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Distributed](https://img.shields.io/badge/Multiple_GPUs-optional-brightgreen.svg)](https://www.open-mpi.org/) [![SDPLIB](https://img.shields.io/badge/Input-SDPA%20%7C%20SeDuMi%20%7C%20SDPT3-lightgrey.svg)](https://github.com/vsdp/SDPLIB) [![Interface](https://img.shields.io/badge/Interface-Python%20%7C%20C-lightyellow.svg)](https://github.com/vsdp/SDPLIB)

**CARDAL** is a curvature-aware, rank-adaptive, distributed augmented-Lagrangian solver for large-scale semidefinite programs. It operates on a Burer-Monteiro low-rank factorization and supports both single- and multi-GPU systems. CARDAL targets semidefinite programs whose optimal solutions are (or are expected to be) low-rank.

A companion manuscript of CARDAL is in preparation.



### Problem Formulation

CARDAL solves standard-form semidefinite programs with block-diagonal PSD variables and an optional nonnegative LP block:

```math
\begin{aligned}
\min_{X} \quad & \langle C, X \rangle \\
\text{s.t.} \quad & \langle A_i, X \rangle = b_i, \quad i = 1,\dots,m, \\
                  & X = \operatorname{blkdiag}(X_1, \dots, X_p, x_{\text{LP}}), \\
                  & X_c \succeq 0 \text{ for } c=1,\dots,p, \qquad x_{\text{LP}} \ge 0.
\end{aligned}
```

$C$ and each $A_i$ are symmetric block-diagonal matrices. Every matrix is stored by its **lower triangle only**, and off-diagonal entries are **not** implicitly doubled &mdash; the same convention applies to the Python API. Each PSD block $X_c \in \mathbb{R}^{n_c \times n_c}$ is stored as a factor $V_c \in \mathbb{R}^{n_c \times r_c}$ with $X_c = V_c V_c^\top$.

## Features

- **GPU-native.** Most operations are implemented natively on GPUs.
- **Multi-GPU.** MPI + NCCL parallelization across constraint, rank, and cone axes, enabled by default (`-DENABLE_MPI=OFF` to opt out).
- **Algorithm.** Adaptive rank augmentation, negative curvature escape.

## Requirements

| Component   | Minimum / Recommended                                                            |
|:------------|:---------------------------------------------------------------------------------|
| OS          | Linux x86_64                                                                     |
| CUDA        | CUDA 12.x; CUDA 12.6 or newer recommended                                        |
| Compiler    | C++17-capable host compiler compatible with the selected CUDA toolkit            |
| Build tools | CMake 3.20 or newer, plus `make` or `ninja`                                      |
| zlib        | Required for gzipped file input                                                  |
| matio       | Optional; auto-fetched (`tbeu/matio` v1.5.23). Disable with `-DENABLE_MATIO=OFF` |
| MPI         | Optional; OpenMPI 4.1+ or another MPI-3.1 implementation for distributed builds  |
| NCCL        | Optional; NCCL 2.18+ for on-device collectives                                   |

On-device collectives run through NCCL, so a CUDA-aware MPI is **not** required. If your system has multiple CUDA versions or the default nvcc is outdated (e.g., in `/usr/bin/nvcc`), you should explicitly specify the path to your modern CUDA compiler using the `CUDACXX` environment variable.

## Installation

### Build from source (C++ CLI)

Multi-GPU support (MPI + NCCL) is **enabled by default**; add `-DENABLE_MPI=OFF` for a single-GPU-only binary. If the default `nvcc` is outdated or missing, prefix the first `cmake` invocation with `CUDACXX=/path/to/nvcc`.

```bash
git clone https://github.com/Lhongpei/CARDAL.git
cd CARDAL
cmake -S . -B build
cmake --build build --clean-first
```

The main binary lands at `./build/cardal`. A sibling binary `./build/cardal_qubo` (built when `CARDAL_BUILD_QUBO=ON`, default) specializes in QUBO-lifted SDPs, takes the same core CLI flags with a QUBO-specific input parser (`./build/cardal_qubo -f model.qubo` or `-g qubo,<n>,<density>`), and the main `cardal` binary rejects QUBO inputs. Run `./build/cardal_qubo -h` for its full syntax.

### Python package

CARDAL also ships a NumPy-friendly Python front-end (single-GPU only), see [python/README.md](./python/README.md).

## Quickstart

### First run (no input file required)

Verify the CLI install end-to-end without any external data:

```bash
./build/cardal -g maxcut,1000,1e-2 -v 1
```

The primal objective is a large negative number by the MaxCut sign convention above. What matters is that `Status: OPTIMAL` appears and the three relative residuals are all below `1e-4`. Passing `-O ./output` in addition writes the same summary to `./output/<instance>_summary.txt` for post-processing.

### CLI

```bash
# From a file (SDPA, MATLAB, or PDSDP; format auto-detected)
./build/cardal -f problem.dat-s -O ./output

# From a built-in generator (no external file needed)
./build/cardal -g maxcut,30000,1e-3
./build/cardal -g snl,10000,4,2,0.05,0.0
./build/cardal -g quantum_order,64,3
```

### Python Interface

```python
import cardal

m = cardal.Model()
m.read_file("problem.dat-s")                       # or .dat-s.gz / .mat / .npz
result = m.solve(time_sec_limit=60.0, eps_optimal_relative=1e-4)
print(result.status, result.primal_objective, result.rel_objective_gap)
```

## Advanced Usage

### CLI reference

```
./build/cardal (-f <PATH> | -g <SPEC>) [OPTIONS]
```

Exactly one of `--file` or `--gen` must be provided.

- `-f, --file <path>` reads a problem file (SDPA `.dat-s` / `.dat-s.gz`, MATLAB `.mat`, PDSDP `.npz`; auto-detected).
- `-g, --gen <spec>` synthesizes a random SDP on the fly. Comma-separated spec:

| Generator          | Syntax                                    | Example                       |
|:-------------------|:------------------------------------------|:------------------------------|
| MaxCut             | `maxcut,<n>,<density>`                    | `maxcut,30000,1e-3`           |
| Sensor Net. Loc.   | `snl,<n>,<anchors>,<dim>,<R>,<noise>`     | `snl,10000,4,2,0.05,0.0`      |
| Quantum ordering   | `quantum_order,<N>,<k>` (alias `qorder`)  | `quantum_order,64,3`          |

Run `./build/cardal -h` for the full option list (tolerances, ranks, penalty schedule, preconditioner switches, output directory). Solver tuning defaults come from `set_default_parameters()` in `src/utils.cu` and are shared between the CLI and the Python API; the Python side documents them in [python/README.md](./python/README.md#parameters).

### Multi-GPU with MPI + NCCL

The same binary auto-detects an MPI launch and switches to the distributed solver &mdash; MPI for control-plane messaging, NCCL for on-device collectives:

```bash
# 4 GPUs, process grid auto-selected
mpirun -n 4 ./build/cardal -f problem.dat-s -O ./output

# Explicit row x rank x cone grid
mpirun -n 4 ./build/cardal -f problem.dat-s --grid_size 2,2,1
mpirun -n 8 ./build/cardal -f problem.dat-s --grid_size 1,1,8
```

### Troubleshooting

- **`nvcc: command not found`, or CUDA too old.** Export `CUDACXX=/usr/local/cuda-12.6/bin/nvcc` before invoking `cmake`.
- **CLI build fails with missing MPI.** MPI is on by default; reconfigure with `-DENABLE_MPI=OFF` for a single-GPU build.
- **QUBO input rejected.** The main `cardal` binary refuses QUBO instances; use `cardal_qubo` (`./build/cardal_qubo -h`).

## Citation

A manuscript describing CARDAL is in preparation. Until then, cite the
software metadata in [CITATION.cff](CITATION.cff).

## License

Copyright 2026 Hongpei Li.

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
