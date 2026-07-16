# CARDAL: A Curvature-Aware Rank-Adaptive Distributed Augmented-Lagrangian Solver for Large-Scale SDPs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Distributed](https://img.shields.io/badge/Multiple_GPUs-optional-brightgreen.svg)](https://www.open-mpi.org/) [![SDPLIB](https://img.shields.io/badge/Input-SDPA%20%7C%20SeDuMi%20%7C%20SDPT3-lightgrey.svg)](https://github.com/vsdp/SDPLIB) [![Interface](https://img.shields.io/badge/Interface-Python%20%7C%20C-lightyellow.svg)](https://github.com/vsdp/SDPLIB)

**CARDAL** is a curvature-aware, rank-adaptive, distributed augmented-Lagrangian solver for large-scale semidefinite programs. It operates on a Burer-Monteiro low-rank factorization and supports both single- and multi-GPU systems. CARDAL targets semidefinite programs whose optimal solutions are (or are expected to be) low-rank.

A companion manuscript of CARDAL is in preparation.



### Problem Formulation

CARDAL solves standard-form semidefinite programs with block-diagonal PSD variables and an optional nonnegative LP block:

$$
\begin{aligned}
\min_{X} \quad & \langle C, X \rangle \\
\text{s.t.} \quad & \langle A_i, X \rangle = b_i, \quad i = 1,\dots,m, \\
                  & X = \operatorname{blkdiag}(X_1, \dots, X_p, x_{\text{LP}}), \\
                  & X_c \succeq 0 \text{ for } c=1,\dots,p, \qquad x_{\text{LP}} \ge 0.
\end{aligned}
$$

$C$ and each $A_i$ are symmetric block-diagonal matrices. Every matrix is stored by its **lower triangle only**, and off-diagonal entries are **not** implicitly doubled &mdash; the same convention applies to the Python API. Each PSD block $X_c \in \mathbb{R}^{n_c \times n_c}$ is stored as a factor $V_c \in \mathbb{R}^{n_c \times r_c}$ with $X_c = V_c V_c^\top$.

## Features

- **GPU-native.** Most operations are implemented natively on GPUs.
- **Multi-GPU.** MPI + NCCL parallelization across constraint, rank, and cone axes, enabled by default (`-DENABLE_MPI=OFF` to opt out).
- **Algorithm.** Adaptive rank augmentation, negative curvature escape.

## Requirements

- **Platform:** Linux x86_64 with an NVIDIA GPU and CUDA 12.x (12.6 or newer recommended).
- **Build tools:** CMake 3.20 or newer, a CUDA-compatible C++17 compiler, and Make or Ninja.
- **Distributed (optional):** MPI-3.1 and NCCL 2.18 or newer for multi-GPU support.

When multiple CUDA versions are installed, select one with `CUDACXX`, for example `CUDACXX=/usr/local/cuda-12.6/bin/nvcc`.

## Installation

### Build from source (C++ CLI)

Multi-GPU support (MPI + NCCL) is **enabled by default**; add `-DENABLE_MPI=OFF` for a single-GPU-only binary. If the default `nvcc` is outdated or missing, prefix the first `cmake` invocation with `CUDACXX=/path/to/nvcc`.

```bash
git clone https://github.com/Lhongpei/CARDAL.git
cd CARDAL
cmake -S . -B build
cmake --build build --clean-first
```

The main binary lands at `./build/cardal`. A sibling binary `./build/cardal_qubo` (built when `CARDAL_BUILD_QUBO=ON`, default) specializes in QUBO-lifted SDPs, takes the same core CLI flags with a QUBO-specific input parser (`./build/cardal_qubo -f model.qubo`), and the main `cardal` binary rejects QUBO inputs. Run `./build/cardal_qubo -h` for its full syntax.

### Python package

CARDAL also ships a NumPy-friendly Python front-end (single-GPU only), see [python/README.md](./python/README.md).

## Quickstart

### Solve a problem file

CARDAL auto-detects SDPA, MATLAB, and PDSDP input formats:

```bash
./build/cardal -f problem.dat-s -O ./output
```

The solve summary is printed to the terminal. Passing `-O ./output` also writes it to `./output/<instance>_summary.txt` for post-processing.

### CLI

```bash
# From a file (SDPA, MATLAB, or PDSDP; format auto-detected)
./build/cardal -f problem.dat-s -O ./output
```

### Python Interface

```python
import cardal

m = cardal.Model()
m.read_file("problem.dat-s")                       # or .dat-s.gz / .mat / .npz
result = m.solve(
    time_sec_limit=60.0,
    eps_primal_relative=1e-4,
    eps_dual_relative=1e-4,
    eps_optimal_relative=1e-4,
)
print(result.status, result.primal_objective, result.rel_objective_gap)
```

## Advanced Usage

### CLI reference

```
./build/cardal -f <PATH> [OPTIONS]
```

- `-f, --file <path>` reads a problem file (SDPA `.dat-s` / `.dat-s.gz`, MATLAB `.mat`, PDSDP `.npz`; auto-detected).

#### Solver parameters

| Option | Type | Description | Default |
|:-------|:-----|:------------|:--------|
| `-e, --eps` | float | Set the primal, dual, and objective-gap tolerances together. | `1e-4` |
| `--eps-primal` | float | Relative primal residual tolerance. | `1e-4` |
| `--eps-dual` | float | Relative dual residual tolerance. | `1e-4` |
| `--eps-gap` | float | Relative objective-gap tolerance. | `1e-4` |
| `-r, --rank` | int | Initial Burer-Monteiro rank. | `ceil(2 log m)` |
| `-R, --max-rank` | int | Maximum rank of each PSD block. | Pataki bound |
| `-i, --inner-iters` | int | L-BFGS iteration limit per outer step. | `30000` |
| `-o, --outer-iters` | int | Augmented-Lagrangian outer iteration limit. | `20000000` |
| `-p, --penalty-fac` | float | Penalty coefficient multiplier. | `1.2` |
| `-c, --init-penalty` | float | Initial penalty coefficient. | `2 / sqrt(N)` |
| `-M, --max-penalty` | float | Maximum penalty coefficient. | `5e5` |
| `-L, --lbfgs-hist` | int | L-BFGS history size. | `5` |
| `-T, --time-limit` | float | Wall-clock limit in seconds; `0` disables it. | `3600` |
| `-v, --verbose` | int | Log level: `0` silent, `1` summary, `2` iterations, `3` debug. | `2` |
| `-O, --output-dir` | path | Write `<instance>_summary.txt` to this directory. | None |

#### Distributed and scaling parameters

| Option | Type | Description | Default |
|:-------|:-----|:------------|:--------|
| `-z, --grid_size` | string | MPI grid as `row,rank,cone`; dimensions must multiply to the MPI process count. | Auto |
| `--shuffle` | string | Distributed constraint ordering: `none`, `uniform`, `block`, or `col`. | `col` |
| `--l_inf_ruiz_iter` | int | Number of L-infinity Ruiz scaling iterations; `0` disables them. | `10` |
| `--pock_chambolle_alpha` | float | Pock-Chambolle scaling exponent. | `1.0` |
| `--no_pock_chambolle` | flag | Disable Pock-Chambolle scaling. | Off |
| `--no_bound_obj_rescaling` | flag | Disable bound-objective rescaling. | Off |
| `--psd_scale_mode` | string | PSD scaling mode: `per-element` or `per-cone`. | `per-element` |
| `--no_scaling` | flag | Disable all scaling stages. | Off |

Run `./build/cardal -h` for the authoritative CLI help. The Python interface
uses the same solver defaults; its keyword parameters are documented in
[python/README.md](./python/README.md#parameters).

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
