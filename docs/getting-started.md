# Getting Started

This guide covers the shortest path to a working Python installation or a
native CARDAL executable.

## Requirements

- Linux x86_64
- An NVIDIA GPU
- CUDA 12.x; CUDA 12.6 or newer is recommended
- Python 3.9 or newer for the Python package
- CMake 3.20 or newer and a CUDA-compatible C++17 compiler for source builds
- MPI 3.1 and NCCL 2.18 or newer for distributed builds

When several CUDA toolkits are installed, select one explicitly:

```bash
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
```

## Install Python

The PyPI package provides the single-GPU Python interface:

```bash
python -m pip install cardal
```

CARDAL compiles its CUDA extension during installation. The CUDA toolkit,
including `nvcc`, must therefore be available on the machine performing the
installation.

SciPy is optional and only needed for `scipy.sparse` input:

```bash
python -m pip install "cardal[scipy]"
```

Verify the installation:

```bash
python -c "import cardal; print(cardal.Model.default_params())"
```

## Build the native solver

Clone and build with the default configuration:

```bash
git clone https://github.com/Lhongpei/CARDAL.git
cd CARDAL
cmake -S . -B build
cmake --build build -j
```

The default source configuration builds:

- `build/cardal`, the SDP command-line solver
- `build/cardal_qubo`, the QUBO-specific command-line solver
- the static CARDAL core library
- MPI and NCCL support

For a single-GPU build without MPI:

```bash
cmake -S . -B build -DENABLE_MPI=OFF
cmake --build build -j
```

To omit the QUBO executable:

```bash
cmake -S . -B build -DCARDAL_BUILD_QUBO=OFF
```

## First Python solve

```python
import cardal

model = cardal.Model()
model.read_file("problem.dat-s")
result = model.solve(
    time_sec_limit=60.0,
    eps_primal_relative=1e-4,
    eps_dual_relative=1e-4,
    eps_optimal_relative=1e-4,
    verbose=1,
)

print(result.status)
print(result.primal_objective)
print(result.rel_objective_gap)
```

`Model` may be reused: loading or constructing another problem replaces its
current problem handle, while each returned `Result` remains independent.

## First CLI solve

```bash
./build/cardal -f problem.dat-s -O results
```

The solver prints a summary to the terminal. With `-O results`, it also writes
the summary, low-rank primal factor, per-cone rank list, and dual solution to
`results`.

Run the built-in help to inspect the executable's accepted options:

```bash
./build/cardal -h
```

## Where to continue

- [Using CARDAL](using-cardal/problem-definition.md) explains problem input,
  interfaces, file formats, and outputs.
- [Parameters](parameters.md) maps every shared setting across Python, C, and
  the CLI.
- [Multi-GPU](multi-gpu.md) covers distributed builds and launches.
