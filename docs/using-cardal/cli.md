# Command-Line Interface

The native build provides two executables:

- `cardal` for SDP files and generated SDP instances
- `cardal_qubo` for QUBO input, SDP relaxation, and rounding

## Solve a file

```bash
./build/cardal -f problem.dat-s
```

Add an output directory to retain the summary and solution:

```bash
./build/cardal -f problem.dat-s -O results
```

CARDAL always writes these four files under `results`:

```text
<instance>_summary.txt
<instance>_primal_factor.txt
<instance>_rank_list.txt
<instance>_dual_solution.txt
```

Problems with linear blocks additionally produce
`<instance>_lp_primal.txt` and/or `<instance>_free_primal.txt`.

See [Results and Outputs](results.md#cli-output-files) for the factor layout.

## Set tolerances

Use `--eps` to set all three relative tolerances together:

```bash
./build/cardal -f problem.dat-s --eps 1e-4
```

Or set them independently:

```bash
./build/cardal -f problem.dat-s \
  --eps-primal 1e-4 \
  --eps-dual 1e-4 \
  --eps-gap 1e-4
```

## Select rank behavior

```bash
./build/cardal -f problem.dat-s \
  --rank 16 \
  --max-rank 128 \
  --augmentation-mode qp
```

Accepted augmentation modes are `random`, `qp`, `closed-form`, and `sdp`.
Leaving rank arguments unset uses CARDAL's automatic initial rank and rank cap.

## Generate test problems

The main executable can generate several problem families:

```bash
# Max-Cut: nodes, edge density
./build/cardal -g maxcut,30000,1e-3

# Sensor network localization: nodes, anchors, dimension, radius, noise
./build/cardal -g snl,10000,4,2,0.05,0.0

# Quantum-order relaxation: moment-matrix size, level
./build/cardal -g quantum_order,64,3
```

Generators are CLI-only. To use generated data from Python, write or convert it
to a supported problem file first.

## Scaling controls

```bash
./build/cardal -f problem.dat-s \
  --l-inf-ruiz-iter 10 \
  --pock-chambolle-alpha 1.0 \
  --psd-scale-mode per-element
```

Individual stages can be disabled with `--no-pock-chambolle` and
`--no-bound-obj-rescaling`; `--no-scaling` disables all scaling stages.

The [Parameters](../parameters.md) page contains the complete option mapping.
The executable remains the authoritative parser reference:

```bash
./build/cardal -h
```

## QUBO executable

`cardal_qubo` accepts either a QUBO file or a generated random QUBO:

```bash
./build/cardal_qubo -f model.qubo

./build/cardal_qubo \
  --size 10000 \
  --density 1e-3 \
  --mode chordal \
  --trials 4096
```

QUBO-specific settings include:

| Option | Description | Default |
|:--|:--|:--|
| `--mode` | `dense` single-cone or `chordal` multi-cone relaxation | `dense` |
| `--seed` | Random-QUBO seed | `0` |
| `--trials` | Random-hyperplane rounding trials | `4096` |
| `--ls-iters` | Per-trial local-search cap; `0` skips local search | Automatic |
| `--round-seed` | Rounding RNG seed | `42` |

The main `cardal` executable intentionally rejects QUBO input.

## Distributed launch

An MPI launch is detected automatically:

```bash
mpirun -n 4 ./build/cardal -f problem.dat-s
```

See [Multi-GPU](../multi-gpu.md) for process-grid semantics.
