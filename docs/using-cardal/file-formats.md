# File Formats

CARDAL detects text and MATLAB formats from their headers. PDSDP NPZ input is
selected by the `.npz` suffix.

## Supported formats

| Format | Typical suffix | Native CLI | PyPI Python package | Notes |
|:--|:--|:--:|:--:|:--|
| SDPA sparse text | `.dat-s` | Yes | Yes | Standard SDPA sparse format |
| Gzip-compressed SDPA | `.dat-s.gz` | Yes | Yes | Decompressed transparently |
| MATLAB v5, SeDuMi | `.mat` | Yes | No | Requires a build with `ENABLE_MATIO=ON` |
| MATLAB v5, SDPT3 | `.mat` | Yes | No | Requires a build with `ENABLE_MATIO=ON` |
| PDSDP-style NPZ | `.npz` | Yes | Yes | Schema documented below |

The PyPI build disables MATIO to keep the package build self-contained.
Source builds enable MATIO by default and either use a system installation or
build `libmatio`.

## SDPA

CARDAL accepts standard sparse SDPA text, including gzip-compressed files.
Block structure, right-hand sides, objective entries, and constraint entries
follow the SDPA convention. The reader converts the SDPA objective sign to
CARDAL's internal minimization convention.

```bash
./build/cardal -f instance.dat-s
./build/cardal -f instance.dat-s.gz
```

## MATLAB

MATLAB v5 containers are inspected for either:

- SeDuMi variables such as `At` or `A`, `b`, `c`, and `K`
- SDPT3 block data such as `blk`, `At`, `C`, and `b`

Configure a native build with MAT support:

```bash
cmake -S . -B build -DENABLE_MATIO=ON
cmake --build build -j
```

## PDSDP NPZ

The NPZ archive uses zero-based indices:

| Entry | Dtype and shape | Required | Meaning |
|:--|:--|:--:|:--|
| `tI_size` | `int32` or `int64`, `(num_cones,)` | Yes | PSD block dimensions |
| `b` | `float64`, `(m,)` | Yes | Equality right-hand side |
| `A` | `float64`, `(nnz_a, 5)` | Yes | `[constraint, block, col, row, value]` |
| `C` | `float64`, `(nnz_c, 4)` | No | `[block, col, row, value]` |
| `c` | `float64`, `(lp_dim,)` | No | Nonnegative LP objective |
| `a` | `float64`, `(nnz_lp, 3)` | No | `[constraint, lp_column, value]` |

Each symmetric PSD entry is stored once. Both C- and Fortran-ordered NumPy
arrays are accepted.

## In-memory formats

Python and C callers do not need to serialize a file:

- Python `Model.set_problem` accepts dense NumPy and SciPy sparse blocks.
- Python `Model.set_problem_coo` accepts parallel COO arrays.
- C `cardal_build_problem` accepts `cardal_problem_data`.

These APIs use lower-triangular `row, col` coordinates and zero-based indices.
See [Problem Definition](problem-definition.md) for the shared matrix
conventions.

