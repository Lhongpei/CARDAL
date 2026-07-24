# Problem Definition

## Standard form

CARDAL solves minimization-form semidefinite programs

\[
\begin{aligned}
\min_X \quad & \langle C, X\rangle\\
\mathrm{s.t.}\quad
& \langle A_i,X\rangle=b_i,\qquad i=1,\ldots,m,\\
& X=\mathrm{blkdiag}(X_1,\ldots,X_p,x_{\mathrm{LP}}),\\
& X_c\succeq0,\qquad c=1,\ldots,p,\\
& x_{\mathrm{LP}}\geq0.
\end{aligned}
\]

Here \(X_c\in\mathbb{S}^{n_c}\) are PSD blocks and the optional
\(x_{\mathrm{LP}}\in\mathbb{R}^{d_{\mathrm{LP}}}\) is a nonnegative linear
block.

## Low-rank representation

Each PSD block is represented as

\[
X_c=V_cV_c^\top,\qquad
V_c\in\mathbb{R}^{n_c\times r_c}.
\]

CARDAL stores and optimizes the factors \(V_c\), not dense \(n_c\times n_c\)
primal matrices. A solve returns the flattened factors together with the final
per-cone ranks.

## Supported cones

| Cone | Representation |
|:--|:--|
| Positive semidefinite | One or more symmetric blocks declared by `block_dims` |
| Nonnegative orthant | One optional LP tail declared by `lp_dim` |

Second-order, exponential, power, and generic bounded cones are not currently
supported.

## Symmetric matrix storage

Provide exactly one triangle of every symmetric PSD matrix. The Python and C
in-memory interfaces use lower-triangular coordinates:

```text
row >= col
```

Off-diagonal entries are stored once and are not expected in both triangles.
For a dense or SciPy matrix passed to `Model.set_problem`, entries above the
diagonal are ignored.

!!! warning "Do not duplicate symmetric entries"
    Supplying both \((i,j)\) and \((j,i)\) does not express a more accurate
    symmetric matrix. Canonicalize each matrix to one triangle before passing
    it to CARDAL.

## Objective sign

The in-memory Python and C construction APIs solve the objective exactly as
provided:

\[
\min_X\langle C,X\rangle.
\]

SDPA files use their own \(F_0\) convention. The file reader converts that
convention to CARDAL's internal minimization objective. Do not apply an
additional sign change when using `read_file` or `cardal_read_sdpa`.

## Input routes

The same mathematical problem can enter CARDAL through several routes:

| Route | Best suited for |
|:--|:--|
| `Model.read_file` / `cardal_read_sdpa` / `cardal -f` | Existing SDPA, MATLAB, or PDSDP files |
| `Model.set_problem` | Dense NumPy or SciPy block matrices |
| `Model.set_problem_coo` | Large sparse problems already stored as triplets |
| `cardal_build_problem` | Native applications using C-owned arrays |

The [File Formats](file-formats.md) page documents file schemas. The
[Python](python.md), [C API](c-api.md), and [CLI](cli.md) pages show the
interface-specific calls.

