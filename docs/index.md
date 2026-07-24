---
hide:
  - toc
---

<section class="cardal-masthead" markdown="1">

<div class="cardal-kicker">GPU-accelerated semidefinite optimization</div>

# CARDAL

<div class="cardal-lead">
CARDAL is an open-source GPU low-rank solver for large-scale semidefinite
programs, with a NumPy-friendly Python interface, a stable C API, a command-line
solver, and distributed multi-GPU execution.
</div>

<div class="cardal-badges">
  <a href="https://pypi.org/project/cardal/"><img alt="PyPI version" src="https://badge.fury.io/py/cardal.svg"></a>
  <a href="https://arxiv.org/abs/2607.17933"><img alt="arXiv paper" src="https://img.shields.io/badge/arXiv-2607.17933-b31b1b.svg"></a>
  <a href="https://github.com/Lhongpei/CARDAL/blob/main/LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/License-Apache_2.0-245a78.svg"></a>
  <a href="https://github.com/Lhongpei/CARDAL"><img alt="GitHub repository" src="https://img.shields.io/badge/GitHub-CARDAL-173f35.svg"></a>
</div>

<div class="cardal-actions">
  <a class="md-button md-button--primary" href="getting-started/">Get started</a>
  <a class="md-button" href="parameters/">View parameters</a>
  <a class="md-button" href="https://github.com/Lhongpei/CARDAL">Source code</a>
</div>

</section>

## Install and solve

Install the single-GPU Python package:

```bash
pip install cardal
```

```python
import cardal

model = cardal.Model()
model.read_file("problem.dat-s")

result = model.solve(
    eps_primal_relative=1e-4,
    eps_dual_relative=1e-4,
    eps_optimal_relative=1e-4,
)

print(result.summary())
```

For the C/CUDA command-line solver, build from source and run:

```bash
./build/cardal -f problem.dat-s -O results
```

## Choose an interface

<div class="cardal-link-grid">
  <div class="cardal-link">
    <strong><a href="using-cardal/python/">Python</a></strong>
    <p>Read files or construct SDPs from NumPy and SciPy arrays.</p>
  </div>
  <div class="cardal-link">
    <strong><a href="using-cardal/c-api/">C API</a></strong>
    <p>Use opaque problem and result handles from native applications.</p>
  </div>
  <div class="cardal-link">
    <strong><a href="using-cardal/cli/">Command line</a></strong>
    <p>Solve files, generate test problems, and record summaries.</p>
  </div>
  <div class="cardal-link">
    <strong><a href="multi-gpu/">Multi-GPU</a></strong>
    <p>Distribute work across constraint, rank, and cone axes with MPI and NCCL.</p>
  </div>
</div>

## Problem class

CARDAL solves block-structured SDPs with an optional nonnegative LP tail:

\[
\begin{aligned}
\min_X \quad & \langle C, X \rangle \\
\mathrm{s.t.}\quad & \langle A_i, X \rangle = b_i,
  \qquad i=1,\ldots,m, \\
& X = \mathrm{blkdiag}(X_1,\ldots,X_p,x_{\mathrm{LP}}),\\
& X_c \succeq 0,\qquad x_{\mathrm{LP}}\ge 0.
\end{aligned}
\]

Each PSD block is represented through a Burer-Monteiro factor
\(X_c=V_cV_c^\top\). See [Problem Definition](using-cardal/problem-definition.md)
for storage conventions and supported cones.

## Learn more

The algorithm is described in
[A Curvature-Aware Rank-Adaptive Distributed Augmented-Lagrangian Solver for
Large-Scale SDPs](https://arxiv.org/abs/2607.17933). Citation metadata and
BibTeX are available on the [Paper and Citation](paper-and-citation.md) page.

