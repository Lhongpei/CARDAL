# Parameters

CARDAL shares solver defaults across the Python package, public C API, and
native CLI. Python and C use the same field names; the CLI uses hyphenated
options.

## Solver parameters

| Python / C field | CLI option | Type | Default | Description |
|:--|:--|:--|:--|:--|
| `eps_primal_relative` | `--eps-primal` | float | `1e-4` | Relative primal residual tolerance |
| `eps_dual_relative` | `--eps-dual` | float | `1e-4` | Relative dual residual tolerance |
| `eps_optimal_relative` | `--eps-gap` | float | `1e-4` | Relative objective-gap tolerance |
| `time_sec_limit` | `--time-limit` | float | `3600.0` | Wall-clock budget in seconds; `0` disables |
| `iteration_limit` | `--outer-iters` | int | `20000000` | Outer ALM iteration cap |
| `initial_rank` | `--rank` | int | `-1` | Initial rank; `-1` selects automatically |
| `max_rank` | `--max-rank` | int | `-1` | Per-cone rank cap; `-1` uses the Pataki bound |
| `augmentation_mode` | `--augmentation-mode` | enum / string | `random` | Rank augmentation backend |
| `lbfgs_history_size` | `--lbfgs-hist` | int | `5` | L-BFGS history depth |
| `penalty_factor` | `--penalty-fac` | float | `3.3` | Multiplicative ALM penalty update |
| `initial_penalty_coef` | `--init-penalty` | float | `-1.0` | Initial penalty; `-1.0` uses \(2/\sqrt{N}\) |
| `max_penalty_coef` | `--max-penalty` | float | `5e5` | Maximum penalty coefficient |
| `inner_iterations_limit` | `--inner-iters` | int | `30000` | Inner L-BFGS cap per outer step |
| `verbose` | `--verbose` | int | `2` | Logging level from `0` to `3` |

The CLI convenience option `--eps VALUE` sets all three relative tolerances.
The solver reports `OPTIMAL` only after the primal, dual, and objective-gap
criteria are all satisfied.

## Rank augmentation

Python accepts the following strings for `augmentation_mode`:

| Value | C enum |
|:--|:--|
| `random` | `CARDAL_AUGMENTATION_RANDOM` |
| `qp` | `CARDAL_AUGMENTATION_QP` |
| `closed-form` | `CARDAL_AUGMENTATION_CLOSED_FORM` |
| `sdp` | `CARDAL_AUGMENTATION_SDP` |

The default initial rank is selected from the problem size. The default maximum
rank is the Pataki-style bound

\[
\left\lceil\frac{\sqrt{8m+1}-1}{2}\right\rceil.
\]

## Preconditioning and scaling

| Python / C field | CLI option | Type | Default | Description |
|:--|:--|:--|:--|:--|
| `l_inf_ruiz_iterations` | `--l-inf-ruiz-iter` | int | `10` | L-infinity Ruiz scaling iterations; `0` disables |
| `pock_chambolle_rescaling` | `--no-pock-chambolle` | bool / negative flag | `True` | Enable Pock-Chambolle scaling |
| `pock_chambolle_alpha` | `--pock-chambolle-alpha` | float | `1.0` | Pock-Chambolle exponent |
| `bound_objective_rescaling` | `--no-bound-obj-rescaling` | bool / negative flag | `True` | Enable bound-objective scaling |
| `psd_scale_mode` | `--psd-scale-mode` | enum / string | `per-element` | PSD scaling granularity |

Python example:

```python
result = model.solve(
    l_inf_ruiz_iterations=10,
    pock_chambolle_rescaling=True,
    pock_chambolle_alpha=1.0,
    bound_objective_rescaling=True,
    psd_scale_mode="per-element",
)
```

To disable all scaling in Python or C:

```python
result = model.solve(
    l_inf_ruiz_iterations=0,
    pock_chambolle_rescaling=False,
    bound_objective_rescaling=False,
)
```

The equivalent CLI shortcut is:

```bash
./build/cardal -f problem.dat-s --no-scaling
```

`psd_scale_mode` accepts `per-element` or `per-cone`. The corresponding C enum
values are `CARDAL_PSD_SCALE_PER_ELEMENT` and
`CARDAL_PSD_SCALE_PER_CONE`.

## CLI-only parameters

| Option | Default | Description |
|:--|:--|:--|
| `--grid-size row,rank,cone` | Automatic | Distributed process-grid topology |
| `--shuffle` | `col` | Distributed constraint ordering: `none`, `uniform`, `block`, or `col` |
| `--output-dir` | None | Write the summary and primal/dual solution files |

These settings control execution or CLI output rather than the mathematical
solver configuration, so they are not members of the single-GPU Python
parameter dictionary.

## Inspect defaults

Python:

```python
print(cardal.Model.default_params())
```

C:

```c
cardal_params params;
cardal_default_params(&params);
```

CLI:

```bash
./build/cardal -h
```
