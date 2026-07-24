# C API

CARDAL exposes a C-compatible ABI in
[`include/cardal.h`](https://github.com/Lhongpei/CARDAL/blob/main/include/cardal.h).
It uses POD parameter structs and opaque problem and result handles.

## Minimal solve

```c
#include <stdio.h>
#include "cardal.h"

int main(void) {
  cardal_error error = CARDAL_OK;
  cardal_problem *problem = cardal_read_sdpa("problem.dat-s", &error);
  if (problem == NULL) {
    fprintf(stderr, "failed to read problem: %d\n", (int)error);
    return 1;
  }

  cardal_params params;
  cardal_default_params(&params);
  params.time_sec_limit = 300.0;
  params.eps_primal_relative = 1e-4;
  params.eps_dual_relative = 1e-4;
  params.eps_optimal_relative = 1e-4;

  cardal_result *result = cardal_solve(problem, &params);
  if (result == NULL) {
    cardal_problem_free(problem);
    return 1;
  }

  printf("status: %d\n", (int)cardal_result_status(result));
  printf("primal objective: %.8e\n",
         cardal_result_primal_objective(result));

  cardal_result_free(result);
  cardal_problem_free(problem);
  return 0;
}
```

Always initialize `cardal_params` with `cardal_default_params` before changing
individual fields.

## Build from arrays

Use `cardal_build_problem` to construct an SDP without a file:

```c
cardal_problem_data data = {0};

data.num_constraints = m;
data.num_cones = p;
data.lp_dim = lp_dim;
data.blk_dims = block_dims;

data.nnz_c = nnz_c;
data.c_cone_ind = c_cone;
data.c_row_ind = c_row;
data.c_col_ind = c_col;
data.c_val = c_val;

data.nnz_a = nnz_a;
data.a_constr_ind = a_constraint;
data.a_cone_ind = a_cone;
data.a_row_ind = a_row;
data.a_col_ind = a_col;
data.a_val = a_val;

data.lp_obj = lp_obj;
data.nnz_lp = nnz_lp;
data.lp_constr_ind = lp_constraint;
data.lp_col_ind = lp_col;
data.lp_val = lp_val;
data.b = b;

cardal_error error = CARDAL_OK;
cardal_problem *problem = cardal_build_problem(&data, &error);
```

All coordinate indices are zero-based. PSD coordinates contain one triangle
per symmetric block, normally `row >= col`. CARDAL copies all supplied arrays;
the caller may release them after `cardal_build_problem` returns.

## Problem functions

| Function | Purpose |
|:--|:--|
| `cardal_read_sdpa` | Read a supported problem file |
| `cardal_build_problem` | Build a problem from COO arrays |
| `cardal_problem_num_constraints` | Return \(m\) |
| `cardal_problem_num_cones` | Return the number of PSD blocks |
| `cardal_problem_num_variables` | Return the stored variable count |
| `cardal_problem_lp_dim` | Return the nonnegative LP dimension |
| `cardal_problem_get_block_dims` | Copy PSD block dimensions |
| `cardal_problem_free` | Release a problem handle |

## Solve and result functions

`cardal_solve(problem, NULL)` uses all defaults. Passing a populated
`cardal_params` overrides them.

Result accessors return scalar values or borrowed pointers. Borrowed arrays
remain valid until `cardal_result_free`:

| Function group | Values |
|:--|:--|
| `cardal_result_status` | Termination status |
| `cardal_result_*objective*` | Primal objective, dual objective, and gap |
| `cardal_result_rel_*` | Relative primal residual, dual residual, and gap |
| `cardal_result_*iters` | Outer and total inner iterations |
| `cardal_result_primal_factor` | Flattened low-rank factors |
| `cardal_result_dual` | Dual vector |
| `cardal_result_rank_list` | Final rank of each PSD block |

See [Results and Outputs](results.md) for layouts and status meanings.

## Ownership

| Object | Owner | Release |
|:--|:--|:--|
| Input arrays passed to `cardal_build_problem` | Caller | Any time after the call returns |
| `cardal_problem *` | Caller | `cardal_problem_free` |
| `cardal_result *` | Caller | `cardal_result_free` |
| Arrays returned by result accessors | Result handle | Do not free directly |

Cancellation is process-wide:

```c
cardal_request_cancel();
cardal_clear_cancel();
int pending = cardal_cancel_requested();
```

The [Parameters](../parameters.md) page documents `cardal_params` and its CLI
and Python equivalents.

