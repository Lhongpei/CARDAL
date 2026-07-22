/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 *
 * CARDAL public C ABI.
 *
 * This header is the stable contract between the CARDAL C/CUDA core and any
 * outside caller (Python bindings, third-party integrations, downstream
 * language bindings). It is `extern "C"` and uses only POD types and opaque
 * handles — no C++ headers cross this boundary, so internal refactors of
 * `cardal_parameters_t` / `compressed_sdp_problem_t` / `sdp_result_t` do not
 * break bindings.
 *
 * Ownership model:
 *   - Every _new / _read / _solve returns a handle owned by the caller.
 *   - Free with the matching _free.
 *   - Result accessors return borrowed pointers valid until cardal_result_free.
 */

#ifndef CARDAL_H
#define CARDAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * Opaque handles
 * ----------------------------------------------------------------------- */

typedef struct cardal_problem cardal_problem;
typedef struct cardal_result  cardal_result;

typedef enum {
  CARDAL_AUGMENTATION_RANDOM = 0,
  CARDAL_AUGMENTATION_QP = 1,
  CARDAL_AUGMENTATION_CLOSED_FORM = 2,
  CARDAL_AUGMENTATION_SDP = 3
} cardal_augmentation_mode;

typedef enum {
  CARDAL_PSD_SCALE_PER_ELEMENT = 0,
  CARDAL_PSD_SCALE_PER_CONE = 1
} cardal_psd_scale_mode;

/* -----------------------------------------------------------------------
 * Parameters (POD, ABI-stable). All doubles / ints; no pointers.
 *
 * Special values:
 *   double  -1.0  => auto / disabled (per field docstring)
 *   int     -1    => auto / disabled (per field docstring)
 *   time_sec_limit == 0.0  => no time limit
 * ----------------------------------------------------------------------- */

typedef struct {
  /* Independent relative termination tolerances. The solver stops when the
     primal residual, dual residual, and objective gap all clear. */
  double eps_primal_relative;       /* default 1e-4 */
  double eps_dual_relative;         /* default 1e-4 */
  double eps_optimal_relative;      /* objective-gap tolerance; default 1e-4 */
  double time_sec_limit;            /* default 3600.0; 0.0 disables */
  int    iteration_limit;           /* default 20000000 (outer ALM cap) */

  /* Burer–Monteiro rank management. -1 => auto. */
  int    initial_rank;              /* auto: ceil(2*log m) */
  int    max_rank;                  /* auto: Pataki bound */
  int    augmentation_mode;         /* cardal_augmentation_mode; default random */

  /* Inner solver + penalty schedule. */
  int    lbfgs_history_size;        /* default 5 */
  double penalty_factor;            /* default 3.3 */
  double initial_penalty_coef;      /* -1.0 => auto (2/sqrt N) */
  double max_penalty_coef;          /* default 5e5 */
  long   inner_iterations_limit;    /* default 30000 (per outer iter cap) */

  /* Logging. 0=silent, 1=banner+summary, 2=+iter table, 3=debug. */
  int    verbose;

  /* Preconditioning and scaling. Boolean fields use 0=false, 1=true. */
  int    l_inf_ruiz_iterations;       /* default 10; 0 disables */
  int    pock_chambolle_rescaling;    /* default 1 */
  double pock_chambolle_alpha;        /* default 1.0 */
  int    bound_objective_rescaling;   /* default 1 */
  int    psd_scale_mode;              /* cardal_psd_scale_mode */
} cardal_params;

/* -----------------------------------------------------------------------
 * Status codes
 * ----------------------------------------------------------------------- */

typedef enum {
  CARDAL_STATUS_UNSPECIFIED     = 0,
  CARDAL_STATUS_OPTIMAL         = 1,
  CARDAL_STATUS_TIME_LIMIT      = 2,
  CARDAL_STATUS_ITERATION_LIMIT = 3,
  CARDAL_STATUS_USER_INTERRUPT  = 4
} cardal_status;

/* -----------------------------------------------------------------------
 * Cooperative cancellation
 *
 * Set the cancel flag from any thread / signal handler while a solve is
 * running; the ALM outer loop polls it at each iteration boundary and
 * terminates cleanly with CARDAL_STATUS_USER_INTERRUPT.
 *
 * The flag is process-wide. If two solves run concurrently in the same
 * process, both will terminate; callers that need per-solve cancellation
 * should serialize.
 * ----------------------------------------------------------------------- */

void cardal_request_cancel(void);
void cardal_clear_cancel(void);
int  cardal_cancel_requested(void);

/* -----------------------------------------------------------------------
 * Error codes returned by APIs that don't naturally return a handle.
 * ----------------------------------------------------------------------- */

typedef enum {
  CARDAL_OK             = 0,
  CARDAL_E_NULL_ARG     = 1,
  CARDAL_E_FILE_IO      = 2,
  CARDAL_E_PARSE        = 3,
  CARDAL_E_INTERNAL     = 4
} cardal_error;

/* -----------------------------------------------------------------------
 * Parameter defaults
 * ----------------------------------------------------------------------- */

/* Populate `p` with the same defaults as the CLI. `p` must be non-NULL. */
void cardal_default_params(cardal_params *p);

/* -----------------------------------------------------------------------
 * Problem construction
 * ----------------------------------------------------------------------- */

/* Load an SDP from a file (SDPA .dat-s / .dat-s.gz / MATLAB / PDSDP NPZ —
 * whichever format the parser autodetects).
 *
 * Returns NULL on failure. If `err_out` is non-NULL, receives a CARDAL_E_*
 * code identifying the failure mode.
 * Free the returned handle with cardal_problem_free(). */
cardal_problem *cardal_read_sdpa(const char *path, cardal_error *err_out);

/* Build a problem directly from user-supplied COO triplet arrays.
 *
 * All integer indices are 0-indexed. The caller retains ownership of every
 * pointer — this function copies all input arrays internally, so the input
 * memory may be released as soon as it returns.
 *
 * Sign convention:
 *   `c_val`  are the entries of the primal cost matrix C. The returned
 *            problem minimizes <C, X>. No sign flipping is performed;
 *            this differs from cardal_read_sdpa() which flips the SDPA
 *            file's F0 to C internally.
 *   `a_val`  are the entries of the constraint matrices A_i (constraint i
 *            is <A_i, X> = b_i).
 *   `lp_obj` is the cost vector on the LP block: min <lp_obj, x_LP>.
 *
 * PSD triangle convention:
 *   Provide only one triangle per block (typically lower: row >= col).
 *   The solver treats each block as symmetric; off-diagonal entries are
 *   NOT doubled internally. If you provide both triangles the objective
 *   will be double-counted.
 *
 * On success returns a fresh cardal_problem handle (free with
 * cardal_problem_free). On failure returns NULL and, if err_out is
 * non-NULL, sets it to a CARDAL_E_* code.
 */
typedef struct {
  /* Sizes */
  int  num_constraints;   /* m — length of b */
  int  num_cones;         /* p — length of blk_dims (0 allowed for pure LP) */
  int  lp_dim;            /* 0 for pure SDP */
  const int    *blk_dims; /* length num_cones */

  /* Objective C — sparse COO over (cone, row, col, val) */
  int           nnz_c;
  const int    *c_cone_ind;   /* length nnz_c, in [0, num_cones) */
  const int    *c_row_ind;    /* length nnz_c, in [0, blk_dims[cone]) */
  const int    *c_col_ind;    /* length nnz_c */
  const double *c_val;        /* length nnz_c */

  /* Constraints A_i — sparse COO over (constr, cone, row, col, val) */
  int           nnz_a;
  const int    *a_constr_ind; /* length nnz_a, in [0, num_constraints) */
  const int    *a_cone_ind;   /* length nnz_a, in [0, num_cones) */
  const int    *a_row_ind;    /* length nnz_a */
  const int    *a_col_ind;    /* length nnz_a */
  const double *a_val;        /* length nnz_a */

  /* LP objective (dense) — NULL iff lp_dim == 0 */
  const double *lp_obj;       /* length lp_dim */

  /* LP constraints — sparse COO over (constr, col, val), NULL iff nnz_lp==0 */
  int           nnz_lp;
  const int    *lp_constr_ind; /* length nnz_lp, in [0, num_constraints) */
  const int    *lp_col_ind;    /* length nnz_lp, in [0, lp_dim) */
  const double *lp_val;        /* length nnz_lp */

  /* RHS */
  const double *b;            /* length num_constraints */
} cardal_problem_data;

cardal_problem *cardal_build_problem(const cardal_problem_data *data,
                                     cardal_error *err_out);

/* Problem introspection. Return 0 (or negative) on invalid handle. */
int cardal_problem_num_constraints(const cardal_problem *p);
int cardal_problem_num_cones(const cardal_problem *p);
int cardal_problem_num_variables(const cardal_problem *p);
int cardal_problem_lp_dim(const cardal_problem *p);

/* Copy per-cone dimensions into `out_dims` (size >= cardal_problem_num_cones).
 * Returns CARDAL_OK on success. */
cardal_error cardal_problem_get_block_dims(const cardal_problem *p,
                                           int *out_dims);

void cardal_problem_free(cardal_problem *p);

/* -----------------------------------------------------------------------
 * Solve
 * ----------------------------------------------------------------------- */

/* Run the single-GPU solver. `params` may be NULL (defaults used). Returns
 * NULL on failure; free the returned result with cardal_result_free(). */
cardal_result *cardal_solve(cardal_problem *problem,
                            const cardal_params *params);

/* -----------------------------------------------------------------------
 * Result accessors (borrowed pointers valid until cardal_result_free)
 * ----------------------------------------------------------------------- */

cardal_status cardal_result_status(const cardal_result *r);

double cardal_result_primal_objective(const cardal_result *r);
double cardal_result_dual_objective(const cardal_result *r);
double cardal_result_objective_gap(const cardal_result *r);

double cardal_result_rel_primal_residual(const cardal_result *r);
double cardal_result_rel_dual_residual(const cardal_result *r);
double cardal_result_rel_objective_gap(const cardal_result *r);

double cardal_result_runtime_sec(const cardal_result *r);
int    cardal_result_outer_iters(const cardal_result *r);
int    cardal_result_inner_iters(const cardal_result *r);

int    cardal_result_num_cones(const cardal_result *r);
int    cardal_result_num_variables(const cardal_result *r);
int    cardal_result_num_constraints(const cardal_result *r);

/* Total BM rank across cones (sum of rank_list). */
int    cardal_result_total_rank(const cardal_result *r);

/* Low-rank primal factor V (flattened block-diagonal, column-major per
 * cone). Length written to *out_length; returns NULL if unavailable. */
const double *cardal_result_primal_factor(const cardal_result *r,
                                          int *out_length);

/* Dual solution y (length = num_constraints). */
const double *cardal_result_dual(const cardal_result *r,
                                 int *out_length);

/* Per-cone BM rank list (length = num_cones). */
const int    *cardal_result_rank_list(const cardal_result *r,
                                      int *out_length);

void cardal_result_free(cardal_result *r);

#ifdef __cplusplus
}
#endif

#endif /* CARDAL_H */
