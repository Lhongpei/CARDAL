/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

/* CARDAL public C ABI — implementation.
 *
 * Thin wrapper around the internal solver:
 *   sdp_problem_parse()  → basic_sdp_t
 *   convert_to_compressed() → compressed_sdp_problem_t
 *   optimize() → sdp_result_t
 *
 * We hide all three internal types behind opaque cardal_problem / cardal_result
 * so the ABI is stable across internal refactors. */

#include "cardal.h"

#include "sdp_types.h"     /* basic_sdp_t, compressed_sdp_problem_t, cardal_parameters_t, sdp_result_t */
#include "parser.h"        /* sdp_problem_parse, free_basic_sdp */
#include "solver.h"        /* optimize */
#include "utils.h"         /* convert_to_compressed, free_compressed_sdp, safe_malloc, set_default_parameters */

#include <signal.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Cooperative cancellation flag.
 *
 * Set from a signal handler (typically the Python binding's SIGINT trap);
 * polled from the ALM outer loop (`check_termination` in outer_loop.cuh).
 * When observed non-zero the outer loop returns TERMINATION_REASON_USER_INTERRUPT
 * cleanly, so the C-side state is freed correctly and the caller can decide
 * what to do (Python binding re-raises KeyboardInterrupt).
 *
 * The flag lives in a plain C TU so both a C signal handler and a CUDA
 * .cu poll site can see it.
 * ----------------------------------------------------------------------- */
volatile sig_atomic_t g_cardal_cancel_request = 0;

void cardal_request_cancel(void) { g_cardal_cancel_request = 1; }
void cardal_clear_cancel(void)   { g_cardal_cancel_request = 0; }
int  cardal_cancel_requested(void) { return (int)g_cardal_cancel_request; }

/* -----------------------------------------------------------------------
 * Opaque handle definitions
 * ----------------------------------------------------------------------- */

struct cardal_problem {
  basic_sdp_t              *basic;      /* owned; raw parsed form */
  compressed_sdp_problem_t *compressed; /* owned; solver-ready form */
};

struct cardal_result {
  sdp_result_t *inner;                  /* owned */
};

/* -----------------------------------------------------------------------
 * Defaults
 * ----------------------------------------------------------------------- */

void cardal_default_params(cardal_params *p) {
  if (p == NULL)
    return;
  cardal_parameters_t d;
  set_default_parameters(&d);
  p->eps_primal_relative    = d.termination_criteria.eps_primal_relative;
  p->eps_dual_relative      = d.termination_criteria.eps_dual_relative;
  p->eps_optimal_relative   = d.termination_criteria.eps_optimal_relative;
  p->time_sec_limit         = d.termination_criteria.time_sec_limit;
  p->iteration_limit        = d.termination_criteria.iteration_limit;
  p->initial_rank           = d.initial_rank;
  p->max_rank               = d.max_rank;
  p->augmentation_mode      = (int)d.augmentation_mode;
  p->lbfgs_history_size     = d.lbfgs_history_size;
  p->penalty_factor         = d.penalty_factor;
  p->initial_penalty_coef   = d.initial_penalty_coef;
  p->max_penalty_coef       = d.max_penalty_coef;
  p->inner_iterations_limit = (long)d.inner_iterations_limit;
  p->verbose                = d.verbose;
}

/* Copy cardal_params (public POD) → cardal_parameters_t (internal). */
static void cardal_params_to_internal(const cardal_params *src,
                                      cardal_parameters_t   *dst) {
  /* Start with baseline defaults so any field not covered by the public
     ABI (rescaling flags, grid_size, etc.) gets a sensible value. */
  set_default_parameters(dst);
  if (src == NULL)
    return;
  dst->termination_criteria.eps_optimal_relative  = src->eps_optimal_relative;
  dst->termination_criteria.eps_primal_relative   = src->eps_primal_relative;
  dst->termination_criteria.eps_dual_relative     = src->eps_dual_relative;
  dst->termination_criteria.time_sec_limit        = src->time_sec_limit;
  dst->termination_criteria.iteration_limit       = src->iteration_limit;
  dst->initial_rank           = src->initial_rank;
  dst->max_rank               = src->max_rank;
  if (src->augmentation_mode >= CARDAL_AUGMENTATION_RANDOM &&
      src->augmentation_mode <= CARDAL_AUGMENTATION_SDP)
    dst->augmentation_mode = (augmentation_mode_t)src->augmentation_mode;
  dst->lbfgs_history_size     = src->lbfgs_history_size;
  dst->penalty_factor         = src->penalty_factor;
  dst->initial_penalty_coef   = src->initial_penalty_coef;
  dst->max_penalty_coef       = src->max_penalty_coef;
  dst->inner_iterations_limit = (double)src->inner_iterations_limit;
  dst->verbose                = src->verbose;
}

/* -----------------------------------------------------------------------
 * Problem
 * ----------------------------------------------------------------------- */

cardal_problem *cardal_read_sdpa(const char *path, cardal_error *err_out) {
#define SET_ERR(code) do { if (err_out) *err_out = (code); } while (0)
  if (path == NULL) {
    SET_ERR(CARDAL_E_NULL_ARG);
    return NULL;
  }
  basic_sdp_t *basic = sdp_problem_parse(path);
  if (basic == NULL) {
    SET_ERR(CARDAL_E_PARSE);
    return NULL;
  }
  compressed_sdp_problem_t *comp = convert_to_compressed(basic);
  if (comp == NULL) {
    free_basic_sdp(basic);
    SET_ERR(CARDAL_E_INTERNAL);
    return NULL;
  }

  cardal_problem *out = (cardal_problem *)calloc(1, sizeof(cardal_problem));
  if (out == NULL) {
    free_compressed_sdp(comp);
    free_basic_sdp(basic);
    SET_ERR(CARDAL_E_INTERNAL);
    return NULL;
  }
  out->basic      = basic;
  out->compressed = comp;
  SET_ERR(CARDAL_OK);
  return out;
#undef SET_ERR
}

/* Copy `n` ints from src to a fresh safe_malloc'd buffer; returns NULL if n==0. */
static int *dup_ints(const int *src, int n) {
  if (n <= 0 || src == NULL) return NULL;
  int *dst = (int *)safe_malloc((size_t)n * sizeof(int));
  memcpy(dst, src, (size_t)n * sizeof(int));
  return dst;
}

static double *dup_doubles(const double *src, int n) {
  if (n <= 0 || src == NULL) return NULL;
  double *dst = (double *)safe_malloc((size_t)n * sizeof(double));
  memcpy(dst, src, (size_t)n * sizeof(double));
  return dst;
}

cardal_problem *cardal_build_problem(const cardal_problem_data *data,
                                     cardal_error *err_out) {
#define SET_ERR(code) do { if (err_out) *err_out = (code); } while (0)
  if (data == NULL) {
    SET_ERR(CARDAL_E_NULL_ARG);
    return NULL;
  }
  /* Basic shape validation. Reject obviously malformed input rather than
   * segfault deep inside convert_to_compressed. */
  if (data->num_constraints < 0 || data->num_cones < 0 || data->lp_dim < 0) {
    SET_ERR(CARDAL_E_NULL_ARG);
    return NULL;
  }
  if (data->num_cones == 0 && data->lp_dim == 0) {
    SET_ERR(CARDAL_E_NULL_ARG);
    return NULL;
  }
  if (data->num_constraints > 0 && data->b == NULL) {
    SET_ERR(CARDAL_E_NULL_ARG);
    return NULL;
  }
  if (data->num_cones > 0 && data->blk_dims == NULL) {
    SET_ERR(CARDAL_E_NULL_ARG);
    return NULL;
  }
  if (data->lp_dim > 0 && data->lp_obj == NULL) {
    SET_ERR(CARDAL_E_NULL_ARG);
    return NULL;
  }

  basic_sdp_t *basic = (basic_sdp_t *)safe_malloc(sizeof(basic_sdp_t));
  memset(basic, 0, sizeof(*basic));
  basic->m       = data->num_constraints;
  basic->n_cones = data->num_cones;
  basic->lp_dim  = data->lp_dim;
  basic->blk_dims        = dup_ints(data->blk_dims, data->num_cones);
  basic->right_hand_side = dup_doubles(data->b,     data->num_constraints);

  /* PSD objective (C) */
  basic->nnz_psd_obj = data->nnz_c;
  if (data->nnz_c > 0) {
    basic->psd_cone_objective =
        (psd_cone_objective_t *)safe_malloc(sizeof(psd_cone_objective_t));
    basic->psd_cone_objective->cone_ind = dup_ints(data->c_cone_ind, data->nnz_c);
    basic->psd_cone_objective->row_ind  = dup_ints(data->c_row_ind,  data->nnz_c);
    basic->psd_cone_objective->col_ind  = dup_ints(data->c_col_ind,  data->nnz_c);
    basic->psd_cone_objective->val      = dup_doubles(data->c_val,   data->nnz_c);
  }

  /* PSD constraints (A_i) */
  basic->nnz_psd_constr = data->nnz_a;
  if (data->nnz_a > 0) {
    basic->psd_cone_constraints =
        (psd_cone_constraint_t *)safe_malloc(sizeof(psd_cone_constraint_t));
    basic->psd_cone_constraints->constr_ind = dup_ints(data->a_constr_ind, data->nnz_a);
    basic->psd_cone_constraints->cone_ind   = dup_ints(data->a_cone_ind,   data->nnz_a);
    basic->psd_cone_constraints->row_ind    = dup_ints(data->a_row_ind,    data->nnz_a);
    basic->psd_cone_constraints->col_ind    = dup_ints(data->a_col_ind,    data->nnz_a);
    basic->psd_cone_constraints->val        = dup_doubles(data->a_val,     data->nnz_a);
  }

  /* LP objective (dense) */
  if (data->lp_dim > 0) {
    basic->lp_objective = dup_doubles(data->lp_obj, data->lp_dim);
    basic->nnz_lp_obj   = data->lp_dim;
  }

  /* LP constraints (A_lp) */
  basic->nnz_lp_constr = data->nnz_lp;
  if (data->nnz_lp > 0) {
    basic->lp_constraints =
        (lp_constraint_t *)safe_malloc(sizeof(lp_constraint_t));
    basic->lp_constraints->row_ind = dup_ints(data->lp_constr_ind, data->nnz_lp);
    basic->lp_constraints->col_ind = dup_ints(data->lp_col_ind,    data->nnz_lp);
    basic->lp_constraints->val     = dup_doubles(data->lp_val,     data->nnz_lp);
  }

  compressed_sdp_problem_t *comp = convert_to_compressed(basic);
  if (comp == NULL) {
    free_basic_sdp(basic);
    SET_ERR(CARDAL_E_INTERNAL);
    return NULL;
  }

  cardal_problem *out = (cardal_problem *)calloc(1, sizeof(cardal_problem));
  if (out == NULL) {
    free_compressed_sdp(comp);
    free_basic_sdp(basic);
    SET_ERR(CARDAL_E_INTERNAL);
    return NULL;
  }
  out->basic      = basic;
  out->compressed = comp;
  SET_ERR(CARDAL_OK);
  return out;
#undef SET_ERR
}

int cardal_problem_num_constraints(const cardal_problem *p) {
  return (p && p->compressed) ? p->compressed->num_constraints : 0;
}

int cardal_problem_num_cones(const cardal_problem *p) {
  return (p && p->compressed) ? p->compressed->n_blks : 0;
}

int cardal_problem_num_variables(const cardal_problem *p) {
  return (p && p->compressed) ? p->compressed->n_active_vars : 0;
}

int cardal_problem_lp_dim(const cardal_problem *p) {
  return (p && p->compressed) ? p->compressed->lp_dim : 0;
}

cardal_error cardal_problem_get_block_dims(const cardal_problem *p,
                                           int *out_dims) {
  if (p == NULL || p->compressed == NULL || out_dims == NULL)
    return CARDAL_E_NULL_ARG;
  const int n = p->compressed->n_blks;
  for (int i = 0; i < n; i++)
    out_dims[i] = p->compressed->blk_dims[i];
  return CARDAL_OK;
}

void cardal_problem_free(cardal_problem *p) {
  if (p == NULL)
    return;
  if (p->compressed)
    free_compressed_sdp(p->compressed);
  if (p->basic)
    free_basic_sdp(p->basic);
  free(p);
}

/* -----------------------------------------------------------------------
 * Solve
 * ----------------------------------------------------------------------- */

cardal_result *cardal_solve(cardal_problem *problem,
                            const cardal_params *params) {
  if (problem == NULL || problem->compressed == NULL)
    return NULL;

  cardal_parameters_t p;
  cardal_params_to_internal(params, &p);
  p.instance_label    = NULL;
  p.summary_file_path = NULL;

  sdp_result_t *r = optimize(problem->compressed, &p);
  if (r == NULL)
    return NULL;

  cardal_result *out = (cardal_result *)calloc(1, sizeof(cardal_result));
  if (out == NULL) {
    free_sdp_result(r);
    return NULL;
  }
  out->inner = r;
  return out;
}

/* -----------------------------------------------------------------------
 * Result accessors
 * ----------------------------------------------------------------------- */

/* Map internal termination reason to public status. */
static cardal_status map_status(termination_reason_t t) {
  switch (t) {
  case TERMINATION_REASON_OPTIMAL:         return CARDAL_STATUS_OPTIMAL;
  case TERMINATION_REASON_TIME_LIMIT:      return CARDAL_STATUS_TIME_LIMIT;
  case TERMINATION_REASON_ITERATION_LIMIT: return CARDAL_STATUS_ITERATION_LIMIT;
  case TERMINATION_REASON_USER_INTERRUPT:  return CARDAL_STATUS_USER_INTERRUPT;
  case TERMINATION_REASON_UNSPECIFIED:
  default:                                 return CARDAL_STATUS_UNSPECIFIED;
  }
}

cardal_status cardal_result_status(const cardal_result *r) {
  return (r && r->inner) ? map_status(r->inner->termination_reason)
                         : CARDAL_STATUS_UNSPECIFIED;
}

#define RETURN_FIELD(r, field, fallback) \
  (((r) && (r)->inner) ? (r)->inner->field : (fallback))

double cardal_result_primal_objective(const cardal_result *r) { return RETURN_FIELD(r, primal_objective_value,     0.0); }
double cardal_result_dual_objective  (const cardal_result *r) { return RETURN_FIELD(r, dual_objective_value,       0.0); }
double cardal_result_objective_gap   (const cardal_result *r) { return RETURN_FIELD(r, objective_gap,              0.0); }
double cardal_result_rel_primal_residual(const cardal_result *r) { return RETURN_FIELD(r, relative_primal_residual, 0.0); }
double cardal_result_rel_dual_residual  (const cardal_result *r) { return RETURN_FIELD(r, relative_dual_residual,   0.0); }
double cardal_result_rel_objective_gap  (const cardal_result *r) { return RETURN_FIELD(r, relative_objective_gap,   0.0); }
double cardal_result_runtime_sec        (const cardal_result *r) { return RETURN_FIELD(r, cumulative_time_sec,      0.0); }
int    cardal_result_outer_iters        (const cardal_result *r) { return RETURN_FIELD(r, total_count,              0);   }
int    cardal_result_inner_iters        (const cardal_result *r) { return RETURN_FIELD(r, total_inner_count,        0);   }
int    cardal_result_num_cones          (const cardal_result *r) { return RETURN_FIELD(r, n_cones,                  0);   }
int    cardal_result_num_variables      (const cardal_result *r) { return RETURN_FIELD(r, num_variables,            0);   }
int    cardal_result_num_constraints    (const cardal_result *r) { return RETURN_FIELD(r, num_constraints,          0);   }
int    cardal_result_total_rank         (const cardal_result *r) { return RETURN_FIELD(r, rank,                     0);   }

#undef RETURN_FIELD

const double *cardal_result_primal_factor(const cardal_result *r,
                                          int *out_length) {
  if (r == NULL || r->inner == NULL) {
    if (out_length) *out_length = 0;
    return NULL;
  }
  if (out_length) *out_length = (int)r->inner->low_rank_solution_length;
  return r->inner->low_rank_primal_solution;
}

const double *cardal_result_dual(const cardal_result *r, int *out_length) {
  if (r == NULL || r->inner == NULL) {
    if (out_length) *out_length = 0;
    return NULL;
  }
  if (out_length) *out_length = r->inner->num_constraints;
  return r->inner->dual_solution;
}

const int *cardal_result_rank_list(const cardal_result *r, int *out_length) {
  if (r == NULL || r->inner == NULL) {
    if (out_length) *out_length = 0;
    return NULL;
  }
  if (out_length) *out_length = r->inner->n_cones;
  return r->inner->rank_list;
}

void cardal_result_free(cardal_result *r) {
  if (r == NULL)
    return;
  if (r->inner)
    free_sdp_result(r->inner);
  free(r);
}
