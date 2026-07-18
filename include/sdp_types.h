/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int row_dims;
  int rank_dims;
  int cone_dims;
  bool decided;
} grid_size_t;

typedef enum {
  SHUFFLE_NONE = 0,
  SHUFFLE_UNIFORM = 1,
  SHUFFLE_BLOCK = 2,
  SHUFFLE_COL_LOCALITY = 3,
} shuffle_type_t;

typedef struct {
  int *constr_ind;
  int *cone_ind;
  int *row_ind;
  int *col_ind;
  double *val;
} psd_cone_constraint_t;

typedef struct {
  int *cone_ind;
  int *row_ind;
  int *col_ind;
  double *val;
} psd_cone_objective_t;

typedef struct {
  int *row_ind;
  int *col_ind;
  double *val;
} lp_constraint_t;

typedef struct {
  // --- Data Pointers ---
  psd_cone_constraint_t *psd_cone_constraints;
  psd_cone_objective_t *psd_cone_objective;

  // LP/Real cone parts
  lp_constraint_t *lp_constraints;
  double *lp_objective;

  double *right_hand_side; // Vector b
  int *blk_dims;           // Block dimensions
  int lp_dim;

  // --- Size Components (Added) ---
  int m;       // Total number of constraints
  int n_cones; // Number of PSD blocks

  int nnz_psd_constr; // Length of arrays in psd_cone_constraints
  int nnz_psd_obj;    // Length of arrays in psd_cone_objective

  int nnz_lp_constr; // Length of arrays in lp_constraints (if used)
  int nnz_lp_obj;    // Length of arrays in lp_objective (if used)

} basic_sdp_t;

typedef struct {
  int *row_ptr;
  int *col_ind;
  double *val;
  int num_rows;
  int num_cols;
  int num_nonzeros;
} sparse_csr_matrix_t;

typedef struct {
  long long *pos;
  double *val;
  int len;
} sparse_vector_t;

typedef struct {
  int num_constraints;
  int n_blks;

  long long total_n_orig;
  int n_active_vars;
  long long *col_mapping;

  sparse_csr_matrix_t *constraint_matrix;
  sparse_csr_matrix_t *constraint_matrix_t;
  sparse_vector_t *objective_vector_sparse;

  int *blk_dims;
  long long *blk_ptr;

  int lp_dim;
  int lp_start_idx;
  double *lp_objective_vector;

  double *right_hand_side;

} compressed_sdp_problem_t;

typedef struct {
  double eps_optimal_relative;
  double eps_primal_relative;
  double eps_dual_relative;
  double time_sec_limit;
  int iteration_limit;
} termination_criteria_t;

typedef enum {
  AUGMENTATION_MODE_RANDOM = 0,
  AUGMENTATION_MODE_QP = 1,
  AUGMENTATION_MODE_CLOSED_FORM = 2,
  AUGMENTATION_MODE_SDP = 3,
} augmentation_mode_t;

typedef struct {
  int initial_rank;
  int max_rank;
  augmentation_mode_t augmentation_mode;
  int lbfgs_history_size;
  double penalty_factor;
  double initial_penalty_coef;
  double max_penalty_coef;
  double inner_iterations_limit;
  termination_criteria_t termination_criteria;
  int verbose; // 0=silent, 1=banner+summary, 2=+subtitles+iter table, 3=debug
  grid_size_t grid_size;
  const char *instance_label;
  const char *summary_file_path;

  int l_inf_ruiz_iterations;
  bool has_pock_chambolle_alpha;
  double pock_chambolle_alpha;
  bool bound_objective_rescaling;

  int psd_scale_mode;
  shuffle_type_t shuffle_mode;
} cardal_parameters_t;

#define PSD_SCALE_MODE_PER_ELEMENT 0
#define PSD_SCALE_MODE_PER_CONE 1

typedef enum {
  TERMINATION_REASON_UNSPECIFIED,
  TERMINATION_REASON_OPTIMAL,
  TERMINATION_REASON_TIME_LIMIT,
  TERMINATION_REASON_ITERATION_LIMIT,
  TERMINATION_REASON_USER_INTERRUPT,
} termination_reason_t;

typedef struct {
  int num_variables;
  int num_constraints;
  int num_nonzeros;
  int rank;

  double *low_rank_primal_solution;
  long long low_rank_solution_length;
  int n_cones;
  int *rank_list;
  double *dual_solution;

  int total_count;
  int total_inner_count;
  double rescaling_time_sec;
  double cumulative_time_sec;

  double absolute_primal_residual;
  double relative_primal_residual;
  double absolute_dual_residual;
  double relative_dual_residual;
  double primal_objective_value;
  double dual_objective_value;
  double objective_gap;
  double relative_objective_gap;
  termination_reason_t termination_reason;
} sdp_result_t;
#ifdef __cplusplus
}
#endif
