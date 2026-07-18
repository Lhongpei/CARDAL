/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once
#include "sdp_types.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#ifdef USE_MPI
#include <mpi.h>
#include <nccl.h>
#endif
#ifdef __cplusplus
extern "C" {
#endif

#ifndef CARDAL_CONE_STREAM_POOL_SIZE
#define CARDAL_CONE_STREAM_POOL_SIZE 32
#endif

#ifndef CARDAL_SMALL_CONE_DIM_THRESHOLD
#define CARDAL_SMALL_CONE_DIM_THRESHOLD 128
#endif

#ifndef CARDAL_MIN_BATCH_SIZE
#define CARDAL_MIN_BATCH_SIZE 16
#endif

typedef enum {
  CONE_BATCH_KIND_PERCONE = 0,
  CONE_BATCH_KIND_CUSTOM = 1,
} cone_batch_kind_t;


typedef struct {
  // Per-cone descriptors (length parent->n_cones).
  int max_rank;               // cached max(rank_c)
  int *blk_idx_h;             // [n_cones] original blk index (0..state->n_blks)
  int *d_blk_idx;             // [n_cones] device mirror; batched
                              // segmented-dot uses this to scatter into
                              // the correct d_p2_per_cone slot
  int *d_ranks;               // [n_cones]
  long long *d_R_offsets;     // [n_cones] offsets into low_rank_solution
  long long *d_G_offsets;     // [n_cones] offsets into low_rank_gradient
  long long *d_D_offsets;     // [n_cones] offsets into low_rank_direction

  // Flat sparsity for matSpA (constraint pattern) across cones in batch.
  int total_nnz_A;
  int *d_entry_cone_A;        // [total_nnz_A] cone idx in [0, n_cones)
  int *d_entry_row_A;
  int *d_entry_col_A;
  int *d_entry_compat_A;      // global scatter target
  int *d_entry_A_to_flatS;    // matching index in d_flat_spS_val

  // Flat sparsity for matSpS (objective union pattern)
  int total_nnz_S;
  int *d_entry_cone_S;
  int *d_entry_row_S;
  int *d_entry_col_S;
  int *d_cone_S_offsets;      // [n_cones+1]

  double *d_flat_objval_S;
  double *d_flat_spS_val;

  // Per-cone matSpS row_ptr flattened (within-cone offsets concatenated).
  int *d_S_row_ptr_flat;
  int *d_S_row_ptr_offsets;   // [n_cones+1]
} cone_batch_data_t;

typedef enum {
  LBFGS_GRAD_CONVERGED = 0,
  LBFGS_TAU_STALL = 1,
  LBFGS_MAX_ITERS = 2,
} lbfgs_exit_reason_t;

// LBFGS device scalar scratch slot indices.
enum {
  LBFGS_SCR_NEG_ONE = 0,
  LBFGS_SCR_SY = 1,
  LBFGS_SCR_YY = 2,
  LBFGS_SCR_SY_LATEST = 3,
  LBFGS_SCR_GAMMA = 4,
  LBFGS_SCR_SQ = 5,
  LBFGS_SCR_NEG_ALPHA = 6,
  LBFGS_SCR_YZ = 7,
  LBFGS_SCR_BETA = 8,
  LBFGS_SCR_FACTOR = 9,
  LBFGS_SCR_DIR_GRAD_DOT = 10,
  LBFGS_SCR_DIR_NORM = 11,
  LBFGS_SCR_DIR_INV_NORM = 12,
  LBFGS_SCR_GRAD_NORM = 13,
  LBFGS_SCR_SIZE = 16
};

typedef struct {
#ifdef USE_MPI
  MPI_Comm comm_global;
  MPI_Comm comm_row;
  MPI_Comm comm_rank;
  MPI_Comm comm_cone;
  ncclComm_t nccl_row;
  ncclComm_t nccl_rank;
  ncclComm_t nccl_cone;
#endif
  int rank_global;
  int coords[3];
  int dims[3];
} grid_context_t;

typedef struct {
    compressed_sdp_problem_t *scaled_problem;
    double *constraint_rescaling;
    double *psd_cone_rescaling;       // [Σ blk_dims], per-X-row factor d^(b)[i]
    double *lp_variable_rescaling;
    double  objective_vector_rescaling;
    double  right_hand_side_rescaling;
    double  unscaled_right_hand_side_norm;
    double  unscaled_objective_vector_norm;
    double  rescaling_time_sec;
} rescale_info_t;

typedef struct {
  int dim;
  int n_cones;
  cone_batch_kind_t kind;

  double *solution;
  double *gradient;
  double *direction;
  sparse_csr_matrix_t *constraint_sparse_pattern;
  sparse_csr_matrix_t *objective_union_constraint_sparse_pattern;
  double *objective_val;
  int rank;

  int *compat_mapping;
  int *constraint_to_union_mapping;

  cusparseDnMatDescr_t matR;
  cusparseDnMatDescr_t matGrad;
  cusparseDnMatDescr_t matD;
  cusparseSpMatDescr_t matSpA;
  cusparseSpMatDescr_t matSpC;
  cusparseSpMatDescr_t matSpS;

  size_t sddmm_buffer_size_A;
  void *sddmm_buffer_A;
  size_t sddmm_buffer_size_C;
  void *sddmm_buffer_C;
  size_t spmm_buffer_size_S;
  void *spmm_buffer_S;

  cudaStream_t stream;

  // Per-row PSD rescale d^(b)[i] for i in [0, dim).
  double *psd_cone_rescaling;
  double *vec_psd_cone_rescaling; // [d^(b)[i] * d^(b)[j]]

  // Specialized for Batched Cone
  cone_batch_data_t *bdata;
} block_low_rank_state_t;

typedef struct {
  // Problem Information
  int num_constraints;
  int n_blks;
  int lp_dim;
  int n_active_vars;
  int *blk_dims;
  long long *blk_ptr;
  long long *col_mapping;
  int *rank_list;
  int *theoretical_max_rank_per_blk;
  int total_rank;

  sparse_csr_matrix_t *constraint_matrix;
  sparse_csr_matrix_t *constraint_matrix_t;
  sparse_vector_t *objective_vector_sparse;
  double *right_hand_side;

  // Low Rank Struct
  block_low_rank_state_t **block_low_rank_state;
  double *low_rank_solution;
  int length_low_rank_solution;
  double *low_rank_gradient;
  double *low_rank_direction;

  // LP Struct
  int lp_start_active_idx;
  int lp_solution_offset;
  double *lp_objective_vector;
  double *lp_slack_buffer;
  double *lp_min_slack_buf;

  // Complete Solution and Calculation Buffer
  double *primal_solution;
  double *primal_direct_solution_cross;
  double *primal_direct_double;
  double *dual_solution;
  double *primal_product;
  double *dual_product;

  cusparseHandle_t sparse_handle;
  cublasHandle_t blas_handle;

  cusparseSpMatDescr_t matA;
  cusparseSpMatDescr_t matAt;
  cusparseSpVecDescr_t vecObj;
  cusparseDnVecDescr_t vec_primal_sol;
  cusparseDnVecDescr_t vec_dual_sol;
  cusparseDnVecDescr_t vec_q1;
  cusparseDnVecDescr_t vec_primal_prod;
  cusparseDnVecDescr_t vec_dual_prod;
  size_t primal_spmv_buffer_size;
  size_t dual_spmv_buffer_size;
  void *primal_spmv_buffer;
  void *dual_spmv_buffer;

  double *q0; // Primal Residual (Ax - b)
  double *q1; // Buffer for (-lambda - rho*q0)
  double *q2; // Buffer for second order residual

  double objective_vector_norm;
  double right_hand_side_norm;
  double objective_vector_linf_norm;
  double right_hand_side_linf_norm;

  // Rescale Info and Buffer
  double *constraint_rescaling;
  double *lp_variable_rescaling;
  double  objective_vector_rescaling;
  double  right_hand_side_rescaling;
  double  unscaled_right_hand_side_norm;
  double  unscaled_objective_vector_norm;
  double *q0_unscaled_buf;

  // Residual And Parameter
  double inner_iterations_limit;
  int lbfgs_history_size;
  double penalty_coef;
  double penalty_factor;
  augmentation_mode_t augmentation_mode;
  double step_size;
  double primal_objective;
  double dual_objective;
  double objective_gap;
  double relative_objective_gap;
  double relative_primal_residual;
  double absolute_primal_residual;
  double absolute_dual_residual;
  double relative_dual_residual;

  int dual_residual_evaluated;

  // 0=LBFGS_GRAD_CONVERGED, 1=LBFGS_TAU_STALL, 2=LBFGS_MAX_ITERS.
  int lbfgs_exit_reason;

  termination_reason_t termination_reason;

  double last_outer_primal_norm;
  double last_outer_obj_gap;
  int outer_stall_count;
  int outer_iters_since_augment_check;
  int lbfgs_maxiter_streak;

  double inner_eta;
  double prev_outer_primal_for_eta;

  double lancelot_eta;

  double cumulative_time_sec;
  int num_outer_iteration;
  int num_inner_iteration;

  int verbose;
  grid_context_t *grid_context;

  cudaStream_t cone_stream_pool[CARDAL_CONE_STREAM_POOL_SIZE];
  int cone_stream_pool_size;

  double *d_p2_per_cone;
  double *h_p2_per_cone;

  double *d_step_scalars;

  double *d_lbfgs_S;
  double *d_lbfgs_Y;
  double *d_lbfgs_R_old;
  double *d_lbfgs_Grad_old;
  double *d_lbfgs_s;
  double *d_lbfgs_y;
  double *d_lbfgs_q;
  double *d_lbfgs_z;
  double *d_lbfgs_rho;
  double *d_lbfgs_alpha;
  double *d_lbfgs_scratch;
  int lbfgs_buf_capacity_m;
  int lbfgs_buf_capacity_n;

  int n_batches;
  int *batch_leaders;

  int gate_fail_streak;
  int curvature_last_check_iter;

  int gap_stall_count;
  int consecutive_gate_pass;
  int force_augment_this_iter;
} cardal_sdp_solver_state_t;

#ifdef __cplusplus
}
#endif
