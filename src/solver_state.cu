/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "internal_types.h"
#include "sdp_op.h"
#include "solver_state.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
typedef struct {
  int r, c;
  int orig_idx;
  double val;
} temp_coo_t;

int cmp_temp_coo(const void *a, const void *b) {
  temp_coo_t *ta = (temp_coo_t *)a;
  temp_coo_t *tb = (temp_coo_t *)b;
  if (ta->r != tb->r)
    return ta->r - tb->r;
  return ta->c - tb->c;
}

void randomize_device_array(double *d_array, size_t length, unsigned seed) {
  double *h_array = (double *)safe_malloc(length * sizeof(double));
  unsigned state = seed ? seed : 1u;
  for (size_t i = 0; i < length; i++) {
    state = state * 1103515245u + 12345u;
    double u = (double)((state >> 16) & 0x7FFFu) / 32768.0;
    h_array[i] = u - 0.5;
  }
  cudaError_t err = cudaMemcpy(d_array, h_array, length * sizeof(double),
                               cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("CUDA Memcpy failed in randomize_device_array: %s\n",
           cudaGetErrorString(err));
  }
  free(h_array);
}

static unsigned get_rank_aware_seed(unsigned base) {
#ifdef USE_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return base;
  }
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  return base + (unsigned)world_rank * 2654435761u;
#else
  return base;
#endif
}

cardal_sdp_solver_state_t *
initialize_solver_state(const compressed_sdp_problem_t *sdp_problem,
                        const cardal_parameters_t *params,
                        int *theoretical_max_rank_per_blk,
                        const rescale_info_t *rescale_info) {
  cardal_sdp_solver_state_t *state = (cardal_sdp_solver_state_t *)safe_calloc(
      1, sizeof(cardal_sdp_solver_state_t));

  int n_cons = sdp_problem->num_constraints;
  int n_cones = sdp_problem->n_blks;
  int n_act_vars = sdp_problem->n_active_vars;
  state->theoretical_max_rank_per_blk = theoretical_max_rank_per_blk;
  state->verbose = params->verbose;
  state->termination_reason = TERMINATION_REASON_UNSPECIFIED;

  double norm_C_sq = 0.0;
  double norm_C_inf = 0.0;

  if (sdp_problem->objective_vector_sparse != NULL) {
    for (int i = 0; i < sdp_problem->objective_vector_sparse->len; i++) {
      double val = sdp_problem->objective_vector_sparse->val[i];
      norm_C_sq += fabs(val);
      norm_C_inf = fmax(fabs(val), norm_C_inf);
    }
  }

  if (sdp_problem->lp_dim > 0 && sdp_problem->lp_objective_vector != NULL) {
    for (int i = 0; i < sdp_problem->lp_dim; i++) {
      double val = sdp_problem->lp_objective_vector[i];
      norm_C_sq += fabs(val);
      norm_C_inf = fmax(fabs(val), norm_C_inf);
    }
  }

  state->objective_vector_norm = norm_C_sq;
  state->objective_vector_linf_norm = norm_C_inf;

  double norm_b_sq = 0.0;
  double norm_b_inf = 0.0;
  for (int i = 0; i < sdp_problem->num_constraints; i++) {
    double val = sdp_problem->right_hand_side[i];
    norm_b_sq += fabs(val);
    norm_b_inf = fmax(fabs(val), norm_b_inf);
  }
  state->right_hand_side_norm = norm_b_sq;
  state->right_hand_side_linf_norm = norm_b_inf;

  state->num_constraints = n_cons;
  state->n_blks = n_cones;

  state->lp_dim = sdp_problem->lp_dim;
  state->lp_start_active_idx = sdp_problem->lp_start_idx;

  state->n_active_vars = n_act_vars;
  state->blk_dims = (int *)safe_malloc(n_cones * sizeof(int));
  state->blk_ptr = (long long *)safe_malloc((n_cones + 1) * sizeof(long long));
  state->col_mapping = (long long *)safe_malloc(n_act_vars * sizeof(long long));
  safe_memcpy(state->blk_dims, sdp_problem->blk_dims, n_cones * sizeof(int));
  safe_memcpy(state->blk_ptr, sdp_problem->blk_ptr,
              (n_cones + 1) * sizeof(long long));
  safe_memcpy(state->col_mapping, sdp_problem->col_mapping,
              n_act_vars * sizeof(long long));
  state->rank_list = (int *)safe_malloc(n_cones * sizeof(int));

  int tot_rank = 0;
  for (int i = 0; i < n_cones; i++) {
    int theoretical_max_r = state->theoretical_max_rank_per_blk[i];
    int max_allowed_rank = (sdp_problem->blk_dims[i] < theoretical_max_r)
                               ? sdp_problem->blk_dims[i]
                               : theoretical_max_r;
    state->rank_list[i] = (params->initial_rank < max_allowed_rank)
                              ? params->initial_rank
                              : max_allowed_rank;
    tot_rank += state->rank_list[i];
  }
  state->total_rank = tot_rank;
  state->penalty_coef = params->initial_penalty_coef;
  state->inner_eta = 0.1;
  state->prev_outer_primal_for_eta = 0.0;
  state->lancelot_eta = 1.0;
  state->gate_fail_streak = 0;
  state->sentinel_last_fire_iter = -10000;
  state->sentinel_last_fire_rank = -1;
  state->gap_stall_count = 0;
  state->consecutive_gate_pass = 0;
  state->force_augment_this_iter = 0;
  state->inner_iterations_limit = params->inner_iterations_limit;
  state->lbfgs_history_size = (params->lbfgs_history_size > 0)
                                  ? params->lbfgs_history_size
                                  : 5;
  state->penalty_factor = params->penalty_factor;

  state->constraint_matrix =
      (sparse_csr_matrix_t *)safe_malloc(sizeof(sparse_csr_matrix_t));
  state->constraint_matrix_t =
      (sparse_csr_matrix_t *)safe_malloc(sizeof(sparse_csr_matrix_t));

  state->constraint_matrix->num_rows = sdp_problem->constraint_matrix->num_rows;
  state->constraint_matrix->num_cols = sdp_problem->constraint_matrix->num_cols;
  state->constraint_matrix->num_nonzeros =
      sdp_problem->constraint_matrix->num_nonzeros;
  state->constraint_matrix_t->num_rows = state->n_active_vars;
  state->constraint_matrix_t->num_cols = state->num_constraints;
  state->constraint_matrix_t->num_nonzeros =
      sdp_problem->constraint_matrix->num_nonzeros;

  ALLOC_AND_COPY_CSR(state->constraint_matrix, sdp_problem->constraint_matrix,
                     sdp_problem->constraint_matrix->num_rows,
                     sdp_problem->constraint_matrix->num_nonzeros);

  CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ptr,
                        (n_act_vars + 1) * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(&state->constraint_matrix_t->col_ind,
                 sdp_problem->constraint_matrix->num_nonzeros * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->val,
                        sdp_problem->constraint_matrix->num_nonzeros *
                            sizeof(double)));

  CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle));
  CUBLAS_CHECK(cublasCreate(&state->blas_handle));
  CUBLAS_CHECK(
      cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));
  if (state->constraint_matrix->num_nonzeros > 0) {
    size_t buffer_size = 0;
    void *buffer = nullptr;
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        state->sparse_handle, state->constraint_matrix->num_rows,
        state->constraint_matrix->num_cols,
        state->constraint_matrix->num_nonzeros, state->constraint_matrix->val,
        state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->col_ind, CUDA_R_64F,
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG_DEFAULT, &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        state->sparse_handle, state->constraint_matrix->num_rows,
        state->constraint_matrix->num_cols,
        state->constraint_matrix->num_nonzeros, state->constraint_matrix->val,
        state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->col_ind, CUDA_R_64F,
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG_DEFAULT, buffer));

    CUDA_CHECK(cudaFree(buffer));
  } else {
    CUDA_CHECK(cudaMemset(state->constraint_matrix_t->row_ptr, 0,
                          (state->n_active_vars + 1) * sizeof(int)));
  }
  CUDA_CHECK(cudaGetLastError());
  ALLOC_AND_COPY(state->right_hand_side, sdp_problem->right_hand_side,
                 n_cons * sizeof(double));

  int total_lowrank_size = 0;
  for (int b = 0; b < state->n_blks; b++) {
    total_lowrank_size += state->blk_dims[b] * state->rank_list[b];
  }

  state->lp_solution_offset = total_lowrank_size;
  total_lowrank_size += state->lp_dim;

  state->length_low_rank_solution = total_lowrank_size;
  ALLOC_ZERO(state->low_rank_solution, total_lowrank_size * sizeof(double));

  randomize_device_array(state->low_rank_solution,
                         state->length_low_rank_solution,
                         get_rank_aware_seed(42u));
  ALLOC_ZERO(state->low_rank_direction, total_lowrank_size * sizeof(double));
  ALLOC_ZERO(state->low_rank_gradient, total_lowrank_size * sizeof(double));
  // ======================================================================

  // LP Objective Vector
  if (state->lp_dim > 0) {
    CUDA_CHECK(cudaMalloc(&state->lp_objective_vector,
                          state->lp_dim * sizeof(double)));
    if (sdp_problem->lp_objective_vector != NULL) {
      CUDA_CHECK(cudaMemcpy(
          state->lp_objective_vector, sdp_problem->lp_objective_vector,
          state->lp_dim * sizeof(double), cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK(cudaMemset(state->lp_objective_vector, 0,
                            state->lp_dim * sizeof(double)));
    }
    CUDA_CHECK(
        cudaMalloc(&state->lp_slack_buffer, state->lp_dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->lp_min_slack_buf, sizeof(double)));
  } else {
    state->lp_objective_vector = NULL;
    state->lp_min_slack_buf = NULL;
  }

  ALLOC_ZERO(state->primal_solution, state->n_active_vars * sizeof(double));
  ALLOC_ZERO(state->primal_direct_double,
             state->n_active_vars * sizeof(double));
  ALLOC_ZERO(state->primal_direct_solution_cross,
             state->n_active_vars * sizeof(double));
  ALLOC_ZERO(state->dual_solution, n_cons * sizeof(double));
  ALLOC_ZERO(state->dual_product, state->n_active_vars * sizeof(double));
  ALLOC_ZERO(state->primal_product, n_cons * sizeof(double));
  ALLOC_ZERO(state->q0, n_cons * sizeof(double));
  ALLOC_ZERO(state->q1, n_cons * sizeof(double));
  ALLOC_ZERO(state->q2, n_cons * sizeof(double));

  // SpMV Buffer
  size_t primal_spmv_buffer_size;
  size_t dual_spmv_buffer_size;

  CUSPARSE_CHECK(cusparseCreateCsr(
      &state->matA, state->num_constraints, state->n_active_vars,
      state->constraint_matrix->num_nonzeros, state->constraint_matrix->row_ptr,
      state->constraint_matrix->col_ind, state->constraint_matrix->val,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_64F));

  CUDA_CHECK(cudaGetLastError());

  CUSPARSE_CHECK(cusparseCreateCsr(
      &state->matAt, state->n_active_vars, state->num_constraints,
      state->constraint_matrix_t->num_nonzeros,
      state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
      state->constraint_matrix_t->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  CUDA_CHECK(cudaGetLastError());

  CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_sol,
                                     state->n_active_vars,
                                     state->primal_solution, CUDA_R_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_sol,
                                     state->num_constraints,
                                     state->dual_solution, CUDA_R_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_prod,
                                     state->num_constraints,
                                     state->primal_product, CUDA_R_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_prod,
                                     state->n_active_vars, state->dual_product,
                                     CUDA_R_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_q1, state->num_constraints,
                                     state->q1, CUDA_R_64F));
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
      state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &primal_spmv_buffer_size));

  CUSPARSE_CHECK(cusparseSpMV_bufferSize(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
      state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &dual_spmv_buffer_size));
  CUDA_CHECK(cudaMalloc(&state->primal_spmv_buffer, primal_spmv_buffer_size));
  CUSPARSE_CHECK(cusparseSpMV_preprocess(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
      state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

  CUDA_CHECK(cudaMalloc(&state->dual_spmv_buffer, dual_spmv_buffer_size));
  CUSPARSE_CHECK(cusparseSpMV_preprocess(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
      state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

  // Initialize the per-state CUDA stream pool shared by all cones.
  state->cone_stream_pool_size =
      (int)(sizeof(state->cone_stream_pool) / sizeof(state->cone_stream_pool[0]));
  for (int i = 0; i < state->cone_stream_pool_size; i++) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&state->cone_stream_pool[i],
                                         cudaStreamNonBlocking));
  }

  // Per-cone p2 accumulator buffers for COMPUTE_EXACT_STEP_SIZE.
  CUDA_CHECK(cudaMalloc(&state->d_p2_per_cone, n_cones * sizeof(double)));
  state->h_p2_per_cone = (double *)safe_malloc(n_cones * sizeof(double));

  CUDA_CHECK(cudaMalloc(&state->d_step_scalars, 11 * sizeof(double)));
  {
    double two_h = 2.0;
    CUDA_CHECK(cudaMemcpy(state->d_step_scalars + 10, &two_h, sizeof(double),
                          cudaMemcpyHostToDevice));
  }

  {
    int m = state->lbfgs_history_size > 0 ? state->lbfgs_history_size : 5;
    int n = state->length_low_rank_solution;
    state->lbfgs_buf_capacity_m = m;
    state->lbfgs_buf_capacity_n = n;
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_S, (size_t)m * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_Y, (size_t)m * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_R_old, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_Grad_old, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_s, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_q, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_z, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_rho, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_alpha, m * sizeof(double)));
    CUDA_CHECK(
        cudaMalloc(&state->d_lbfgs_scratch, LBFGS_SCR_SIZE * sizeof(double)));

    double neg_one_h = -1.0;
    CUDA_CHECK(cudaMemcpy(state->d_lbfgs_scratch + LBFGS_SCR_NEG_ONE,
                          &neg_one_h, sizeof(double), cudaMemcpyHostToDevice));
  }

  // Block Low Rank State
  state->block_low_rank_state = (block_low_rank_state_t **)safe_malloc(
      n_cones * sizeof(block_low_rank_state_t *));

  int act_idx = 0;
  int obj_idx = 0;
  long long r_offset = 0;
  int psd_scale_offset = 0;
  for (int b = 0; b < n_cones; b++) {
    state->block_low_rank_state[b] = (block_low_rank_state_t *)safe_calloc(
        1, sizeof(block_low_rank_state_t));
    block_low_rank_state_t *blk_state = state->block_low_rank_state[b];

    int n_k = state->blk_dims[b];
    int rank = state->rank_list[b];
    long long start_gidx = state->blk_ptr[b];
    long long end_gidx = state->blk_ptr[b + 1];

    blk_state->dim = n_k;
    blk_state->rank = rank;
    
    // Alias into the state-level pool; never destroyed per-cone.
    blk_state->stream =
        state->cone_stream_pool[b % state->cone_stream_pool_size];

    blk_state->solution = state->low_rank_solution + r_offset;
    blk_state->direction = state->low_rank_direction + r_offset;
    blk_state->gradient = state->low_rank_gradient + r_offset;

    r_offset += (long long)n_k * rank;

    int A_start = act_idx;
    while (act_idx < n_act_vars &&
           sdp_problem->col_mapping[act_idx] < end_gidx) {
      act_idx++;
    }
    int nnz_A = act_idx - A_start;

    temp_coo_t *coo_A = NULL;
    if (nnz_A > 0) {
      coo_A = (temp_coo_t *)safe_malloc(nnz_A * sizeof(temp_coo_t));
      for (int i = 0; i < nnz_A; i++) {
        int compact_id = A_start + i;
        long long g_idx = sdp_problem->col_mapping[compact_id];
        long long local_idx = g_idx - start_gidx;

        coo_A[i].c = (int)(local_idx / n_k);
        coo_A[i].r = (int)(local_idx % n_k);
        coo_A[i].orig_idx = compact_id;
      }
      qsort(coo_A, nnz_A, sizeof(temp_coo_t), cmp_temp_coo);
    }

    int C_start = obj_idx;
    while (obj_idx < sdp_problem->objective_vector_sparse->len &&
           sdp_problem->objective_vector_sparse->pos[obj_idx] < end_gidx) {
      obj_idx++;
    }
    int nnz_C = obj_idx - C_start;

    temp_coo_t *coo_C = NULL;
    if (nnz_C > 0) {
      coo_C = (temp_coo_t *)safe_malloc(nnz_C * sizeof(temp_coo_t));
      for (int i = 0; i < nnz_C; i++) {
        long long g_idx =
            sdp_problem->objective_vector_sparse->pos[C_start + i];
        long long local_idx = g_idx - start_gidx;

        coo_C[i].c = (int)(local_idx / n_k);
        coo_C[i].r = (int)(local_idx % n_k);
        coo_C[i].val = sdp_problem->objective_vector_sparse->val[C_start + i];
      }
      qsort(coo_C, nnz_C, sizeof(temp_coo_t), cmp_temp_coo);
    }

    int max_nnz_union = nnz_A + nnz_C;
    temp_coo_t *coo_Union =
        (temp_coo_t *)safe_malloc(max_nnz_union * sizeof(temp_coo_t));

    double *h_objective_val =
        (double *)safe_malloc(max_nnz_union * sizeof(double));
    int *h_constraint_to_union =
        (int *)safe_malloc((nnz_A > 0 ? nnz_A : 1) * sizeof(int));
    int *h_compat_mapping =
        (int *)safe_malloc((nnz_A > 0 ? nnz_A : 1) * sizeof(int));

    int i = 0;
    int j = 0;
    int u_idx = 0;

    while (i < nnz_A || j < nnz_C) {
      int cmp = 0;
      if (i == nnz_A)
        cmp = 1;
      else if (j == nnz_C)
        cmp = -1;
      else
        cmp = cmp_temp_coo(&coo_A[i], &coo_C[j]);

      if (cmp < 0) {
        coo_Union[u_idx].r = coo_A[i].r;
        coo_Union[u_idx].c = coo_A[i].c;
        h_objective_val[u_idx] = 0.0;
        h_compat_mapping[i] = coo_A[i].orig_idx;
        h_constraint_to_union[i] = u_idx;
        i++;
      } else if (cmp > 0) {
        coo_Union[u_idx].r = coo_C[j].r;
        coo_Union[u_idx].c = coo_C[j].c;
        h_objective_val[u_idx] = coo_C[j].val;
        j++;
      } else {
        coo_Union[u_idx].r = coo_A[i].r;
        coo_Union[u_idx].c = coo_A[i].c;
        h_objective_val[u_idx] = coo_C[j].val;
        h_compat_mapping[i] = coo_A[i].orig_idx;
        h_constraint_to_union[i] = u_idx;
        i++;
        j++;
      }
      u_idx++;
    }
    int nnz_Union = u_idx;

    int *h_row_ptr_A = (int *)calloc((n_k + 1), sizeof(int));
    int *h_col_ind_A =
        (int *)safe_malloc((nnz_A > 0 ? nnz_A : 1) * sizeof(int));
    for (int p = 0; p < nnz_A; p++) {
      h_col_ind_A[p] = coo_A[p].c;
      h_row_ptr_A[coo_A[p].r + 1]++;
    }
    for (int p = 0; p < n_k; p++)
      h_row_ptr_A[p + 1] += h_row_ptr_A[p];

    int *h_row_ptr_U = (int *)calloc((n_k + 1), sizeof(int));
    int *h_col_ind_U =
        (int *)safe_malloc((nnz_Union > 0 ? nnz_Union : 1) * sizeof(int));
    for (int p = 0; p < nnz_Union; p++) {
      h_col_ind_U[p] = coo_Union[p].c;
      h_row_ptr_U[coo_Union[p].r + 1]++;
    }
    for (int p = 0; p < n_k; p++)
      h_row_ptr_U[p + 1] += h_row_ptr_U[p];

    if (nnz_A > 0) {
      blk_state->constraint_sparse_pattern =
          (sparse_csr_matrix_t *)safe_malloc(sizeof(sparse_csr_matrix_t));
      blk_state->constraint_sparse_pattern->num_rows = n_k;
      blk_state->constraint_sparse_pattern->num_cols = n_k;
      blk_state->constraint_sparse_pattern->num_nonzeros = nnz_A;

      CUDA_CHECK(cudaMalloc(&blk_state->constraint_sparse_pattern->row_ptr,
                            (n_k + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&blk_state->constraint_sparse_pattern->col_ind,
                            nnz_A * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&blk_state->constraint_sparse_pattern->val,
                            nnz_A * sizeof(double)));

      CUDA_CHECK(cudaMalloc(&blk_state->compat_mapping, nnz_A * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&blk_state->constraint_to_union_mapping,
                            nnz_A * sizeof(int)));

      CUDA_CHECK(cudaMemcpy(blk_state->constraint_sparse_pattern->row_ptr,
                            h_row_ptr_A, (n_k + 1) * sizeof(int),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(blk_state->constraint_sparse_pattern->col_ind,
                            h_col_ind_A, nnz_A * sizeof(int),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(blk_state->compat_mapping, h_compat_mapping,
                            nnz_A * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(blk_state->constraint_to_union_mapping,
                            h_constraint_to_union, nnz_A * sizeof(int),
                            cudaMemcpyHostToDevice));
    }

    if (nnz_Union > 0) {
      blk_state->objective_union_constraint_sparse_pattern =
          (sparse_csr_matrix_t *)safe_malloc(sizeof(sparse_csr_matrix_t));
      blk_state->objective_union_constraint_sparse_pattern->num_rows = n_k;
      blk_state->objective_union_constraint_sparse_pattern->num_cols = n_k;
      blk_state->objective_union_constraint_sparse_pattern->num_nonzeros =
          nnz_Union;

      CUDA_CHECK(cudaMalloc(
          &blk_state->objective_union_constraint_sparse_pattern->row_ptr,
          (n_k + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc(
          &blk_state->objective_union_constraint_sparse_pattern->col_ind,
          nnz_Union * sizeof(int)));
      CUDA_CHECK(
          cudaMalloc(&blk_state->objective_union_constraint_sparse_pattern->val,
                     nnz_Union * sizeof(double)));
      CUDA_CHECK(
          cudaMalloc(&blk_state->objective_val, nnz_Union * sizeof(double)));
      CUDA_CHECK(cudaMemcpy(
          blk_state->objective_union_constraint_sparse_pattern->row_ptr,
          h_row_ptr_U, (n_k + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(
          blk_state->objective_union_constraint_sparse_pattern->col_ind,
          h_col_ind_U, nnz_Union * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(blk_state->objective_val, h_objective_val,
                            nnz_Union * sizeof(double),
                            cudaMemcpyHostToDevice));
    }

    const double *cone_d =
        (rescale_info && rescale_info->psd_cone_rescaling)
            ? rescale_info->psd_cone_rescaling + psd_scale_offset
            : NULL;
    populate_block_psd_cone_rescaling(blk_state, n_k, nnz_Union, h_row_ptr_U,
                                       h_col_ind_U, cone_d);

    if (coo_A)
      free(coo_A);
    if (coo_C)
      free(coo_C);
    free(coo_Union);
    free(h_objective_val);
    free(h_constraint_to_union);
    free(h_compat_mapping);
    free(h_row_ptr_A);
    free(h_col_ind_A);
    free(h_row_ptr_U);
    free(h_col_ind_U);

    CUSPARSE_CHECK(cusparseCreateDnMat(&blk_state->matR, n_k, rank, n_k,
                                       blk_state->solution, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&blk_state->matD, n_k, rank, n_k,
                                       blk_state->direction, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&blk_state->matGrad, n_k, rank, n_k,
                                       blk_state->gradient, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));

    if (nnz_A > 0) {
      CUSPARSE_CHECK(cusparseCreateCsr(
          &blk_state->matSpA, n_k, n_k, nnz_A,
          blk_state->constraint_sparse_pattern->row_ptr,
          blk_state->constraint_sparse_pattern->col_ind,
          blk_state->constraint_sparse_pattern->val, CUSPARSE_INDEX_32I,
          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

      CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_TRANSPOSE, &HOST_ONE, blk_state->matR,
          blk_state->matR, &HOST_ZERO, blk_state->matSpA, CUDA_R_64F,
          CUSPARSE_SDDMM_ALG_DEFAULT, &blk_state->sddmm_buffer_size_A));

      if (blk_state->sddmm_buffer_size_A > 0) {
        CUDA_CHECK(cudaMalloc(&blk_state->sddmm_buffer_A,
                              blk_state->sddmm_buffer_size_A));
      } else {
        blk_state->sddmm_buffer_A = NULL;
      }

      CUSPARSE_CHECK(cusparseSDDMM_preprocess(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_TRANSPOSE, &HOST_ONE, blk_state->matR,
          blk_state->matR, &HOST_ZERO, blk_state->matSpA, CUDA_R_64F,
          CUSPARSE_SDDMM_ALG_DEFAULT, blk_state->sddmm_buffer_A));
    }

    if (nnz_Union > 0) {
      CUSPARSE_CHECK(cusparseCreateCsr(
          &blk_state->matSpC, n_k, n_k, nnz_Union,
          blk_state->objective_union_constraint_sparse_pattern->row_ptr,
          blk_state->objective_union_constraint_sparse_pattern->col_ind,
          blk_state->objective_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

      CUSPARSE_CHECK(cusparseCreateCsr(
          &blk_state->matSpS, n_k, n_k, nnz_Union,
          blk_state->objective_union_constraint_sparse_pattern->row_ptr,
          blk_state->objective_union_constraint_sparse_pattern->col_ind,
          blk_state->objective_union_constraint_sparse_pattern->val,
          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
          CUDA_R_64F));

      CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_TRANSPOSE, &HOST_ONE, blk_state->matR,
          blk_state->matR, &HOST_ZERO, blk_state->matSpC, CUDA_R_64F,
          CUSPARSE_SDDMM_ALG_DEFAULT, &blk_state->sddmm_buffer_size_C));

      if (blk_state->sddmm_buffer_size_C > 0) {
        CUDA_CHECK(cudaMalloc(&blk_state->sddmm_buffer_C,
                              blk_state->sddmm_buffer_size_C));
      } else {
        blk_state->sddmm_buffer_C = NULL;
      }

      CUSPARSE_CHECK(cusparseSDDMM_preprocess(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_TRANSPOSE, &HOST_ONE, blk_state->matR,
          blk_state->matR, &HOST_ZERO, blk_state->matSpC, CUDA_R_64F,
          CUSPARSE_SDDMM_ALG_DEFAULT, blk_state->sddmm_buffer_C));

      CUSPARSE_CHECK(cusparseSpMM_bufferSize(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, blk_state->matSpS,
          blk_state->matR, &HOST_ZERO, blk_state->matGrad, CUDA_R_64F,
          CUSPARSE_SPMM_ALG_DEFAULT, &blk_state->spmm_buffer_size_S));

      if (blk_state->spmm_buffer_size_S > 0) {
        CUDA_CHECK(cudaMalloc(&blk_state->spmm_buffer_S,
                              blk_state->spmm_buffer_size_S));
      } else {
        blk_state->spmm_buffer_S = NULL;
      }

      CUSPARSE_CHECK(cusparseSpMM_preprocess(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, blk_state->matSpS,
          blk_state->matR, &HOST_ZERO, blk_state->matGrad, CUDA_R_64F,
          CUSPARSE_SPMM_ALG_DEFAULT, blk_state->spmm_buffer_S));
    }

    psd_scale_offset += n_k;
  }

  populate_state_scaling_fields(state, rescale_info);

  build_cone_batches(state);

  return state;
}

static void build_one_cone_batch(cardal_sdp_solver_state_t *state,
                                 block_low_rank_state_t *b, int *blk_indices,
                                 int n_cones, int dim,
                                 cone_batch_kind_t kind) {
  // NOTE: do NOT memset(b) here. Per-cone init has already populated the
  // scalar fields on this entry; we only touch identity + bdata.
  b->dim = dim;
  b->n_cones = n_cones;
  b->kind = kind;

  if (kind != CONE_BATCH_KIND_CUSTOM) {
    // PERCONE 1-cone path: caller doesn't keep blk_indices, free it.
    b->bdata = NULL;
    free(blk_indices);
    return;
  }

  // Allocate the CUSTOM batched payload and populate per-cone descriptors
  // + flat sparsity.
  b->bdata = (cone_batch_data_t *)safe_calloc(1, sizeof(cone_batch_data_t));
  cone_batch_data_t *bd = b->bdata;
  bd->blk_idx_h = blk_indices; // ownership transferred

  // Per-cone host arrays
  int *h_ranks = (int *)safe_malloc(n_cones * sizeof(int));
  long long *h_R_off = (long long *)safe_malloc(n_cones * sizeof(long long));
  long long *h_G_off = (long long *)safe_malloc(n_cones * sizeof(long long));
  long long *h_D_off = (long long *)safe_malloc(n_cones * sizeof(long long));
  int max_rank = 0;
  long long total_nnz_A = 0, total_nnz_S = 0;
  for (int c = 0; c < n_cones; c++) {
    int blk_idx = blk_indices[c];
    block_low_rank_state_t *blk = state->block_low_rank_state[blk_idx];
    h_ranks[c] = blk->rank;
    if (blk->rank > max_rank)
      max_rank = blk->rank;
    h_R_off[c] = (long long)(blk->solution - state->low_rank_solution);
    h_G_off[c] = (long long)(blk->gradient - state->low_rank_gradient);
    h_D_off[c] = (long long)(blk->direction - state->low_rank_direction);
    if (blk->constraint_sparse_pattern != NULL)
      total_nnz_A += blk->constraint_sparse_pattern->num_nonzeros;
    if (blk->objective_union_constraint_sparse_pattern != NULL)
      total_nnz_S +=
          blk->objective_union_constraint_sparse_pattern->num_nonzeros;
  }
  bd->max_rank = max_rank;
  bd->total_nnz_A = (int)total_nnz_A;
  bd->total_nnz_S = (int)total_nnz_S;

  // Per-cone descriptors on device
  CUDA_CHECK(cudaMalloc(&bd->d_blk_idx, n_cones * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(bd->d_blk_idx, blk_indices, n_cones * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&bd->d_ranks, n_cones * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&bd->d_R_offsets, n_cones * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&bd->d_G_offsets, n_cones * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&bd->d_D_offsets, n_cones * sizeof(long long)));
  CUDA_CHECK(cudaMemcpy(bd->d_ranks, h_ranks, n_cones * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(bd->d_R_offsets, h_R_off,
                        n_cones * sizeof(long long), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(bd->d_G_offsets, h_G_off,
                        n_cones * sizeof(long long), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(bd->d_D_offsets, h_D_off,
                        n_cones * sizeof(long long), cudaMemcpyHostToDevice));
  free(h_ranks);
  free(h_R_off);
  free(h_G_off);
  free(h_D_off);

  int n_small = n_cones;
  int *small = blk_indices;

  // Build flat sparsity for matSpA.
  int *h_eA_cone = NULL, *h_eA_row = NULL, *h_eA_col = NULL,
      *h_eA_compat = NULL, *h_eA_to_flatS = NULL;
  if (total_nnz_A > 0) {
    h_eA_cone = (int *)safe_malloc(total_nnz_A * sizeof(int));
    h_eA_row = (int *)safe_malloc(total_nnz_A * sizeof(int));
    h_eA_col = (int *)safe_malloc(total_nnz_A * sizeof(int));
    h_eA_compat = (int *)safe_malloc(total_nnz_A * sizeof(int));
    h_eA_to_flatS = (int *)safe_malloc(total_nnz_A * sizeof(int));
  }

  int *h_eS_cone = NULL, *h_eS_row = NULL, *h_eS_col = NULL;
  double *h_eS_objval = NULL;
  int *h_cone_S_off = (int *)safe_malloc((n_small + 1) * sizeof(int));
  h_cone_S_off[0] = 0;
  if (total_nnz_S > 0) {
    h_eS_cone = (int *)safe_malloc(total_nnz_S * sizeof(int));
    h_eS_row = (int *)safe_malloc(total_nnz_S * sizeof(int));
    h_eS_col = (int *)safe_malloc(total_nnz_S * sizeof(int));
    h_eS_objval = (double *)safe_malloc(total_nnz_S * sizeof(double));
  }

  // Per-cone matSpS row_ptr flattened. Each cone contributes dim+1 ints.
  int total_row_ptr_len = 0;
  for (int s = 0; s < n_small; s++)
    total_row_ptr_len += state->block_low_rank_state[small[s]]->dim + 1;
  int *h_S_row_ptr_flat = (int *)safe_malloc(total_row_ptr_len * sizeof(int));
  int *h_S_row_ptr_off = (int *)safe_malloc((n_small + 1) * sizeof(int));
  h_S_row_ptr_off[0] = 0;
  for (int s = 0; s < n_small; s++)
    h_S_row_ptr_off[s + 1] =
        h_S_row_ptr_off[s] + state->block_low_rank_state[small[s]]->dim + 1;

  int write_A = 0, write_S = 0;
  for (int s = 0; s < n_small; s++) {
    int blk_idx = small[s];
    block_low_rank_state_t *blk = state->block_low_rank_state[blk_idx];
    int dim = blk->dim;

    int cone_S_start_for_A = write_S; // S offset for this cone (consistent
                                       // with the parallel loop below)
    if (blk->constraint_sparse_pattern != NULL) {
      sparse_csr_matrix_t *A = blk->constraint_sparse_pattern;
      int nnz = A->num_nonzeros;
      if (nnz > 0) {
        int *h_row_ptr = (int *)safe_malloc((dim + 1) * sizeof(int));
        int *h_col_ind = (int *)safe_malloc(nnz * sizeof(int));
        int *h_compat = (int *)safe_malloc(nnz * sizeof(int));
        int *h_c2u = (int *)safe_malloc(nnz * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_row_ptr, A->row_ptr,
                              (dim + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_col_ind, A->col_ind, nnz * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_compat, blk->compat_mapping,
                              nnz * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_c2u, blk->constraint_to_union_mapping,
                              nnz * sizeof(int), cudaMemcpyDeviceToHost));
        int within_cone_A_idx = 0;
        for (int i = 0; i < dim; i++) {
          for (int p = h_row_ptr[i]; p < h_row_ptr[i + 1]; p++) {
            h_eA_cone[write_A] = s;
            h_eA_row[write_A] = i;
            h_eA_col[write_A] = h_col_ind[p];
            h_eA_compat[write_A] = h_compat[p];
            h_eA_to_flatS[write_A] =
                cone_S_start_for_A + h_c2u[within_cone_A_idx];
            within_cone_A_idx++;
            write_A++;
          }
        }
        free(h_row_ptr);
        free(h_col_ind);
        free(h_compat);
        free(h_c2u);
      }
    }

    for (int r = 0; r <= dim; r++)
      h_S_row_ptr_flat[h_S_row_ptr_off[s] + r] = 0;
    if (blk->objective_union_constraint_sparse_pattern != NULL) {
      sparse_csr_matrix_t *S = blk->objective_union_constraint_sparse_pattern;
      int nnz = S->num_nonzeros;
      if (nnz > 0) {
        int *h_row_ptr = (int *)safe_malloc((dim + 1) * sizeof(int));
        int *h_col_ind = (int *)safe_malloc(nnz * sizeof(int));
        double *h_objval = (double *)safe_malloc(nnz * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_row_ptr, S->row_ptr,
                              (dim + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_col_ind, S->col_ind, nnz * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_objval, blk->objective_val,
                              nnz * sizeof(double),
                              cudaMemcpyDeviceToHost));
        for (int r = 0; r <= dim; r++)
          h_S_row_ptr_flat[h_S_row_ptr_off[s] + r] = h_row_ptr[r];
        for (int i = 0; i < dim; i++) {
          for (int p = h_row_ptr[i]; p < h_row_ptr[i + 1]; p++) {
            h_eS_cone[write_S] = s;
            h_eS_row[write_S] = i;
            h_eS_col[write_S] = h_col_ind[p];
            h_eS_objval[write_S] = h_objval[p];
            write_S++;
          }
        }
        free(h_row_ptr);
        free(h_col_ind);
        free(h_objval);
      }
    }
    h_cone_S_off[s + 1] = write_S;
  }

  if (total_nnz_A > 0) {
    CUDA_CHECK(cudaMalloc(&bd->d_entry_cone_A, total_nnz_A * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bd->d_entry_row_A, total_nnz_A * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bd->d_entry_col_A, total_nnz_A * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bd->d_entry_compat_A, total_nnz_A * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bd->d_entry_A_to_flatS, total_nnz_A * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(bd->d_entry_cone_A, h_eA_cone,
                          total_nnz_A * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd->d_entry_row_A, h_eA_row,
                          total_nnz_A * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd->d_entry_col_A, h_eA_col,
                          total_nnz_A * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd->d_entry_compat_A, h_eA_compat,
                          total_nnz_A * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd->d_entry_A_to_flatS, h_eA_to_flatS,
                          total_nnz_A * sizeof(int), cudaMemcpyHostToDevice));
    free(h_eA_cone);
    free(h_eA_row);
    free(h_eA_col);
    free(h_eA_compat);
    free(h_eA_to_flatS);
  }

  if (total_nnz_S > 0) {
    CUDA_CHECK(cudaMalloc(&bd->d_entry_cone_S, total_nnz_S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bd->d_entry_row_S, total_nnz_S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bd->d_entry_col_S, total_nnz_S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bd->d_flat_objval_S, total_nnz_S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bd->d_flat_spS_val, total_nnz_S * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(bd->d_entry_cone_S, h_eS_cone,
                          total_nnz_S * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd->d_entry_row_S, h_eS_row,
                          total_nnz_S * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd->d_entry_col_S, h_eS_col,
                          total_nnz_S * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd->d_flat_objval_S, h_eS_objval,
                          total_nnz_S * sizeof(double),
                          cudaMemcpyHostToDevice));
    free(h_eS_cone);
    free(h_eS_row);
    free(h_eS_col);
    free(h_eS_objval);
  }

  CUDA_CHECK(cudaMalloc(&bd->d_cone_S_offsets, (n_small + 1) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(bd->d_cone_S_offsets, h_cone_S_off,
                        (n_small + 1) * sizeof(int), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&bd->d_S_row_ptr_flat,
                        total_row_ptr_len * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(bd->d_S_row_ptr_flat, h_S_row_ptr_flat,
                        total_row_ptr_len * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMalloc(&bd->d_S_row_ptr_offsets, (n_small + 1) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(bd->d_S_row_ptr_offsets, h_S_row_ptr_off,
                        (n_small + 1) * sizeof(int), cudaMemcpyHostToDevice));
  free(h_S_row_ptr_flat);
  free(h_S_row_ptr_off);

  if (total_nnz_S > 0) {
    for (int s = 0; s < n_small; s++) {
      int blk_idx = small[s];
      block_low_rank_state_t *blk = state->block_low_rank_state[blk_idx];
      if (blk->objective_union_constraint_sparse_pattern == NULL)
        continue;
      sparse_csr_matrix_t *S = blk->objective_union_constraint_sparse_pattern;
      if (S->num_nonzeros == 0)
        continue;
      double *new_val = bd->d_flat_spS_val + h_cone_S_off[s];
      CUDA_CHECK(cudaFree(S->val));
      S->val = new_val;
      CUSPARSE_CHECK(cusparseCsrSetPointers(blk->matSpS, S->row_ptr,
                                            S->col_ind, S->val));
    }
  }
  free(h_cone_S_off);
}

// Bucket cones by dim; small cones go to CUSTOM batches, larger to PERCONE.
void build_cone_batches(cardal_sdp_solver_state_t *state) {
  int n_blks = state->n_blks;
  state->n_batches = 0;
  state->batch_leaders = NULL;
  if (n_blks <= 0)
    return;

  int max_unique = n_blks;
  int *dim_seen = (int *)safe_malloc(max_unique * sizeof(int));
  int **indexes = (int **)safe_malloc(max_unique * sizeof(int *));
  int *counts = (int *)safe_malloc(max_unique * sizeof(int));
  int n_unique = 0;
  for (int blk_idx = 0; blk_idx < n_blks; blk_idx++) {
    int d = state->block_low_rank_state[blk_idx]->dim;
    int slot = -1;
    for (int s = 0; s < n_unique; s++) {
      if (dim_seen[s] == d) {
        slot = s;
        break;
      }
    }
    if (slot < 0) {
      slot = n_unique++;
      dim_seen[slot] = d;
      indexes[slot] = (int *)safe_malloc(n_blks * sizeof(int));
      counts[slot] = 0;
    }
    indexes[slot][counts[slot]++] = blk_idx;
  }

#define IS_CUSTOM_BUCKET(d, c)                                                 \
  ((d) <= CARDAL_SMALL_CONE_DIM_THRESHOLD && (c) >= CARDAL_MIN_BATCH_SIZE)

  for (int i = 0; i < n_unique - 1; i++) {
    for (int j = 0; j < n_unique - 1 - i; j++) {
      int a_custom = IS_CUSTOM_BUCKET(dim_seen[j], counts[j]);
      int b_custom = IS_CUSTOM_BUCKET(dim_seen[j + 1], counts[j + 1]);
      int swap = 0;
      if (a_custom != b_custom) {
        if (!a_custom && b_custom)
          swap = 1;
      } else if (dim_seen[j] > dim_seen[j + 1]) {
        swap = 1;
      }
      if (swap) {
        int td = dim_seen[j];
        dim_seen[j] = dim_seen[j + 1];
        dim_seen[j + 1] = td;
        int *ti = indexes[j];
        indexes[j] = indexes[j + 1];
        indexes[j + 1] = ti;
        int tc = counts[j];
        counts[j] = counts[j + 1];
        counts[j + 1] = tc;
      }
    }
  }

  int n_batches = 0;
  for (int s = 0; s < n_unique; s++) {
    if (IS_CUSTOM_BUCKET(dim_seen[s], counts[s]))
      n_batches += 1;
    else
      n_batches += counts[s];
  }
  state->n_batches = n_batches;
  state->batch_leaders = (int *)safe_malloc(n_batches * sizeof(int));

  int bi = 0;
  for (int s = 0; s < n_unique; s++) {
    int is_custom = IS_CUSTOM_BUCKET(dim_seen[s], counts[s]);
    if (is_custom) {
      int *blk_indices = (int *)safe_malloc(counts[s] * sizeof(int));
      memcpy(blk_indices, indexes[s], counts[s] * sizeof(int));
      free(indexes[s]);
      int leader = blk_indices[0];
      state->batch_leaders[bi++] = leader;
      build_one_cone_batch(state, state->block_low_rank_state[leader],
                           blk_indices, counts[s], dim_seen[s],
                           CONE_BATCH_KIND_CUSTOM);
    } else {
      for (int c = 0; c < counts[s]; c++) {
        int blk_idx = indexes[s][c];
        int *one = (int *)safe_malloc(sizeof(int));
        one[0] = blk_idx;
        state->batch_leaders[bi++] = blk_idx;
        build_one_cone_batch(state, state->block_low_rank_state[blk_idx], one,
                             1, dim_seen[s], CONE_BATCH_KIND_PERCONE);
      }
      free(indexes[s]);
    }
  }
  free(dim_seen);
  free(indexes);
  free(counts);
#undef IS_CUSTOM_BUCKET
}

void refresh_cone_batches(cardal_sdp_solver_state_t *state) {
  for (int bi = 0; bi < state->n_batches; bi++) {
    block_low_rank_state_t *b =
        state->block_low_rank_state[state->batch_leaders[bi]];
    if (b->kind != CONE_BATCH_KIND_CUSTOM || b->bdata == NULL)
      continue;
    cone_batch_data_t *bd = b->bdata;
    int n = b->n_cones;
    int *h_ranks = (int *)safe_malloc(n * sizeof(int));
    long long *h_R_off = (long long *)safe_malloc(n * sizeof(long long));
    long long *h_G_off = (long long *)safe_malloc(n * sizeof(long long));
    long long *h_D_off = (long long *)safe_malloc(n * sizeof(long long));
    int max_rank = 0;
    for (int c = 0; c < n; c++) {
      int blk_idx = bd->blk_idx_h[c];
      block_low_rank_state_t *blk = state->block_low_rank_state[blk_idx];
      h_ranks[c] = blk->rank;
      if (blk->rank > max_rank)
        max_rank = blk->rank;
      h_R_off[c] = (long long)(blk->solution - state->low_rank_solution);
      h_G_off[c] = (long long)(blk->gradient - state->low_rank_gradient);
      h_D_off[c] = (long long)(blk->direction - state->low_rank_direction);
    }
    bd->max_rank = max_rank;
    CUDA_CHECK(cudaMemcpy(bd->d_ranks, h_ranks, n * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd->d_R_offsets, h_R_off, n * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd->d_G_offsets, h_G_off, n * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd->d_D_offsets, h_D_off, n * sizeof(long long),
                          cudaMemcpyHostToDevice));
    free(h_ranks);
    free(h_R_off);
    free(h_G_off);
    free(h_D_off);
  }
}

// Max neg-curvature directions in the full ALORA SDP per cone; rest use
// rho-aware diagonal closed form.
#ifndef ALORA_SDP_RANK_CAP
#define ALORA_SDP_RANK_CAP 4
#endif

void augment_system_rank(cardal_sdp_solver_state_t *state,
                         const int *rank_incs,
                         double *const *neg_eigvecs,
                         double *const *neg_eigvals) {
  int new_total_len = 0;

  int actually_augmented_blocks = 0;

  for (int b = 0; b < state->n_blks; b++) {
    int dim = state->block_low_rank_state[b]->dim;
    int old_rank = state->block_low_rank_state[b]->rank;
    int theoretical_max_r = state->theoretical_max_rank_per_blk[b];

    int proposed_new_rank = old_rank + rank_incs[b];
    int max_allowed_rank = (dim < theoretical_max_r) ? dim : theoretical_max_r;
    int actual_new_rank = (proposed_new_rank > max_allowed_rank)
                              ? max_allowed_rank
                              : proposed_new_rank;

    if (actual_new_rank > old_rank)
      actually_augmented_blocks++;

    new_total_len += dim * actual_new_rank;
  }

  int new_lp_solution_offset = new_total_len;
  new_total_len += state->lp_dim;

  if (actually_augmented_blocks == 0) {
    if (state->verbose >= 3)
      printf(">>> All targeted blocks have reached maximum full rank. "
             "Augmentation skipped.\n\n");
    return;
  }

  double *new_global_R, *new_global_G, *new_global_D;
  CUDA_CHECK(cudaMalloc(&new_global_R, new_total_len * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&new_global_G, new_total_len * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&new_global_D, new_total_len * sizeof(double)));

  CUDA_CHECK(cudaMemset(new_global_G, 0, new_total_len * sizeof(double)));
  CUDA_CHECK(cudaMemset(new_global_D, 0, new_total_len * sizeof(double)));

  int old_offset = 0;
  int new_offset = 0;

  int tot_rank = 0;
  for (int b = 0; b < state->n_blks; b++) {
    block_low_rank_state_t *blk = state->block_low_rank_state[b];
    int dim = blk->dim;
    int old_rank = blk->rank;

    int proposed_new_rank = old_rank + rank_incs[b];
    int theoretical_max_r = state->theoretical_max_rank_per_blk[b];
    int max_allowed_rank = (dim < theoretical_max_r) ? dim : theoretical_max_r;
    int new_rank = (proposed_new_rank > max_allowed_rank) ? max_allowed_rank
                                                          : proposed_new_rank;
    int actual_inc = new_rank - old_rank;

    if (old_rank > 0) {
      CUDA_CHECK(cudaMemcpy(
          new_global_R + new_offset, state->low_rank_solution + old_offset,
          dim * old_rank * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    if (actual_inc > 0) {
      if (state->verbose >= 3)
        printf("  [Block %d] Rank: %d -> %d (Dim: %d)\n", b, old_rank, new_rank,
               dim);
      double *new_cols = new_global_R + new_offset + (dim * old_rank);
      int new_elements = dim * actual_inc;
      double *eigvec = (neg_eigvecs != NULL) ? neg_eigvecs[b] : NULL;
      double *eigval = (neg_eigvals != NULL) ? neg_eigvals[b] : NULL;

      // gate_fail_streak < K -> noise augment (q0-invariant);
      // streak >= K -> SDP-coupled eigvec augment.
      //
      // Default: always take the noise augment path. The eigvec-direct
      // branch inserts raw negative-curvature eigenvectors as BM columns
      // without q0-budget scaling; at high rho this causes a primal-residual
      // jump after every augment, rho then balloons, and the dual update
      // y += rho*(Ax-b) amplifies feasibility noise into large dobj drift.
      const int noise_augment_override = 1;
      const int sdp_fail_streak_K = 2;
      const double trust_alpha = 1.0;
      int use_noise = noise_augment_override ||
                      (state->gate_fail_streak < sdp_fail_streak_K);
      if (use_noise) {
        eigvec = NULL;
        eigval = NULL;
      }
      double q0_norm = 0.0;
      CUBLAS_CHECK(cublasDnrm2(state->blas_handle, state->num_constraints,
                               state->q0, 1, &q0_norm));
      double q0_norm_budget = trust_alpha * q0_norm;
      int is_batched = (blk->kind == CONE_BATCH_KIND_CUSTOM);
      int single_gpu = (state->grid_context == NULL) ||
                       (state->grid_context->dims[0] == 1 &&
                        state->grid_context->dims[1] == 1 &&
                        state->grid_context->dims[2] == 1);
      int r_cap = actual_inc;
      if (r_cap > ALORA_SDP_RANK_CAP)
        r_cap = ALORA_SDP_RANK_CAP;
      int sdp_ok = (eigvec != NULL && eigval != NULL && !is_batched &&
                    single_gpu);
      if (sdp_ok) {
        double rho = state->penalty_coef;
        double used = solve_alora_sdp_percone(state, blk, eigvec, eigval,
                                              r_cap, rho, q0_norm_budget,
                                              new_cols);
        double remaining_budget = q0_norm_budget;
        if (used > 0.0) {
          double sq = sqrt(q0_norm_budget * q0_norm_budget - used * used);
          remaining_budget = isnan(sq) ? 0.0 : sq;
        }
        for (int c = r_cap; c < actual_inc; c++) {
          double *col = new_cols + (size_t)c * dim;
          if (blk->psd_cone_rescaling != NULL) {
            CUBLAS_CHECK(cublasDdgmm(state->blas_handle, CUBLAS_SIDE_LEFT, dim,
                                     1, eigvec + (size_t)c * dim, dim,
                                     blk->psd_cone_rescaling, 1, col, dim));
          } else {
            CUDA_CHECK(cudaMemcpy(col, eigvec + (size_t)c * dim,
                                  dim * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
          }
          double q = compute_Asuu_norm_sq_percone(state, blk, col);
          double lam = fabs(eigval[c]);
          double s_star = (rho > 0.0 && q > 1e-30) ? lam / (rho * q) : 1e-8;
          int n_diag_remaining = actual_inc - c;
          if (q > 1e-30 && q0_norm_budget > 0.0 && n_diag_remaining > 0) {
            double per_col_budget =
                remaining_budget / sqrt((double)n_diag_remaining);
            double s_cap = per_col_budget / sqrt(q);
            if (s_star > s_cap) s_star = s_cap;
          }
          if (s_star > 1e4) s_star = 1e4;
          double scale = sqrt(s_star);
          CUBLAS_CHECK(cublasDscal(state->blas_handle, dim, &scale, col, 1));
          double used_c = sqrt(q) * s_star;
          double newrem = remaining_budget * remaining_budget - used_c * used_c;
          remaining_budget = (newrem > 0.0) ? sqrt(newrem) : 0.0;
        }
      } else if (eigvec != NULL && !is_batched) {
        if (blk->psd_cone_rescaling != NULL) {
          CUBLAS_CHECK(cublasDdgmm(state->blas_handle, CUBLAS_SIDE_LEFT, dim,
                                   actual_inc, eigvec, dim,
                                   blk->psd_cone_rescaling, 1, new_cols, dim));
        } else {
          CUDA_CHECK(cudaMemcpy(new_cols, eigvec, new_elements * sizeof(double),
                                cudaMemcpyDeviceToDevice));
        }
      } else {

        unsigned aug_seed =
            (unsigned)(0x9E3779B9u +
                       (unsigned)state->num_outer_iteration * 2654435761u +
                       (unsigned)b * 40503u);
        randomize_device_array(new_cols, new_elements, aug_seed);
        double noise_scale = 1e-4;
        CUBLAS_CHECK(cublasDscal(state->blas_handle, new_elements, &noise_scale,
                                 new_cols, 1));
      }
    }

    tot_rank += new_rank;
    state->rank_list[b] = new_rank;
    blk->rank = new_rank;
    blk->solution = new_global_R + new_offset;
    blk->gradient = new_global_G + new_offset;
    blk->direction = new_global_D + new_offset;

    CUSPARSE_CHECK(cusparseDestroyDnMat(blk->matR));
    CUSPARSE_CHECK(cusparseDestroyDnMat(blk->matGrad));
    CUSPARSE_CHECK(cusparseDestroyDnMat(blk->matD));

    CUSPARSE_CHECK(cusparseCreateDnMat(&blk->matR, dim, new_rank, dim,
                                       blk->solution, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&blk->matGrad, dim, new_rank, dim,
                                       blk->gradient, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&blk->matD, dim, new_rank, dim,
                                       blk->direction, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));

    double alpha = 1.0, beta = 0.0;

    if (blk->constraint_sparse_pattern != NULL &&
        blk->constraint_sparse_pattern->num_nonzeros > 0) {
      CUDA_CHECK(cudaFree(blk->sddmm_buffer_A));
      size_t bufferSizeA = 0;
      CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_TRANSPOSE, &alpha, blk->matR, blk->matR, &beta,
          blk->matSpA, CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSizeA));

      if (bufferSizeA > 0) {
        CUDA_CHECK(cudaMalloc(&blk->sddmm_buffer_A, bufferSizeA));
      } else {
        blk->sddmm_buffer_A = NULL;
      }

      CUSPARSE_CHECK(cusparseSDDMM_preprocess(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_TRANSPOSE, &alpha, blk->matR, blk->matR, &beta,
          blk->matSpA, CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT,
          blk->sddmm_buffer_A));
    }

    if (blk->objective_union_constraint_sparse_pattern != NULL &&
        blk->objective_union_constraint_sparse_pattern->num_nonzeros > 0) {
      CUDA_CHECK(cudaFree(blk->sddmm_buffer_C));
      size_t bufferSizeC = 0;
      CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_TRANSPOSE, &alpha, blk->matR, blk->matR, &beta,
          blk->matSpS, CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSizeC));

      if (bufferSizeC > 0) {
        CUDA_CHECK(cudaMalloc(&blk->sddmm_buffer_C, bufferSizeC));
      } else {
        blk->sddmm_buffer_C = NULL;
      }

      CUSPARSE_CHECK(cusparseSDDMM_preprocess(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_TRANSPOSE, &alpha, blk->matR, blk->matR, &beta,
          blk->matSpS, CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT,
          blk->sddmm_buffer_C));

      CUDA_CHECK(cudaFree(blk->spmm_buffer_S));
      size_t bufferSizeS = 0;
      CUSPARSE_CHECK(cusparseSpMM_bufferSize(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, blk->matSpS, blk->matR,
          &beta, blk->matGrad, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
          &bufferSizeS));

      if (bufferSizeS > 0) {
        CUDA_CHECK(cudaMalloc(&blk->spmm_buffer_S, bufferSizeS));
      } else {
        blk->spmm_buffer_S = NULL;
      }
      CUSPARSE_CHECK(cusparseSpMM_preprocess(
          state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, blk->matSpS, blk->matR,
          &beta, blk->matGrad, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
          blk->spmm_buffer_S));
    }

    old_offset += dim * old_rank;
    new_offset += dim * new_rank;
  }

  if (state->lp_dim > 0) {
    CUDA_CHECK(cudaMemcpy(new_global_R + new_lp_solution_offset,
                          state->low_rank_solution + state->lp_solution_offset,
                          state->lp_dim * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    state->lp_solution_offset = new_lp_solution_offset;
  }

  CUDA_CHECK(cudaFree(state->low_rank_solution));
  CUDA_CHECK(cudaFree(state->low_rank_gradient));
  CUDA_CHECK(cudaFree(state->low_rank_direction));

  state->total_rank = tot_rank;
  state->low_rank_solution = new_global_R;
  state->low_rank_gradient = new_global_G;
  state->low_rank_direction = new_global_D;
  state->length_low_rank_solution = new_total_len;

  if (new_total_len > state->lbfgs_buf_capacity_n) {
    int m = state->lbfgs_buf_capacity_m;
    int n = new_total_len;
    CUDA_CHECK(cudaFree(state->d_lbfgs_S));
    CUDA_CHECK(cudaFree(state->d_lbfgs_Y));
    CUDA_CHECK(cudaFree(state->d_lbfgs_R_old));
    CUDA_CHECK(cudaFree(state->d_lbfgs_Grad_old));
    CUDA_CHECK(cudaFree(state->d_lbfgs_s));
    CUDA_CHECK(cudaFree(state->d_lbfgs_y));
    CUDA_CHECK(cudaFree(state->d_lbfgs_q));
    CUDA_CHECK(cudaFree(state->d_lbfgs_z));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_S, (size_t)m * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_Y, (size_t)m * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_R_old, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_Grad_old, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_s, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_q, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_lbfgs_z, n * sizeof(double)));
    state->lbfgs_buf_capacity_n = n;
  }

  refresh_cone_batches(state);

  if (state->verbose >= 3)
    printf(">>> Rank Augmentation Complete. New Total Flat Dim: %d\n\n",
           new_total_len);
}

sdp_result_t *
create_result_from_state(cardal_sdp_solver_state_t *state,
                      const compressed_sdp_problem_t *sdp_problem) {
  sdp_result_t *result = (sdp_result_t *)calloc(1, sizeof(sdp_result_t));
  result->num_variables = sdp_problem->n_active_vars;
  result->num_constraints = sdp_problem->num_constraints;
  result->num_nonzeros = sdp_problem->constraint_matrix->num_nonzeros;
  result->rank = state->total_rank;

  result->low_rank_primal_solution =
      (double *)safe_malloc(state->length_low_rank_solution * sizeof(double));
  CUDA_CHECK(cudaMemcpy(result->low_rank_primal_solution,
                        state->low_rank_solution,
                        state->length_low_rank_solution * sizeof(double),
                        cudaMemcpyDeviceToHost));
  result->low_rank_solution_length = state->length_low_rank_solution;
  result->n_cones = state->n_blks;
  result->rank_list = (int *)safe_malloc(state->n_blks * sizeof(int));
  safe_memcpy(result->rank_list, state->rank_list,
              state->n_blks * sizeof(int));

  result->dual_solution =
      (double *)safe_malloc(state->num_constraints * sizeof(double));
  CUDA_CHECK(cudaMemcpy(result->dual_solution, state->dual_solution,
                        state->num_constraints * sizeof(double),
                        cudaMemcpyDeviceToHost));

  result->total_count = state->num_outer_iteration;
  result->total_inner_count = state->num_inner_iteration;
  result->cumulative_time_sec = state->cumulative_time_sec;
  result->absolute_primal_residual = state->absolute_primal_residual;
  result->relative_primal_residual = state->relative_primal_residual;
  result->absolute_dual_residual = state->absolute_dual_residual;
  result->relative_dual_residual = state->relative_dual_residual;
  result->primal_objective_value = state->primal_objective;
  result->dual_objective_value = state->dual_objective;
  result->objective_gap = state->objective_gap;
  result->relative_objective_gap = state->relative_objective_gap;
  result->termination_reason = state->termination_reason;
  return result;
}

// ---------------------------------------------------------------------
// Free everything allocated by initialize_solver_state (+ any resources
// grown by augment_system_rank / refresh_cone_batches).  Safe on a
// partially-initialised state: each field is NULL-checked, so callers
// can invoke this after any point in setup.
// ---------------------------------------------------------------------
static void free_sparse_csr(sparse_csr_matrix_t *csr) {
  if (csr == NULL)
    return;
  if (csr->row_ptr) CUDA_CHECK(cudaFree(csr->row_ptr));
  if (csr->col_ind) CUDA_CHECK(cudaFree(csr->col_ind));
  if (csr->val)     CUDA_CHECK(cudaFree(csr->val));
  free(csr);
}

// Free a CUSTOM batched-cone's auxiliary data. Owned device arrays only;
// the member blocks' aliased pointers (S->val) must have been detached
// (set to NULL) by the caller BEFORE calling this so that per-block
// teardown does not double-free d_flat_spS_val.
static void free_bdata(cone_batch_data_t *bd) {
  if (bd == NULL) return;
  if (bd->d_blk_idx)           CUDA_CHECK(cudaFree(bd->d_blk_idx));
  if (bd->d_ranks)             CUDA_CHECK(cudaFree(bd->d_ranks));
  if (bd->d_R_offsets)         CUDA_CHECK(cudaFree(bd->d_R_offsets));
  if (bd->d_G_offsets)         CUDA_CHECK(cudaFree(bd->d_G_offsets));
  if (bd->d_D_offsets)         CUDA_CHECK(cudaFree(bd->d_D_offsets));
  if (bd->d_entry_cone_A)      CUDA_CHECK(cudaFree(bd->d_entry_cone_A));
  if (bd->d_entry_row_A)       CUDA_CHECK(cudaFree(bd->d_entry_row_A));
  if (bd->d_entry_col_A)       CUDA_CHECK(cudaFree(bd->d_entry_col_A));
  if (bd->d_entry_compat_A)    CUDA_CHECK(cudaFree(bd->d_entry_compat_A));
  if (bd->d_entry_A_to_flatS)  CUDA_CHECK(cudaFree(bd->d_entry_A_to_flatS));
  if (bd->d_entry_cone_S)      CUDA_CHECK(cudaFree(bd->d_entry_cone_S));
  if (bd->d_entry_row_S)       CUDA_CHECK(cudaFree(bd->d_entry_row_S));
  if (bd->d_entry_col_S)       CUDA_CHECK(cudaFree(bd->d_entry_col_S));
  if (bd->d_cone_S_offsets)    CUDA_CHECK(cudaFree(bd->d_cone_S_offsets));
  if (bd->d_flat_objval_S)     CUDA_CHECK(cudaFree(bd->d_flat_objval_S));
  if (bd->d_flat_spS_val)      CUDA_CHECK(cudaFree(bd->d_flat_spS_val));
  if (bd->d_S_row_ptr_flat)    CUDA_CHECK(cudaFree(bd->d_S_row_ptr_flat));
  if (bd->d_S_row_ptr_offsets) CUDA_CHECK(cudaFree(bd->d_S_row_ptr_offsets));
  free(bd->blk_idx_h);
  free(bd);
}

void free_solver_state(cardal_sdp_solver_state_t *state) {
  if (state == NULL)
    return;

  // ---- Phase 0: detach S->val aliases from batched-cone members ----
  // For each CUSTOM batch, member blocks' objective-union pattern .val
  // is aliased into a slice of bdata->d_flat_spS_val. NULL them out so
  // free_sparse_csr below does not free the pooled buffer per-slice.
  if (state->batch_leaders != NULL && state->block_low_rank_state != NULL) {
    for (int bi = 0; bi < state->n_batches; bi++) {
      block_low_rank_state_t *leader =
          state->block_low_rank_state[state->batch_leaders[bi]];
      if (leader == NULL || leader->kind != CONE_BATCH_KIND_CUSTOM ||
          leader->bdata == NULL)
        continue;
      cone_batch_data_t *bd = leader->bdata;
      for (int c = 0; c < leader->n_cones; c++) {
        int idx = bd->blk_idx_h ? bd->blk_idx_h[c] : -1;
        if (idx < 0 || idx >= state->n_blks) continue;
        block_low_rank_state_t *mem = state->block_low_rank_state[idx];
        if (mem && mem->objective_union_constraint_sparse_pattern)
          mem->objective_union_constraint_sparse_pattern->val = NULL;
      }
    }
  }

  // ---- Per-block state ----
  if (state->block_low_rank_state != NULL) {
    for (int b = 0; b < state->n_blks; b++) {
      block_low_rank_state_t *blk = state->block_low_rank_state[b];
      if (blk == NULL) continue;

      // cuSPARSE descriptors (aliased into global R/G/D — descriptors only)
      if (blk->matR)    cusparseDestroyDnMat(blk->matR);
      if (blk->matGrad) cusparseDestroyDnMat(blk->matGrad);
      if (blk->matD)    cusparseDestroyDnMat(blk->matD);
      if (blk->matSpA)  cusparseDestroySpMat(blk->matSpA);
      if (blk->matSpC)  cusparseDestroySpMat(blk->matSpC);
      if (blk->matSpS)  cusparseDestroySpMat(blk->matSpS);

      // Sparse patterns (union .val may have been NULL'd in Phase 0)
      free_sparse_csr(blk->constraint_sparse_pattern);
      free_sparse_csr(blk->objective_union_constraint_sparse_pattern);

      // Per-block device arrays
      if (blk->objective_val)                CUDA_CHECK(cudaFree(blk->objective_val));
      if (blk->compat_mapping)               CUDA_CHECK(cudaFree(blk->compat_mapping));
      if (blk->constraint_to_union_mapping)  CUDA_CHECK(cudaFree(blk->constraint_to_union_mapping));
      if (blk->psd_cone_rescaling)           CUDA_CHECK(cudaFree(blk->psd_cone_rescaling));
      if (blk->vec_psd_cone_rescaling)       CUDA_CHECK(cudaFree(blk->vec_psd_cone_rescaling));

      // cuSPARSE workspace buffers
      if (blk->sddmm_buffer_A) CUDA_CHECK(cudaFree(blk->sddmm_buffer_A));
      if (blk->sddmm_buffer_C) CUDA_CHECK(cudaFree(blk->sddmm_buffer_C));
      if (blk->spmm_buffer_S)  CUDA_CHECK(cudaFree(blk->spmm_buffer_S));

      // Batched-cone auxiliary data (owned by the batch leader).
      // Member cones share the pooled buffers via aliases which were
      // detached in Phase 0, so this final free reclaims them cleanly.
      free_bdata(blk->bdata);

      free(blk);
    }
    free(state->block_low_rank_state);
    state->block_low_rank_state = NULL;
  }
  if (state->batch_leaders) free(state->batch_leaders);

  // ---- State-level GPU arrays ----
  if (state->low_rank_solution)  CUDA_CHECK(cudaFree(state->low_rank_solution));
  if (state->low_rank_gradient)  CUDA_CHECK(cudaFree(state->low_rank_gradient));
  if (state->low_rank_direction) CUDA_CHECK(cudaFree(state->low_rank_direction));

  if (state->primal_solution)              CUDA_CHECK(cudaFree(state->primal_solution));
  if (state->primal_direct_solution_cross) CUDA_CHECK(cudaFree(state->primal_direct_solution_cross));
  if (state->primal_direct_double)         CUDA_CHECK(cudaFree(state->primal_direct_double));
  if (state->dual_solution)                CUDA_CHECK(cudaFree(state->dual_solution));
  if (state->primal_product)               CUDA_CHECK(cudaFree(state->primal_product));
  if (state->dual_product)                 CUDA_CHECK(cudaFree(state->dual_product));

  if (state->q0)              CUDA_CHECK(cudaFree(state->q0));
  if (state->q1)              CUDA_CHECK(cudaFree(state->q1));
  if (state->q2)              CUDA_CHECK(cudaFree(state->q2));
  if (state->q0_unscaled_buf) CUDA_CHECK(cudaFree(state->q0_unscaled_buf));

  if (state->constraint_rescaling)  CUDA_CHECK(cudaFree(state->constraint_rescaling));
  if (state->lp_variable_rescaling) CUDA_CHECK(cudaFree(state->lp_variable_rescaling));

  if (state->lp_objective_vector) CUDA_CHECK(cudaFree(state->lp_objective_vector));
  if (state->lp_slack_buffer)     CUDA_CHECK(cudaFree(state->lp_slack_buffer));
  if (state->lp_min_slack_buf)    CUDA_CHECK(cudaFree(state->lp_min_slack_buf));

  if (state->primal_spmv_buffer) CUDA_CHECK(cudaFree(state->primal_spmv_buffer));
  if (state->dual_spmv_buffer)   CUDA_CHECK(cudaFree(state->dual_spmv_buffer));

  if (state->d_p2_per_cone) CUDA_CHECK(cudaFree(state->d_p2_per_cone));
  if (state->d_step_scalars) CUDA_CHECK(cudaFree(state->d_step_scalars));

  if (state->d_lbfgs_S)       CUDA_CHECK(cudaFree(state->d_lbfgs_S));
  if (state->d_lbfgs_Y)       CUDA_CHECK(cudaFree(state->d_lbfgs_Y));
  if (state->d_lbfgs_R_old)   CUDA_CHECK(cudaFree(state->d_lbfgs_R_old));
  if (state->d_lbfgs_Grad_old)CUDA_CHECK(cudaFree(state->d_lbfgs_Grad_old));
  if (state->d_lbfgs_s)       CUDA_CHECK(cudaFree(state->d_lbfgs_s));
  if (state->d_lbfgs_y)       CUDA_CHECK(cudaFree(state->d_lbfgs_y));
  if (state->d_lbfgs_q)       CUDA_CHECK(cudaFree(state->d_lbfgs_q));
  if (state->d_lbfgs_z)       CUDA_CHECK(cudaFree(state->d_lbfgs_z));
  if (state->d_lbfgs_rho)     CUDA_CHECK(cudaFree(state->d_lbfgs_rho));
  if (state->d_lbfgs_alpha)   CUDA_CHECK(cudaFree(state->d_lbfgs_alpha));
  if (state->d_lbfgs_scratch) CUDA_CHECK(cudaFree(state->d_lbfgs_scratch));

  // ---- State-level cuSPARSE descriptors ----
  if (state->matA)           cusparseDestroySpMat(state->matA);
  if (state->matAt)          cusparseDestroySpMat(state->matAt);
  if (state->vecObj)         cusparseDestroySpVec(state->vecObj);
  if (state->vec_primal_sol) cusparseDestroyDnVec(state->vec_primal_sol);
  if (state->vec_dual_sol)   cusparseDestroyDnVec(state->vec_dual_sol);
  if (state->vec_q1)         cusparseDestroyDnVec(state->vec_q1);
  if (state->vec_primal_prod)cusparseDestroyDnVec(state->vec_primal_prod);
  if (state->vec_dual_prod)  cusparseDestroyDnVec(state->vec_dual_prod);

  // ---- Constraint matrices ----
  free_sparse_csr(state->constraint_matrix);
  free_sparse_csr(state->constraint_matrix_t);
  if (state->objective_vector_sparse) {
    if (state->objective_vector_sparse->pos)
      CUDA_CHECK(cudaFree(state->objective_vector_sparse->pos));
    if (state->objective_vector_sparse->val)
      CUDA_CHECK(cudaFree(state->objective_vector_sparse->val));
    free(state->objective_vector_sparse);
  }
  if (state->right_hand_side) CUDA_CHECK(cudaFree(state->right_hand_side));

  // ---- Cone stream pool ----
  for (int i = 0; i < state->cone_stream_pool_size; i++) {
    if (state->cone_stream_pool[i])
      CUDA_CHECK(cudaStreamDestroy(state->cone_stream_pool[i]));
  }

  // ---- cuSPARSE / cuBLAS handles ----
  if (state->sparse_handle) cusparseDestroy(state->sparse_handle);
  if (state->blas_handle)   cublasDestroy(state->blas_handle);

  // ---- Host arrays ----
  free(state->blk_dims);
  free(state->blk_ptr);
  free(state->col_mapping);
  free(state->rank_list);
  free(state->theoretical_max_rank_per_blk);
  free(state->h_p2_per_cone);

  // ---- The struct itself ----
  free(state);
}
