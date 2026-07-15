/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#define IS_DISTRIBUTED 1

#include "distribution_solver.h"
#include "distribution_utils.h"
#include "preconditioner.h"
#include "sdp_op.h"
#include "sdp_types.h"
#include "solver_state.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <string.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

static void preprocess_distributed_solver(cardal_sdp_solver_state_t *state);
static void sync_primal_solution(cardal_sdp_solver_state_t *state);

#define ALM_INNER_SOLVER SOLVE_INNER_LBFGS

#define SYNC_PRIMAL_SOLUTION(state) sync_primal_solution(state)
#define COMPUTE_GLOBAL_Q0_NORM_SQ(state, local_sq, global_sq)                  \
  do {                                                                         \
    if ((state)->grid_context->dims[0] > 1) {                                  \
      MPI_Allreduce(&(local_sq), &(global_sq), 1, MPI_DOUBLE, MPI_SUM,         \
                    (state)->grid_context->comm_row);                          \
    } else {                                                                   \
      (global_sq) = (local_sq);                                                \
    }                                                                          \
  } while (0)

#include "inner_solvers.cuh"
#include "outer_loop.cuh"
#include "solver_core_op.cuh"

sdp_result_t *
distributed_optimize(const compressed_sdp_problem_t *global_sdp_problem,
                     const cardal_parameters_t *params) {
  cardal_parameters_t sub_params = *params;
  select_valid_grid_size(params, global_sdp_problem, &sub_params);
  grid_context_t grid_context = initialize_parallel_context(
      sub_params.grid_size.row_dims, sub_params.grid_size.rank_dims,
      sub_params.grid_size.cone_dims);
  sub_params.verbose =
      (grid_context.rank_global == 0) ? params->verbose : 0;

  void *bcast_buffer = NULL;
  size_t total_buf_size = 0;

  if (grid_context.rank_global == 0) {
    serialize_compressed_sdp(global_sdp_problem, &bcast_buffer,
                             &total_buf_size);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  big_bcast_bytes(&bcast_buffer, &total_buf_size, 0, grid_context.comm_global);

  compressed_sdp_problem_t *synced_global_prob;
  if (grid_context.rank_global == 0) {
    synced_global_prob = (compressed_sdp_problem_t *)global_sdp_problem;
  } else {
    synced_global_prob = deserialize_compressed_sdp(bcast_buffer);
  }

  if (bcast_buffer)
    free(bcast_buffer);

  shuffle_type_t shuffle_mode = params->shuffle_mode;
  int *global_perm = permute_global_problem_constraints(synced_global_prob,
                                                        shuffle_mode, 128, 42);

  rescale_info_t *global_rescale_info =
      rescale_problem(&sub_params, synced_global_prob);
  const compressed_sdp_problem_t *solve_global_prob =
      global_rescale_info ? global_rescale_info->scaled_problem
                          : synced_global_prob;

  compressed_sdp_problem_t *sdp_problem =
      partition_problem(solve_global_prob, &grid_context);
  rescale_info_t *local_rescale_info = partition_rescale_info(
      global_rescale_info, solve_global_prob, &grid_context);

  int n_blks_global = solve_global_prob->n_blks;
  int *global_max_rank_per_blk =
      (int *)safe_malloc(n_blks_global * sizeof(int));
  compute_per_block_max_rank(solve_global_prob, params->max_rank,
                             global_max_rank_per_blk);
  int n_blks_local = sdp_problem->n_blks;
  int *local_max_rank_per_blk = NULL;
  if (n_blks_local > 0) {
    local_max_rank_per_blk =
        (int *)safe_malloc(n_blks_local * sizeof(int));
    int gb = 0;
    for (int lb = 0; lb < n_blks_local; lb++) {
      while (gb < n_blks_global &&
             solve_global_prob->blk_ptr[gb] != sdp_problem->blk_ptr[lb])
        gb++;
      local_max_rank_per_blk[lb] = global_max_rank_per_blk[gb];
      gb++;
    }
  }
  free(global_max_rank_per_blk);

  if (params->initial_rank <= 0)
    sub_params.initial_rank =
        compute_initial_rank(solve_global_prob->num_constraints);
  if (params->initial_penalty_coef <= 0.0) {
    sub_params.initial_penalty_coef =
        compute_initial_penalty_coef(solve_global_prob);
  }

  int global_rank = sub_params.initial_rank;
  int P_rank = grid_context.dims[1];
  int my_rank = grid_context.coords[1];

  int base_rank = global_rank / P_rank;
  int rem_rank = global_rank % P_rank;
  sub_params.initial_rank = base_rank + (my_rank < rem_rank ? 1 : 0);

  cardal_sdp_solver_state_t *state = initialize_solver_state(
      sdp_problem, &sub_params, local_max_rank_per_blk, local_rescale_info);

  state->grid_context = &grid_context;
  preprocess_distributed_solver(state);

  g_log_verbose =
      (grid_context.rank_global == 0) ? params->verbose : 0;

  print_cardal_banner();
  int world_size_for_print = grid_context.dims[0] * grid_context.dims[1] *
                             grid_context.dims[2];
  print_runtime_environment_section(2, /*is_distributed=*/1,
                                    world_size_for_print, grid_context.dims[0],
                                    grid_context.dims[1], grid_context.dims[2]);
  if (params->instance_label != NULL) {
    print_subtitle("Input");
    print_kv_str("Source", params->instance_label);
  }
  print_parameters_section(&sub_params, 2);
  print_problem_statistics_section(synced_global_prob, 2);

  print_per_rank_workload(&grid_context, sdp_problem,
                          synced_global_prob->num_constraints,
                          global_rank);

  {
    long long my_nnz_union = 0, my_nnz_A = 0;
    for (int b = 0; b < state->n_blks; b++) {
      block_low_rank_state_t *blk = state->block_low_rank_state[b];
      if (blk->objective_union_constraint_sparse_pattern)
        my_nnz_union +=
            blk->objective_union_constraint_sparse_pattern->num_nonzeros;
      if (blk->constraint_sparse_pattern)
        my_nnz_A += blk->constraint_sparse_pattern->num_nonzeros;
    }

    long long my_hypoth_local = 0;
    if (sdp_problem->constraint_matrix &&
        sdp_problem->n_active_vars > 0) {
      char *col_used =
          (char *)safe_calloc(sdp_problem->n_active_vars, 1);
      const sparse_csr_matrix_t *lc = sdp_problem->constraint_matrix;
      for (int p = 0; p < lc->num_nonzeros; p++) {
        int c = lc->col_ind[p];
        if (c >= 0 && c < sdp_problem->n_active_vars) col_used[c] = 1;
      }
      for (int c = 0; c < sdp_problem->n_active_vars; c++)
        if (col_used[c]) my_hypoth_local++;
      free(col_used);
    }
    long long max_nnz_union = 0, sum_nnz_union = 0;
    long long max_nnz_A = 0, sum_nnz_A = 0;
    long long max_hypoth = 0, sum_hypoth = 0;
    MPI_Reduce(&my_nnz_union, &max_nnz_union, 1, MPI_LONG_LONG, MPI_MAX, 0,
               grid_context.comm_global);
    MPI_Reduce(&my_nnz_union, &sum_nnz_union, 1, MPI_LONG_LONG, MPI_SUM, 0,
               grid_context.comm_global);
    MPI_Reduce(&my_nnz_A, &max_nnz_A, 1, MPI_LONG_LONG, MPI_MAX, 0,
               grid_context.comm_global);
    MPI_Reduce(&my_nnz_A, &sum_nnz_A, 1, MPI_LONG_LONG, MPI_SUM, 0,
               grid_context.comm_global);
    MPI_Reduce(&my_hypoth_local, &max_hypoth, 1, MPI_LONG_LONG, MPI_MAX, 0,
               grid_context.comm_global);
    MPI_Reduce(&my_hypoth_local, &sum_hypoth, 1, MPI_LONG_LONG, MPI_SUM, 0,
               grid_context.comm_global);
    if (grid_context.rank_global == 0) {
      printf("  [Per-rank cone pattern]\n");
      printf("    current  nnz_union: max=%lld sum=%lld (per-rank=%lld)\n",
             max_nnz_union, sum_nnz_union, my_nnz_union);
      printf("    current  nnz_A    : max=%lld sum=%lld\n",
             max_nnz_A, sum_nnz_A);
      printf("    hypothetical local-touched cols: max=%lld sum=%lld\n",
             max_hypoth, sum_hypoth);
      fflush(stdout);
    }
  }

  print_subtitle("Optimization");

  // Warm Up MPI
  CHECK_DUAL_INFEASIBILITY(state);
  MPI_Barrier(state->grid_context->comm_global);

  clock_t start_time = clock();

  run_alm_outer_loop(state, &sub_params, start_time);

  clock_t end_time = clock();
  state->cumulative_time_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;


  sdp_result_t *result = create_result_from_state(state, sdp_problem);
  result->rescaling_time_sec =
      global_rescale_info ? global_rescale_info->rescaling_time_sec : 0.0;
  unscale_result(local_rescale_info, state, result);
  if (grid_context.dims[0] > 1) {
    int local_nnz = sdp_problem->constraint_matrix->num_nonzeros;
    int global_nnz = local_nnz;
    MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_INT, MPI_SUM,
                  grid_context.comm_row);
    result->num_nonzeros = global_nnz;
  }
  if (grid_context.dims[1] > 1) {
    int local_rank = state->total_rank;
    int global_rank_total = local_rank;
    MPI_Allreduce(&local_rank, &global_rank_total, 1, MPI_INT, MPI_SUM,
                  grid_context.comm_rank);
    result->rank = global_rank_total;
  }

  int total_m_global = synced_global_prob->num_constraints;
  free(result->dual_solution);
  result->dual_solution = NULL;
  result->num_constraints = total_m_global;

  MPI_Comm collector = MPI_COMM_NULL;
  int color = (grid_context.coords[1] == 0 && grid_context.coords[2] == 0)
                  ? 0
                  : MPI_UNDEFINED;
  MPI_Comm_split(grid_context.comm_global, color, grid_context.coords[0],
                 &collector);

  if (collector != MPI_COMM_NULL) {
    double *h_local_dual =
        (double *)safe_malloc(state->num_constraints * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_local_dual, state->dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToHost));

    if (local_rescale_info != NULL) {
      double tau_c = local_rescale_info->objective_vector_rescaling;
      double inv_tau_c = (tau_c > 0.0) ? 1.0 / tau_c : 1.0;
      if (local_rescale_info->constraint_rescaling != NULL) {
        for (int i = 0; i < state->num_constraints; i++) {
          double D = local_rescale_info->constraint_rescaling[i];
          double factor = (D > 0.0) ? (inv_tau_c / D) : inv_tau_c;
          h_local_dual[i] *= factor;
        }
      } else if (inv_tau_c != 1.0) {
        for (int i = 0; i < state->num_constraints; i++)
          h_local_dual[i] *= inv_tau_c;
      }
    }

    int P_row = grid_context.dims[0];
    int *recvcounts = NULL;
    int *displs = NULL;
    double *h_gathered = NULL;

    int my_local_m = state->num_constraints;
    int *all_local_m = (int *)safe_malloc(P_row * sizeof(int));
    MPI_Allgather(&my_local_m, 1, MPI_INT, all_local_m, 1, MPI_INT, collector);

    if (grid_context.coords[0] == 0) {
      recvcounts = (int *)safe_malloc(P_row * sizeof(int));
      displs = (int *)safe_malloc(P_row * sizeof(int));
      int cur = 0;
      for (int r = 0; r < P_row; r++) {
        recvcounts[r] = all_local_m[r];
        displs[r] = cur;
        cur += recvcounts[r];
      }
      h_gathered = (double *)safe_malloc(total_m_global * sizeof(double));
    }
    free(all_local_m);

    MPI_Gatherv(h_local_dual, state->num_constraints, MPI_DOUBLE, h_gathered,
                recvcounts, displs, MPI_DOUBLE, 0, collector);

    if (grid_context.coords[0] == 0) {
      double *h_unpermuted =
          (double *)safe_malloc(total_m_global * sizeof(double));
      unpermute_dual_solution(total_m_global, h_gathered, h_unpermuted,
                              global_perm);
      result->dual_solution = h_unpermuted;
      free(h_gathered);
      free(recvcounts);
      free(displs);
    }
    free(h_local_dual);
    MPI_Comm_free(&collector);
  }

  if (global_perm)
    free(global_perm);

  gather_sdp_result(result, state);

  print_optimization_footer(result, params->summary_file_path, 1);

  int rank_for_cleanup = grid_context.rank_global;
  free_solver_state(state);
  if (sdp_problem != NULL)
    free_compressed_sdp(sdp_problem);
  if (rank_for_cleanup != 0 && synced_global_prob != NULL)
    free_compressed_sdp(synced_global_prob);

  if (local_rescale_info)
    free_rescale_info(local_rescale_info);
  if (global_rescale_info)
    free_rescale_info(global_rescale_info);

  cleanup_parallel_context(&grid_context);
  return result;
}

static void preprocess_distributed_solver(cardal_sdp_solver_state_t *state) {
  double local_l1_b = state->right_hand_side_norm;
  double local_linf_b = state->right_hand_side_linf_norm;
  double global_l1_b = 0.0;
  double global_linf_b = 0.0;

  MPI_Allreduce(&local_l1_b, &global_l1_b, 1, MPI_DOUBLE, MPI_SUM,
                state->grid_context->comm_row);
  MPI_Allreduce(&local_linf_b, &global_linf_b, 1, MPI_DOUBLE, MPI_MAX,
                state->grid_context->comm_row);

  state->right_hand_side_norm = global_l1_b;
  state->right_hand_side_linf_norm = global_linf_b;

  if (state->grid_context->dims[2] > 1) {
    double local_l1_c = state->objective_vector_norm;
    double local_linf_c = state->objective_vector_linf_norm;
    double global_l1_c = 0.0, global_linf_c = 0.0;
    MPI_Allreduce(&local_l1_c, &global_l1_c, 1, MPI_DOUBLE, MPI_SUM,
                  state->grid_context->comm_cone);
    MPI_Allreduce(&local_linf_c, &global_linf_c, 1, MPI_DOUBLE, MPI_MAX,
                  state->grid_context->comm_cone);
    state->objective_vector_norm = global_l1_c;
    state->objective_vector_linf_norm = global_linf_c;
  }

  if (state->grid_context->coords[0] != 0) {
    for (int b = 0; b < state->n_blks; b++) {
      block_low_rank_state_t *blk = state->block_low_rank_state[b];
      int nnz_Union =
          blk->objective_union_constraint_sparse_pattern->num_nonzeros;
      if (nnz_Union > 0) {
        CUDA_CHECK(
            cudaMemset(blk->objective_val, 0, nnz_Union * sizeof(double)));
      }
    }

    for (int bi = 0; bi < state->n_batches; bi++) {
      block_low_rank_state_t *leader =
          state->block_low_rank_state[state->batch_leaders[bi]];
      if (leader->kind != CONE_BATCH_KIND_CUSTOM || leader->bdata == NULL)
        continue;
      if (leader->bdata->total_nnz_S > 0) {
        CUDA_CHECK(cudaMemset(leader->bdata->d_flat_objval_S, 0,
                              leader->bdata->total_nnz_S * sizeof(double)));
      }
    }
    if (state->lp_dim > 0 && state->lp_objective_vector != NULL) {
      CUDA_CHECK(cudaMemset(state->lp_objective_vector, 0,
                            state->lp_dim * sizeof(double)));
    }
  }
}

static void sync_primal_solution(cardal_sdp_solver_state_t *state) {
  if (state->grid_context->dims[0] > 1) {
    NCCL_CHECK(ncclAllReduce(state->low_rank_solution, state->low_rank_solution,
                             state->length_low_rank_solution, ncclDouble,
                             ncclAvg, state->grid_context->nccl_row, 0));
  }
}
