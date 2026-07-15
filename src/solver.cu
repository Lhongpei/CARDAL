/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "preconditioner.h"
#include "sdp_op.h"
#include "sdp_types.h"
#include "solver.h"
#include "solver_state.h"
#include "utils.h"
#include <stdio.h>

#define ALM_INNER_SOLVER SOLVE_INNER_LBFGS

#include "outer_loop.cuh"

sdp_result_t *optimize(const compressed_sdp_problem_t *sdp_problem,
                       const cardal_parameters_t *params) {
  cardal_parameters_t sub_params = *params;
  g_log_verbose = params->verbose;

  print_cardal_banner();
  print_runtime_environment_section(2, /*is_distributed=*/0, 1, 1, 1, 1);
  if (params->instance_label != NULL) {
    print_subtitle("Input");
    print_kv_str("Source", params->instance_label);
  }
  print_parameters_section(&sub_params, 2);
  print_problem_statistics_section(sdp_problem, 2);
  print_subtitle("Optimization");

  rescale_info_t *rescale_info = rescale_problem(&sub_params, sdp_problem);
  const compressed_sdp_problem_t *solve_problem =
      rescale_info ? rescale_info->scaled_problem : sdp_problem;
  if (rescale_info && params->verbose > 2) {
    printf("  [rescale] tau_c (obj_rescale)  = %.6e\n",
           rescale_info->objective_vector_rescaling);
    printf("  [rescale] tau_b (rhs_rescale)  = %.6e\n",
           rescale_info->right_hand_side_rescaling);
    printf("  [rescale] ||C||_unscaled       = %.6e\n",
           rescale_info->unscaled_objective_vector_norm);
    printf("  [rescale] ||b||_unscaled       = %.6e\n",
           rescale_info->unscaled_right_hand_side_norm);
    printf("  [rescale] eps_machine          = %.3e\n", 2.220446e-16);
    printf("  [rescale] tau_c / eps_machine  = %.3e %s\n",
           rescale_info->objective_vector_rescaling / 2.220446e-16,
           (rescale_info->objective_vector_rescaling < 1e-12)
               ? "  <-- WARNING: near machine precision"
               : "");
  }

  int *max_rank_per_blk =
      (int *)safe_malloc(solve_problem->n_blks * sizeof(int));
  compute_per_block_max_rank(solve_problem, params->max_rank, max_rank_per_blk);
  if (params->initial_rank <= 0)
    sub_params.initial_rank =
        compute_initial_rank(solve_problem->num_constraints);
  if (params->initial_penalty_coef <= 0.0)
    sub_params.initial_penalty_coef = compute_initial_penalty_coef(solve_problem);

  cardal_sdp_solver_state_t *state = initialize_solver_state(
      solve_problem, &sub_params, max_rank_per_blk, rescale_info);

  clock_t start_time = clock();

  run_alm_outer_loop(state, params, start_time);

  clock_t end_time = clock();
  state->cumulative_time_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  sdp_result_t *result = create_result_from_state(state, solve_problem);
  result->rescaling_time_sec =
      rescale_info ? rescale_info->rescaling_time_sec : 0.0;

  unscale_result(rescale_info, state, result);

  // ---------- Post-solve log ----------
  print_optimization_footer(result, params->summary_file_path, 1);

  free_solver_state(state);
  if (rescale_info)
    free_rescale_info(rescale_info);
  return result;
}