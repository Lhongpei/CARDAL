/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#ifndef OUTER_LOOP_CUH
#define OUTER_LOOP_CUH

#include "inner_solvers.cuh"
#include "sdp_op.h"
#include "sdp_types.h"
#include "solver_core_op.cuh"
#include "solver_state.h"
#include "utils.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifndef SYNC_PRIMAL_SOLUTION
#define SYNC_PRIMAL_SOLUTION(state)                                            \
  do {                                                                         \
  } while (0)
#endif

#ifndef COMPUTE_GLOBAL_Q0_NORM_SQ
#define COMPUTE_GLOBAL_Q0_NORM_SQ(state, local_sq, global_sq)                  \
  do {                                                                         \
    (global_sq) = (local_sq);                                                  \
  } while (0)
#endif

#ifndef ALM_INNER_SOLVER
#define ALM_INNER_SOLVER SOLVE_INNER_LBFGS
#endif

/* Cooperative-cancellation flag lives in src/cardal_c_api.c as
 * `volatile sig_atomic_t`; we read it via the accessor to avoid dragging
 * <signal.h> into every CUDA TU. Set from a signal handler (typically the
 * Python binding's SIGINT trap). */
extern "C" int cardal_cancel_requested(void);

static inline termination_reason_t
check_termination(const cardal_sdp_solver_state_t *state,
                  const cardal_parameters_t *params) {
  if (cardal_cancel_requested()) {
    return TERMINATION_REASON_USER_INTERRUPT;
  }
  if (state->relative_primal_residual <
          params->termination_criteria.eps_primal_relative &&
      state->relative_dual_residual <
          params->termination_criteria.eps_dual_relative &&
      state->relative_objective_gap <
          params->termination_criteria.eps_optimal_relative) {
    return TERMINATION_REASON_OPTIMAL;
  }
  if (params->termination_criteria.time_sec_limit > 0.0 &&
      state->cumulative_time_sec >=
          params->termination_criteria.time_sec_limit) {
    return TERMINATION_REASON_TIME_LIMIT;
  }
  if (state->num_outer_iteration >=
      params->termination_criteria.iteration_limit) {
    return TERMINATION_REASON_ITERATION_LIMIT;
  }
  return TERMINATION_REASON_UNSPECIFIED;
}

static inline double compute_global_q0_norm(cardal_sdp_solver_state_t *state) {
  double local_norm = 0.0;
  if (state->constraint_rescaling != NULL) {
    local_norm = compute_unscaled_q0_l2_norm(state);
  } else {
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, state->num_constraints,
                             state->q0, 1, &local_norm));
  }
  double local_sq = local_norm * local_norm;
  double global_sq = local_sq;
  COMPUTE_GLOBAL_Q0_NORM_SQ(state, local_sq, global_sq);
  return sqrt(global_sq);
}

static inline void run_alm_outer_loop(cardal_sdp_solver_state_t *state,
                                      const cardal_parameters_t *params,
                                      clock_t start_time) {
  double denom_b = 1.0 + state->unscaled_right_hand_side_norm;

  const double primal_stall_eps = 1e-4;
  const double outer_stall_eps = 5e-2;
  const int max_stall_uncv = 3;
  const int max_middle_iters = 3;

  if (state->verbose >= 2)
    print_header();

  state->outer_stall_count = 0;
  state->last_outer_primal_norm = 0.0;
  state->last_outer_obj_gap = 0.0;
  state->outer_iters_since_augment_check = 0;
  state->lbfgs_maxiter_streak = 0;

  const int gap_stall_augment_iters = 5;
  const double augment_primal_threshold = 1e-3;
  const double dual_step_acceleration = 1.618;
  const double dual_step_acceleration_threshold = 1e-5;
  const double rho_jump_suppression_threshold = 1e-6;
  const int curvature_check_throttle_iters = 1;
  const int curvature_check_gap_stall_iters = 3;
  // After this many consecutive gate-pass outer iters with no progress,
  // force a rho jump and trigger augmentation.
  const int gate_pass_force_iters = 10;

  state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
  while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED) {

    SYNC_PRIMAL_SOLUTION(state);

    COMPUTE_PRIMAL_RESIDUAL(state);
    COMPUTE_GRADIENT(state);

    COMPUTE_OBJECTIVE_GAP(state);
    state->absolute_primal_residual = compute_global_q0_norm(state);
    state->relative_primal_residual =
        state->absolute_primal_residual / denom_b;
    double current_primal = state->relative_primal_residual;
    double current_gap = state->relative_objective_gap;
    double pre_gap_improvement =
        (state->last_outer_obj_gap > 1e-30)
            ? (state->last_outer_obj_gap - current_gap) /
                  state->last_outer_obj_gap
            : 1.0;
    if (pre_gap_improvement < outer_stall_eps)
      state->gap_stall_count++;
    else
      state->gap_stall_count = 0;

    int augment_now =
        (state->gap_stall_count >= gap_stall_augment_iters) &&
        (current_primal < augment_primal_threshold);
    if (augment_now) {
      if (state->verbose >= 3)
        printf("[augment] gap_stall=%d p_res=%.2e -> ALORA\n",
               state->gap_stall_count, current_primal);
      update_dual_slack_S(state);
      CHECK_DUAL_INFEASIBILITY_AND_AUGMENT(state);
      state->gap_stall_count = 0;
      state->last_outer_obj_gap = current_gap;
      state->last_outer_primal_norm = current_primal;
      state->cumulative_time_sec =
          (double)(clock() - start_time) / CLOCKS_PER_SEC;
      state->num_outer_iteration++;
      state->termination_reason = check_termination(state, params);
      continue;  // Skip middle loop, dual update, and rho update.
    }

    double prev_middle_primal = compute_global_q0_norm(state);
    int inner_iters_total_this_round = 0;

    for (int middle_iter = 0; middle_iter < max_middle_iters; middle_iter++) {
      double grad_norm = 0.0;
      CUBLAS_CHECK(cublasDnrm2(state->blas_handle,
                               state->length_low_rank_solution,
                               state->low_rank_gradient, 1, &grad_norm));
      if (isnan(grad_norm) || isinf(grad_norm)) {
        if (state->verbose >= 3)
          printf("\n[Warning] NaN/Inf detected in gradient! Aborting "
                 "middle loop.\n");
        break;
      }

      double scale_factor = 1.0 + state->objective_vector_linf_norm;
      double rel_primal_for_tol = state->relative_primal_residual;
      if (rel_primal_for_tol <= 1e-30 || isinf(rel_primal_for_tol))
        rel_primal_for_tol = 1.0;
      double rho_tol = (0.1 / state->penalty_coef) * scale_factor;
      double primal_tol =
          state->inner_eta * rel_primal_for_tol * scale_factor;
      double tol = (rho_tol > primal_tol) ? rho_tol : primal_tol;
      double tol_floor =
          0.5 * params->termination_criteria.eps_primal_relative *
          scale_factor;
      double tol_cap = 0.5 * scale_factor;
      double min_hardware_tol = 1e-7 * scale_factor;
      if (tol_floor < min_hardware_tol)
        tol_floor = min_hardware_tol;
      if (tol < tol_floor)
        tol = tol_floor;
      if (tol > tol_cap)
        tol = tol_cap;

      const double negative_curvature_threshold_factor = 1e-4;
      const int max_curvature_escape_steps = 3;
      int curvature_detection_disabled = 0;
#ifdef IS_DISTRIBUTED
      if (state->grid_context != NULL && state->grid_context->dims[2] > 1)
        curvature_detection_disabled = 1;
#endif
      int inner_iters = ALM_INNER_SOLVER(state, tol);
      state->num_inner_iteration += inner_iters;
      inner_iters_total_this_round += inner_iters;

      int stationary = (state->lbfgs_exit_reason == LBFGS_GRAD_CONVERGED) ||
                       (state->lbfgs_exit_reason == LBFGS_TAU_STALL);
      int gap_stalled =
          (state->gap_stall_count >= curvature_check_gap_stall_iters);
      int iters_since_curvature_check =
          (state->curvature_last_check_iter < 0)
              ? INT_MAX
              : (state->num_outer_iteration -
                 state->curvature_last_check_iter);
      int throttle_ok =
          (iters_since_curvature_check >= curvature_check_throttle_iters);
      int should_detect_curvature = stationary && gap_stalled && throttle_ok;
#ifdef IS_DISTRIBUTED
      if (!curvature_detection_disabled && state->grid_context != NULL) {
        int local_detect = should_detect_curvature ? 1 : 0;
        int global_detect = 0;
        MPI_Allreduce(&local_detect, &global_detect, 1, MPI_INT, MPI_MAX,
                      state->grid_context->comm_global);
        should_detect_curvature = global_detect;
      }
#endif
      if (!curvature_detection_disabled && should_detect_curvature) {
        state->curvature_last_check_iter = state->num_outer_iteration;
        state->gap_stall_count = 0;
        {
          const double post_esc_tol_factor = 0.01;
          double post_esc_tol = tol * post_esc_tol_factor;
          if (post_esc_tol < min_hardware_tol) post_esc_tol = min_hardware_tol;

          for (int esc = 0; esc < max_curvature_escape_steps; esc++) {
            int n_esc = DETECT_NEGATIVE_CURVATURE_AND_ESCAPE(
                state, negative_curvature_threshold_factor);
            int global_n_esc = n_esc;
#ifdef IS_DISTRIBUTED
            if (state->grid_context != NULL) {
              MPI_Allreduce(&n_esc, &global_n_esc, 1, MPI_INT, MPI_SUM,
                            state->grid_context->comm_global);
            }
#endif
            if (global_n_esc == 0) break;

            COMPUTE_PRIMAL_RESIDUAL(state);
            COMPUTE_GRADIENT(state);

            int n2 = ALM_INNER_SOLVER(state, post_esc_tol);
            state->num_inner_iteration += n2;
            inner_iters_total_this_round += n2;
          }
        }
      }

      double dual_step = state->penalty_coef;
      if (state->relative_primal_residual <
          dual_step_acceleration_threshold) {
        dual_step *= dual_step_acceleration;
      }
      CUBLAS_CHECK(cublasDaxpy(state->blas_handle, state->num_constraints,
                               &dual_step, state->q0, 1,
                               state->dual_solution, 1));

      COMPUTE_PRIMAL_RESIDUAL(state);
      COMPUTE_GRADIENT(state);

      double cur_middle_primal = compute_global_q0_norm(state);
      double mid_improvement = (prev_middle_primal > 1e-30)
                                   ? (prev_middle_primal - cur_middle_primal) /
                                         prev_middle_primal
                                   : 1.0;
      prev_middle_primal = cur_middle_primal;
      if (mid_improvement < primal_stall_eps)
        break;
    }

    double q0_scaled_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, state->num_constraints,
                             state->q0, 1, &q0_scaled_norm));
    {
      double local_sq = q0_scaled_norm * q0_scaled_norm;
      double global_sq = local_sq;
      COMPUTE_GLOBAL_Q0_NORM_SQ(state, local_sq, global_sq);
      q0_scaled_norm = sqrt(global_sq);
    }
    double denom_b_scaled = 1.0 + state->right_hand_side_norm;
    double rel_q0_for_gate = (denom_b_scaled > 0.0)
                                 ? q0_scaled_norm / denom_b_scaled
                                 : 0.0;
    int eta_gate_passed = (rel_q0_for_gate <= state->lancelot_eta);
    double eta_floor =
        0.5 * params->termination_criteria.eps_primal_relative;

    int allow_rho_jump = !eta_gate_passed;
    if (allow_rho_jump &&
        state->relative_primal_residual < rho_jump_suppression_threshold) {
      allow_rho_jump = 0;
    }
    // BM rank deficiency can keep the primal residual below eta while the gap
    // stalls. Force a rho jump and augmentation after repeated gate passes.
    int force_combo = 0;
    if (!allow_rho_jump) {
      state->consecutive_gate_pass++;
      if (state->consecutive_gate_pass >= gate_pass_force_iters) {
        force_combo = 1;
        state->consecutive_gate_pass = 0;
      }
    } else {
      state->consecutive_gate_pass = 0;
    }
    if (force_combo || allow_rho_jump) {
      const double rho_jump = 3.33;
      state->penalty_coef *= rho_jump;
      if (params->max_penalty_coef > 0.0 &&
          state->penalty_coef > params->max_penalty_coef) {
        state->penalty_coef = params->max_penalty_coef;
      }
      if (force_combo) {
        state->force_augment_this_iter = 1;
      } else {
        state->gate_fail_streak++;
      }
    } else {
      double new_eta = 0.5 * state->lancelot_eta;
      if (new_eta < rel_q0_for_gate)
        new_eta = rel_q0_for_gate;
      if (new_eta < eta_floor)
        new_eta = eta_floor;
      state->lancelot_eta = new_eta;
      state->gate_fail_streak = 0;
    }

    COMPUTE_OBJECTIVE_GAP(state);

    state->absolute_primal_residual = compute_global_q0_norm(state);
    state->relative_primal_residual = state->absolute_primal_residual / denom_b;

    double cur_primal = state->relative_primal_residual;
    double cur_gap = state->relative_objective_gap;
    double primal_improvement = (state->last_outer_primal_norm > 1e-30)
                                    ? (state->last_outer_primal_norm -
                                       cur_primal) /
                                          state->last_outer_primal_norm
                                    : 1.0;
    double gap_improvement = (state->last_outer_obj_gap > 1e-30)
                                 ? (state->last_outer_obj_gap - cur_gap) /
                                       state->last_outer_obj_gap
                                 : 1.0;
    int primal_stalled = (primal_improvement < outer_stall_eps);
    int gap_stalled = (gap_improvement < outer_stall_eps);

    state->last_outer_primal_norm = cur_primal;
    state->last_outer_obj_gap = cur_gap;
    state->outer_iters_since_augment_check++;

    int should_augment = 0;
    if (primal_stalled || gap_stalled) {
      if (state->lbfgs_exit_reason == LBFGS_GRAD_CONVERGED) {
        should_augment = 1;
        state->outer_stall_count = 0;
      } else {
        state->outer_stall_count++;
        if (state->outer_stall_count >= max_stall_uncv) {
          should_augment = 1;
          state->outer_stall_count = 0;
        }
      }
    } else {
      state->outer_stall_count = 0;
    }

    const int max_lbfgs_maxiter_streak = 3;
    if (state->lbfgs_exit_reason == LBFGS_MAX_ITERS) {
      state->lbfgs_maxiter_streak++;
      if (!should_augment &&
          state->lbfgs_maxiter_streak >= max_lbfgs_maxiter_streak) {
        should_augment = 1;
        state->lbfgs_maxiter_streak = 0;
      }
    } else {
      state->lbfgs_maxiter_streak = 0;
    }

    const int augment_check_max_gap = 10;
    if (!should_augment &&
        state->outer_iters_since_augment_check >= augment_check_max_gap) {
      should_augment = 1;
    }

    const int augment_cooldown_iters = 5;
    if (should_augment &&
        state->outer_iters_since_augment_check < augment_cooldown_iters) {
      should_augment = 0;
    }

    // BM-rank-deficiency safety: LANCELOT block set this when too many
    // consecutive gate-pass iters detected no real progress. Force augment
    // bypassing the cooldown (we explicitly want both rho jump and rank
    // augmentation to trigger in lockstep).
    if (state->force_augment_this_iter) {
      should_augment = 1;
      state->force_augment_this_iter = 0;
    }

    if (should_augment)
      state->outer_iters_since_augment_check = 0;

    state->dual_residual_evaluated = 0;
    state->absolute_dual_residual = 1e30;
    state->relative_dual_residual = 1e30;

    if (should_augment) {
      if (state->verbose >= 3)
        printf("\n[Augment] primal_impr=%.2e gap_impr=%.2e lbfgs_reason=%d "
               "(stalled=%d/%d, periodic=%d) - probing dual slack\n",
               primal_improvement, gap_improvement, state->lbfgs_exit_reason,
               primal_stalled, gap_stalled,
               state->outer_iters_since_augment_check == 0);
      update_dual_slack_S(state);
      CHECK_DUAL_INFEASIBILITY_AND_AUGMENT(state);
      state->dual_residual_evaluated = 1;
    }

    if (!state->dual_residual_evaluated &&
        state->relative_primal_residual <
            params->termination_criteria.eps_primal_relative &&
        state->relative_objective_gap <
            params->termination_criteria.eps_optimal_relative) {
      update_dual_slack_S(state);
      CHECK_DUAL_INFEASIBILITY(state);
      state->dual_residual_evaluated = 1;
    }

    {
      double primal_ratio = 1.0;
      if (state->prev_outer_primal_for_eta > 1e-30 && cur_primal > 0.0) {
        primal_ratio = cur_primal / state->prev_outer_primal_for_eta;
      }
      double eta_target = primal_ratio;
      if (state->lbfgs_exit_reason == LBFGS_MAX_ITERS) {
        double relax_target = state->inner_eta * 2.0;
        if (eta_target < relax_target)
          eta_target = relax_target;
      }
      state->inner_eta = 0.5 * state->inner_eta + 0.5 * eta_target;
      const double eta_min = 1e-3;
      const double eta_max = 1.0;
      if (state->inner_eta < eta_min)
        state->inner_eta = eta_min;
      if (state->inner_eta > eta_max)
        state->inner_eta = eta_max;
      state->prev_outer_primal_for_eta = cur_primal;
    }

    state->cumulative_time_sec =
        (double)(clock() - start_time) / CLOCKS_PER_SEC;

    if (state->verbose >= 2)
      print_log_entry(state, inner_iters_total_this_round);

    state->num_outer_iteration++;

    state->termination_reason = check_termination(state, params);
    if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
      break;
  }
}
#endif // OUTER_LOOP_CUH
