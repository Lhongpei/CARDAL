/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#ifndef INNER_SOLVERS_CUH
#define INNER_SOLVERS_CUH

#include "sdp_op.h"
#include "sdp_types.h"
#include "solver_core_op.cuh"
#include "solver_state.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>

#ifdef IS_DISTRIBUTED
#include <mpi.h>

#define GLOBAL_DOT(state, local_val, global_val) do { \
    double _tmp_val = (local_val); \
    if ((state)->grid_context->dims[1] > 1) { \
        double _g = 0.0; \
        MPI_Allreduce(&_tmp_val, &_g, 1, MPI_DOUBLE, MPI_SUM, (state)->grid_context->comm_rank); \
        _tmp_val = _g; \
    } \
    if ((state)->grid_context->dims[2] > 1) { \
        double _g = 0.0; \
        MPI_Allreduce(&_tmp_val, &_g, 1, MPI_DOUBLE, MPI_SUM, (state)->grid_context->comm_cone); \
        _tmp_val = _g; \
    } \
    (global_val) = _tmp_val; \
} while(0)

#define GLOBAL_NORM(state, local_norm, global_norm) do { \
    double _sq = (local_norm) * (local_norm); \
    if ((state)->grid_context->dims[1] > 1) { \
        double _g = 0.0; \
        MPI_Allreduce(&_sq, &_g, 1, MPI_DOUBLE, MPI_SUM, (state)->grid_context->comm_rank); \
        _sq = _g; \
    } \
    if ((state)->grid_context->dims[2] > 1) { \
        double _g = 0.0; \
        MPI_Allreduce(&_sq, &_g, 1, MPI_DOUBLE, MPI_SUM, (state)->grid_context->comm_cone); \
        _sq = _g; \
    } \
    (global_norm) = sqrt(_sq); \
} while(0)
#else
#define GLOBAL_DOT(state, local_val, global_val) (global_val) = (local_val)
#define GLOBAL_NORM(state, local_norm, global_norm) (global_norm) = (local_norm)
#endif

static inline int SOLVE_INNER_LBFGS(cardal_sdp_solver_state_t *state,
                                    double tolerance) {
  int len = state->length_low_rank_solution;
  int m_lbfgs = state->lbfgs_history_size > 0 ? state->lbfgs_history_size : 5;
  if (m_lbfgs > state->lbfgs_buf_capacity_m)
    m_lbfgs = state->lbfgs_buf_capacity_m;

  double *d_S = state->d_lbfgs_S;
  double *d_Y = state->d_lbfgs_Y;
  double *d_R_old = state->d_lbfgs_R_old;
  double *d_Grad_old = state->d_lbfgs_Grad_old;
  double *d_s = state->d_lbfgs_s;
  double *d_y = state->d_lbfgs_y;
  double *d_q = state->d_lbfgs_q;
  double *d_z = state->d_lbfgs_z;
  double *d_scr = state->d_lbfgs_scratch;

  double rho[64];
  double alpha[64];

  int head = 0;
  int history_size = 0;

  cublasPointerMode_t saved_mode;
  CUBLAS_CHECK(cublasGetPointerMode(state->blas_handle, &saved_mode));
  CUBLAS_CHECK(
      cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

  CUBLAS_CHECK(cublasDcopy(state->blas_handle, len, state->low_rank_solution, 1,
                           d_R_old, 1));

  state->lbfgs_exit_reason = LBFGS_MAX_ITERS;

  #define DEVICE_DOT_TO_HOST(VEC_A, VEC_B, SLOT, HOST_VAR)                      \
    do {                                                                       \
      CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle,                    \
                                        CUBLAS_POINTER_MODE_DEVICE));          \
      CUBLAS_CHECK(cublasDdot(state->blas_handle, len, (VEC_A), 1, (VEC_B), 1, \
                              d_scr + (SLOT)));                                \
      CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle,                    \
                                        CUBLAS_POINTER_MODE_HOST));            \
      CUDA_CHECK(cudaMemcpy(&(HOST_VAR), d_scr + (SLOT), sizeof(double),       \
                            cudaMemcpyDeviceToHost));                          \
    } while (0)
  #define DEVICE_NRM2_TO_HOST(VEC, SLOT, HOST_VAR)                             \
    do {                                                                       \
      CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle,                    \
                                        CUBLAS_POINTER_MODE_DEVICE));          \
      CUBLAS_CHECK(                                                            \
          cublasDnrm2(state->blas_handle, len, (VEC), 1, d_scr + (SLOT)));     \
      CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle,                    \
                                        CUBLAS_POINTER_MODE_HOST));            \
      CUDA_CHECK(cudaMemcpy(&(HOST_VAR), d_scr + (SLOT), sizeof(double),       \
                            cudaMemcpyDeviceToHost));                          \
    } while (0)

  int inner;
  for (inner = 0; inner < state->inner_iterations_limit; inner++) {
    if (inner > 0)
      COMPUTE_GRADIENT(state);

    double grad_norm_local = 0.0, grad_norm = 0.0;
    DEVICE_NRM2_TO_HOST(state->low_rank_gradient, LBFGS_SCR_GRAD_NORM,
                        grad_norm_local);
    GLOBAL_NORM(state, grad_norm_local, grad_norm);
    if (inner > 0 && grad_norm <= tolerance) {
      state->lbfgs_exit_reason = LBFGS_GRAD_CONVERGED;
      break;
    }

    if (inner > 0) {
      CUBLAS_CHECK(cublasDcopy(state->blas_handle, len,
                               state->low_rank_solution, 1, d_s, 1));
      double minus_one = -1.0;
      CUBLAS_CHECK(
          cublasDaxpy(state->blas_handle, len, &minus_one, d_R_old, 1, d_s, 1));
      CUBLAS_CHECK(cublasDcopy(state->blas_handle, len,
                               state->low_rank_gradient, 1, d_y, 1));
      CUBLAS_CHECK(cublasDaxpy(state->blas_handle, len, &minus_one, d_Grad_old,
                               1, d_y, 1));

      double sy_local = 0.0, sy = 0.0;
      DEVICE_DOT_TO_HOST(d_s, d_y, LBFGS_SCR_SY, sy_local);
      GLOBAL_DOT(state, sy_local, sy);

      if (sy > 1e-10) {
        double *s_ptr = d_S + head * len;
        double *y_ptr = d_Y + head * len;
        CUBLAS_CHECK(cublasDcopy(state->blas_handle, len, d_s, 1, s_ptr, 1));
        CUBLAS_CHECK(cublasDcopy(state->blas_handle, len, d_y, 1, y_ptr, 1));

        rho[head] = 1.0 / sy;
        head = (head + 1) % m_lbfgs;
        if (history_size < m_lbfgs)
          history_size++;
      } else {
        history_size = 0;
        head = 0;
      }
    }

    CUBLAS_CHECK(cublasDcopy(state->blas_handle, len, state->low_rank_solution,
                             1, d_R_old, 1));
    CUBLAS_CHECK(cublasDcopy(state->blas_handle, len, state->low_rank_gradient,
                             1, d_Grad_old, 1));
    CUBLAS_CHECK(cublasDcopy(state->blas_handle, len, state->low_rank_gradient,
                             1, d_q, 1));

    if (history_size == 0) {
      CUBLAS_CHECK(cublasDcopy(state->blas_handle, len, d_q, 1, d_z, 1));
    } else {
      for (int i = 0; i < history_size; i++) {
        int idx = (head - 1 - i + m_lbfgs) % m_lbfgs;
        double *s_ptr = d_S + idx * len;
        double *y_ptr = d_Y + idx * len;

        double sq_local = 0.0, sq = 0.0;
        DEVICE_DOT_TO_HOST(s_ptr, d_q, LBFGS_SCR_SQ, sq_local);
        GLOBAL_DOT(state, sq_local, sq);

        alpha[idx] = rho[idx] * sq;
        double minus_a = -alpha[idx];
        CUBLAS_CHECK(
            cublasDaxpy(state->blas_handle, len, &minus_a, y_ptr, 1, d_q, 1));
      }

      int latest_idx = (head - 1 + m_lbfgs) % m_lbfgs;
      double *s_latest = d_S + latest_idx * len;
      double *y_latest = d_Y + latest_idx * len;

      double yy_local = 0.0, yy = 0.0;
      double sy_latest_local = 0.0, sy_latest = 0.0;
      DEVICE_DOT_TO_HOST(y_latest, y_latest, LBFGS_SCR_YY, yy_local);
      GLOBAL_DOT(state, yy_local, yy);
      DEVICE_DOT_TO_HOST(s_latest, y_latest, LBFGS_SCR_SY_LATEST,
                         sy_latest_local);
      GLOBAL_DOT(state, sy_latest_local, sy_latest);

      double gamma = 1.0;
      if (yy > 1e-16) {
        gamma = sy_latest / yy;
      }
      CUBLAS_CHECK(cublasDcopy(state->blas_handle, len, d_q, 1, d_z, 1));
      CUBLAS_CHECK(cublasDscal(state->blas_handle, len, &gamma, d_z, 1));

      for (int i = history_size - 1; i >= 0; i--) {
        int idx = (head - 1 - i + m_lbfgs) % m_lbfgs;
        double *s_ptr = d_S + idx * len;
        double *y_ptr = d_Y + idx * len;

        double yz_local = 0.0, yz = 0.0;
        DEVICE_DOT_TO_HOST(y_ptr, d_z, LBFGS_SCR_YZ, yz_local);
        GLOBAL_DOT(state, yz_local, yz);

        double beta = rho[idx] * yz;
        double factor = alpha[idx] - beta;
        CUBLAS_CHECK(
            cublasDaxpy(state->blas_handle, len, &factor, s_ptr, 1, d_z, 1));
      }
    }

    double dir_grad_dot_local = 0.0, dir_grad_dot = 0.0;
    DEVICE_DOT_TO_HOST(state->low_rank_gradient, d_z, LBFGS_SCR_DIR_GRAD_DOT,
                       dir_grad_dot_local);
    GLOBAL_DOT(state, dir_grad_dot_local, dir_grad_dot);

    double minus_one = -1.0;
    if (dir_grad_dot > 1e-12) {
      CUBLAS_CHECK(cublasDcopy(state->blas_handle, len, d_z, 1,
                               state->low_rank_direction, 1));
      CUBLAS_CHECK(cublasDscal(state->blas_handle, len, &minus_one,
                               state->low_rank_direction, 1));
    } else {
      CUBLAS_CHECK(cublasDcopy(state->blas_handle, len,
                               state->low_rank_gradient, 1,
                               state->low_rank_direction, 1));
      CUBLAS_CHECK(cublasDscal(state->blas_handle, len, &minus_one,
                               state->low_rank_direction, 1));
      history_size = 0;
      head = 0;
    }

    double dir_norm_local = 0.0, dir_norm = 0.0;
    DEVICE_NRM2_TO_HOST(state->low_rank_direction, LBFGS_SCR_DIR_NORM,
                        dir_norm_local);
    GLOBAL_NORM(state, dir_norm_local, dir_norm);
    if (dir_norm > 1e-12) {
      double inv_norm = 1.0 / dir_norm;
      CUBLAS_CHECK(cublasDscal(state->blas_handle, len, &inv_norm,
                               state->low_rank_direction, 1));
    }

    double optimal_tau = COMPUTE_EXACT_STEP_SIZE(state);
    if (fabs(optimal_tau) < 1e-8) {
      state->lbfgs_exit_reason = LBFGS_TAU_STALL;
      CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, saved_mode));
      return inner;
    }
    CUBLAS_CHECK(cublasDaxpy(state->blas_handle, len, &optimal_tau,
                             state->low_rank_direction, 1,
                             state->low_rank_solution, 1));
  }

  CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, saved_mode));

  #undef DEVICE_DOT_TO_HOST
  #undef DEVICE_NRM2_TO_HOST
  return inner;
}

#endif // INNER_SOLVERS_CUH
