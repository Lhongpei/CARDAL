/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once
#include "solver_state.h"
#include <cuda_runtime.h>
#include <cusparse.h>

#ifdef __cplusplus
extern "C" {
#endif
__device__ double atomicMinDouble(double *address, double val);
__global__ void compute_q1_kernel(double *__restrict__ q1,
                                  const double *__restrict__ lambda,
                                  const double *__restrict__ q0, double rho,
                                  int m);
__global__ void add_Ay_to_S_kernel(double *__restrict__ S_val,
                                   const double *__restrict__ Ay_val_global,
                                   const int *__restrict__ compat_mapping,
                                   const int *__restrict__ A_to_Union_mapping,
                                   int nnz_A);
__global__ void
scatter_sddmm_to_global_kernel(const double *__restrict__ local_vals,
                               const int *__restrict__ compat_mapping,
                               double *__restrict__ global_primal_solution,
                               int nnz);
__global__ void compute_lp_primal_kernel(double *__restrict__ primal_sol,
                                         const double *__restrict__ lp_v,
                                         int lp_dim);
__global__ void compute_lp_gradient_kernel(double *__restrict__ lp_grad,
                                           const double *__restrict__ lp_C,
                                           const double *__restrict__ dual_prod,
                                           const double *__restrict__ lp_v,
                                           int lp_dim);
__global__ void compute_lp_line_search_kernel(const double *__restrict__ v,
                                              const double *__restrict__ dv,
                                              double *__restrict__ primal_rd,
                                              double *__restrict__ primal_d,
                                              int lp_dim);
void compute_RR_block(int blk_idx, cardal_sdp_solver_state_t *state);
void update_dual_slack_S(cardal_sdp_solver_state_t *state);
void update_al_gradient_S(cardal_sdp_solver_state_t *state);
// PERCONE only.
void update_penalty_only_S(cardal_sdp_solver_state_t *state);
double compute_penalty_perturbation_fnorm(cardal_sdp_solver_state_t *state);
// PERCONE only, single-GPU.
double compute_Asuu_norm_sq_percone(cardal_sdp_solver_state_t *state,
                                    block_low_rank_state_t *blk, double *d_u);
void compute_rank_lift_A_ww(cardal_sdp_solver_state_t *state,
                            block_low_rank_state_t *blk, double *d_w,
                            double *d_out_m);
void compute_rank_lift_A_uv(cardal_sdp_solver_state_t *state,
                            block_low_rank_state_t *blk, double *d_u,
                            double *d_v, double *d_out_m);
// PERCONE only. q0_norm_budget <= 0 disables trust cap. Returns realized ||Cs||.
double solve_alora_sdp_percone(cardal_sdp_solver_state_t *state,
                               block_low_rank_state_t *blk, const double *d_V,
                               int r, double rho, double q0_norm_budget,
                               double *d_new_cols);

void compute_RD_DD_block(int blk_idx, cardal_sdp_solver_state_t *state);

__global__ void elementwise_multiply_kernel(const double *__restrict__ in,
                                            const double *__restrict__ scale,
                                            double *__restrict__ out, int n);
__global__ void
elementwise_multiply_inplace_kernel(double *__restrict__ inout,
                                    const double *__restrict__ scale, int n);
__global__ void
elementwise_multiply_scaled_kernel(double *__restrict__ inout,
                                   const double *__restrict__ scale,
                                   double scalar, int n);

// Unscaled action: y = D * S_scaled * D * x, D = diag(psd_cone_rescaling).
void unscaled_dual_spmv(cardal_sdp_solver_state_t *state,
                        block_low_rank_state_t *blk, double *d_in,
                        double *d_out, double *d_scratch,
                        cusparseDnVecDescr_t vec_in,
                        cusparseDnVecDescr_t vec_out, void *dBuffer);

double compute_unscaled_q0_l2_norm(cardal_sdp_solver_state_t *state);

void populate_state_scaling_fields(cardal_sdp_solver_state_t *state,
                                   const rescale_info_t *info);

void populate_block_psd_cone_rescaling(block_low_rank_state_t *blk_state,
                                       int n_k, int nnz_Union,
                                       const int *h_row_ptr_U,
                                       const int *h_col_ind_U,
                                       const double *d_per_row);

void unscale_result(const rescale_info_t *info,
                    const cardal_sdp_solver_state_t *state,
                    sdp_result_t *result);
#ifdef __cplusplus
}
#endif
