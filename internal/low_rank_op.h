/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once

#include "solver_state.h"

#ifdef __cplusplus
extern "C" {
#endif

void initialize_low_rank_blocks(cardal_sdp_solver_state_t *state,
                                const compressed_sdp_problem_t *problem);
void initialize_batched_low_rank(cardal_sdp_solver_state_t *state,
                                 block_low_rank_state_t *batch);
void free_low_rank_block(low_rank_block_data_t *data);

/* Add A_lr(X X^T) to out_constraints and/or the low-rank objective value to
 * out_objective. Output pointers are device pointers and are accumulated. */
void low_rank_eval_self(cardal_sdp_solver_state_t *state,
                        block_low_rank_state_t *blk, const double *d_X,
                        int nrhs, double *d_out_constraints,
                        double *d_out_objective);

/* Add A_lr(X Y^T) to out_constraints and the corresponding objective value.
 * The symmetric directional derivative is twice this quantity. */
void low_rank_eval_cross(cardal_sdp_solver_state_t *state,
                         block_low_rank_state_t *blk, const double *d_X,
                         const double *d_Y, int nrhs, double *d_out_constraints,
                         double *d_out_objective);

/* out += alpha * (C_lr + A_lr^*(q)) X. */
void low_rank_add_operator_times_matrix(cardal_sdp_solver_state_t *state,
                                        block_low_rank_state_t *blk,
                                        const double *d_q,
                                        int include_objective,
                                        const double *d_X, int nrhs,
                                        double alpha, double *d_out);

/* out += alpha * (C_lr + A_lr^*(q)) x. */
void low_rank_add_operator_times_vector(cardal_sdp_solver_state_t *state,
                                        block_low_rank_state_t *blk,
                                        const double *d_q,
                                        int include_objective,
                                        const double *d_x, double alpha,
                                        double *d_out);

/* Warp-batched variants for custom small-cone batches. X/Y/out are state-level
 * pools; their per-cone offsets select the corresponding member matrices. */
void batched_low_rank_eval_self(const block_low_rank_state_t *batch,
                                const double *d_X, const long long *d_X_offsets,
                                double *d_out_constraints,
                                double *d_out_objective, cudaStream_t stream);
void batched_low_rank_eval_cross(const block_low_rank_state_t *batch,
                                 const double *d_X,
                                 const long long *d_X_offsets,
                                 const double *d_Y,
                                 const long long *d_Y_offsets,
                                 double *d_out_constraints,
                                 double *d_out_objective, cudaStream_t stream);
void batched_low_rank_add_operator_times_matrix(
    const block_low_rank_state_t *batch, const double *d_q,
    int include_objective, const double *d_X, const long long *d_X_offsets,
    double alpha, double *d_out, const long long *d_out_offsets,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
