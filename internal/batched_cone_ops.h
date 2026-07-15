/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once
#include "internal_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_batched_sddmm_self_scatter(const double *X,
                                       const block_low_rank_state_t *batch,
                                       double *global_target,
                                       cudaStream_t stream);

void launch_batched_sddmm_cross_scatter(const double *X, const double *Y,
                                        const block_low_rank_state_t *batch,
                                        double *global_target,
                                        cudaStream_t stream);

void launch_batched_sddmm_self_to_spS(const double *X,
                                      const block_low_rank_state_t *batch,
                                      cudaStream_t stream);

void launch_batched_segmented_dot_p2(const block_low_rank_state_t *batch,
                                     const int *d_small_to_blk_idx,
                                     double *d_p2_per_cone,
                                     cudaStream_t stream);

void launch_batched_copy_objval_to_spS(const block_low_rank_state_t *batch,
                                       cudaStream_t stream);

void launch_batched_add_Ay_to_spS(const block_low_rank_state_t *batch,
                                  const double *dual_product,
                                  cudaStream_t stream);

void launch_batched_spmm_grad(const block_low_rank_state_t *batch,
                              const double *R_global, double *G_global,
                              double alpha, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
