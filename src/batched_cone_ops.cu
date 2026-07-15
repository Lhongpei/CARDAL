/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "batched_cone_ops.h"
#include "internal_types.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ double warp_reduce_sum_d(double v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

template <bool SAME_INPUTS>
__global__ void batched_sddmm_scatter_kernel(
    const double *__restrict__ X, const double *__restrict__ Y,
    const long long *__restrict__ R_offsets_X,
    const long long *__restrict__ R_offsets_Y, int dim,
    const int *__restrict__ ranks, const int *__restrict__ entry_cone,
    const int *__restrict__ entry_row, const int *__restrict__ entry_col,
    const int *__restrict__ entry_compat, int total_nnz,
    double *__restrict__ global_target) {
  int e = blockIdx.x;
  if (e >= total_nnz)
    return;
  int cone = entry_cone[e];
  int row = entry_row[e];
  int col = entry_col[e];
  int rank = ranks[cone];
  const double *Xp = X + R_offsets_X[cone];
  const double *Yp = SAME_INPUTS ? Xp : (Y + R_offsets_Y[cone]);

  int tid = threadIdx.x;
  double acc = 0.0;
  for (int t = tid; t < rank; t += 32) {
    acc += Xp[row + (long long)t * dim] * Yp[col + (long long)t * dim];
  }
  acc = warp_reduce_sum_d(acc);
  if (tid == 0) {
    int target = entry_compat[e];
    global_target[target] = acc;
  }
}

template <bool SAME_INPUTS>
__global__ void batched_sddmm_to_buf_kernel(
    const double *__restrict__ X, const double *__restrict__ Y,
    const long long *__restrict__ R_offsets_X,
    const long long *__restrict__ R_offsets_Y, int dim,
    const int *__restrict__ ranks, const int *__restrict__ entry_cone,
    const int *__restrict__ entry_row, const int *__restrict__ entry_col,
    int total_nnz, double *__restrict__ out_vals) {
  int e = blockIdx.x;
  if (e >= total_nnz)
    return;
  int cone = entry_cone[e];
  int row = entry_row[e];
  int col = entry_col[e];
  int rank = ranks[cone];
  const double *Xp = X + R_offsets_X[cone];
  const double *Yp = SAME_INPUTS ? Xp : (Y + R_offsets_Y[cone]);

  int tid = threadIdx.x;
  double acc = 0.0;
  for (int t = tid; t < rank; t += 32) {
    acc += Xp[row + (long long)t * dim] * Yp[col + (long long)t * dim];
  }
  acc = warp_reduce_sum_d(acc);
  if (tid == 0)
    out_vals[e] = acc;
}

__global__ void
batched_segmented_dot_p2_kernel(const int n_small,
                                const int *__restrict__ cone_S_offsets,
                                const double *__restrict__ vals,
                                const double *__restrict__ objvals,
                                const int *__restrict__ small_to_blk_idx,
                                double *__restrict__ p2_per_cone) {
  int s = blockIdx.x;
  if (s >= n_small)
    return;
  int start = cone_S_offsets[s];
  int end = cone_S_offsets[s + 1];
  int tid = threadIdx.x;
  double acc = 0.0;
  for (int e = start + tid; e < end; e += blockDim.x) {
    acc += vals[e] * objvals[e];
  }
  __shared__ double sdata[8];
  int lane = tid & 31;
  int warp = tid >> 5;
  acc = warp_reduce_sum_d(acc);
  if (lane == 0)
    sdata[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    double v = (tid < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0;
    v = warp_reduce_sum_d(v);
    if (lane == 0) {
      int blk_idx = small_to_blk_idx[s];
      p2_per_cone[blk_idx] = v;
    }
  }
}

extern "C" void
launch_batched_sddmm_self_scatter(const double *X,
                                  const block_low_rank_state_t *batch,
                                  double *global_target,
                                  cudaStream_t stream) {
  if (batch->bdata->total_nnz_A <= 0)
    return;
  dim3 grid(batch->bdata->total_nnz_A);
  dim3 block(32);
  batched_sddmm_scatter_kernel<true><<<grid, block, 0, stream>>>(
      X, X, batch->bdata->d_R_offsets, batch->bdata->d_R_offsets, batch->dim,
      batch->bdata->d_ranks, batch->bdata->d_entry_cone_A, batch->bdata->d_entry_row_A,
      batch->bdata->d_entry_col_A, batch->bdata->d_entry_compat_A, batch->bdata->total_nnz_A,
      global_target);
}

extern "C" void
launch_batched_sddmm_cross_scatter(const double *X, const double *Y,
                                   const block_low_rank_state_t *batch,
                                   double *global_target,
                                   cudaStream_t stream) {
  if (batch->bdata->total_nnz_A <= 0)
    return;
  dim3 grid(batch->bdata->total_nnz_A);
  dim3 block(32);
  batched_sddmm_scatter_kernel<false><<<grid, block, 0, stream>>>(
      X, Y, batch->bdata->d_R_offsets, batch->bdata->d_D_offsets, batch->dim,
      batch->bdata->d_ranks, batch->bdata->d_entry_cone_A, batch->bdata->d_entry_row_A,
      batch->bdata->d_entry_col_A, batch->bdata->d_entry_compat_A, batch->bdata->total_nnz_A,
      global_target);
}

extern "C" void
launch_batched_sddmm_self_to_spS(const double *X,
                                 const block_low_rank_state_t *batch,
                                 cudaStream_t stream) {
  if (batch->bdata->total_nnz_S <= 0)
    return;
  dim3 grid(batch->bdata->total_nnz_S);
  dim3 block(32);
  batched_sddmm_to_buf_kernel<true><<<grid, block, 0, stream>>>(
      X, X, batch->bdata->d_R_offsets, batch->bdata->d_R_offsets, batch->dim,
      batch->bdata->d_ranks, batch->bdata->d_entry_cone_S, batch->bdata->d_entry_row_S,
      batch->bdata->d_entry_col_S, batch->bdata->total_nnz_S, batch->bdata->d_flat_spS_val);
}

__global__ void copy_flat_objval_to_spS_kernel(int n, const double *src,
                                               double *dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = src[i];
}

__global__ void
add_Ay_to_spS_kernel(int total_nnz_A, const int *__restrict__ entry_compat,
                     const int *__restrict__ entry_to_flatS,
                     const double *__restrict__ dual_product,
                     double *__restrict__ flat_spS_val) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= total_nnz_A)
    return;
  int s_slot = entry_to_flatS[e];
  int compat = entry_compat[e];
  flat_spS_val[s_slot] += dual_product[compat];
}

extern "C" void
launch_batched_copy_objval_to_spS(const block_low_rank_state_t *batch,
                                  cudaStream_t stream) {
  if (batch->bdata->total_nnz_S <= 0)
    return;
  int blocks = (batch->bdata->total_nnz_S + 255) / 256;
  copy_flat_objval_to_spS_kernel<<<blocks, 256, 0, stream>>>(
      batch->bdata->total_nnz_S, batch->bdata->d_flat_objval_S, batch->bdata->d_flat_spS_val);
}

extern "C" void
launch_batched_add_Ay_to_spS(const block_low_rank_state_t *batch,
                             const double *dual_product, cudaStream_t stream) {
  if (batch->bdata->total_nnz_A <= 0)
    return;
  int blocks = (batch->bdata->total_nnz_A + 255) / 256;
  add_Ay_to_spS_kernel<<<blocks, 256, 0, stream>>>(
      batch->bdata->total_nnz_A, batch->bdata->d_entry_compat_A, batch->bdata->d_entry_A_to_flatS,
      dual_product, batch->bdata->d_flat_spS_val);
}

extern "C" void
launch_batched_segmented_dot_p2(const block_low_rank_state_t *batch,
                                const int *d_small_to_blk_idx,
                                double *d_p2_per_cone,
                                cudaStream_t stream) {
  if (batch->n_cones <= 0 || batch->bdata->total_nnz_S <= 0)
    return;
  dim3 grid(batch->n_cones);
  dim3 block(256);
  batched_segmented_dot_p2_kernel<<<grid, block, 0, stream>>>(
      batch->n_cones, batch->bdata->d_cone_S_offsets, batch->bdata->d_flat_spS_val,
      batch->bdata->d_flat_objval_S, d_small_to_blk_idx, d_p2_per_cone);
}

template <int TILE_K, int MAX_DIM>
__global__ void batched_spmm_kernel(
    int n_small, int dim, const int *__restrict__ ranks,
    const long long *__restrict__ G_offsets,
    const long long *__restrict__ R_offsets,
    const int *__restrict__ S_row_ptr_flat,
    const int *__restrict__ S_row_ptr_offsets,
    const int *__restrict__ S_col_ind, const double *__restrict__ S_val,
    const int *__restrict__ cone_S_offsets, double alpha,
    const double *__restrict__ R_global, double *__restrict__ G_global) {
  int s = blockIdx.x;
  int rank = ranks[s];
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ double R_tile[MAX_DIM * TILE_K];

  int total_load = dim * TILE_K;
  long long R_base = R_offsets[s] + (long long)blockIdx.y * TILE_K * dim;
  int tid_flat = ty * TILE_K + tx;
  int n_threads = TILE_K * MAX_DIM;
  for (int idx = tid_flat; idx < total_load; idx += n_threads) {
    int local_k = idx / dim;
    int global_k = blockIdx.y * TILE_K + local_k;
    R_tile[idx] = (global_k < rank) ? R_global[R_base + idx] : 0.0;
  }
  __syncthreads();

  int k = blockIdx.y * TILE_K + tx;
  if (ty >= dim || k >= rank)
    return;

  int s_S_off = cone_S_offsets[s];
  int s_row_ptr_off = S_row_ptr_offsets[s];
  int row_start = S_row_ptr_flat[s_row_ptr_off + ty];
  int row_end = S_row_ptr_flat[s_row_ptr_off + ty + 1];

  double acc = 0.0;
  for (int p = row_start; p < row_end; p++) {
    int col = S_col_ind[s_S_off + p];
    double v = S_val[s_S_off + p];
    acc += v * R_tile[col + tx * dim];
  }
  G_global[G_offsets[s] + ty + (long long)k * dim] = alpha * acc;
}

template <int TILE_K_, int MAX_DIM_>
static inline void launch_one(const block_low_rank_state_t *batch,
                              const double *R_global, double *G_global,
                              double alpha, int max_rank,
                              cudaStream_t stream) {
  dim3 grid(batch->n_cones, (max_rank + TILE_K_ - 1) / TILE_K_);
  dim3 block(TILE_K_, MAX_DIM_);
  batched_spmm_kernel<TILE_K_, MAX_DIM_><<<grid, block, 0, stream>>>(
      batch->n_cones, batch->dim, batch->bdata->d_ranks,
      batch->bdata->d_G_offsets, batch->bdata->d_R_offsets,
      batch->bdata->d_S_row_ptr_flat, batch->bdata->d_S_row_ptr_offsets,
      batch->bdata->d_entry_col_S, batch->bdata->d_flat_spS_val,
      batch->bdata->d_cone_S_offsets, alpha, R_global, G_global);
}

extern "C" void launch_batched_spmm_grad(const block_low_rank_state_t *batch,
                                         const double *R_global,
                                         double *G_global, double alpha,
                                         cudaStream_t stream) {
  if (batch->n_cones <= 0)
    return;
  int max_rank = batch->bdata->max_rank;
  if (max_rank == 0)
    return;

  int dim = batch->dim;
  if (dim <= 32) {
    launch_one<32, 32>(batch, R_global, G_global, alpha, max_rank, stream);
  } else if (dim <= 64) {
    launch_one<16, 64>(batch, R_global, G_global, alpha, max_rank, stream);
  } else if (dim <= 128) {
    launch_one<8, 128>(batch, R_global, G_global, alpha, max_rank, stream);
  } else if (dim <= 256) {
    launch_one<4, 256>(batch, R_global, G_global, alpha, max_rank, stream);
  } else {
    fprintf(stderr,
            "[launch_batched_spmm_grad] unsupported batch dim %d (> 256); "
            "host-side grouping should have routed this cone to PERCONE\n",
            dim);
    abort();
  }
}
