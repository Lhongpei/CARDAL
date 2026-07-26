/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "low_rank_op.h"
#include "utils.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string.h>

static __device__ double low_rank_atomic_add(double *address, double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  unsigned long long *bits = (unsigned long long *)address;
  unsigned long long old = *bits;
  unsigned long long assumed;
  do {
    assumed = old;
    old =
        atomicCAS(bits, assumed,
                  __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (old != assumed);
  return __longlong_as_double(old);
#endif
}

static __global__ void reduce_low_rank_projection_kernel(
    const double *__restrict__ px, const double *__restrict__ py, int k,
    int nrhs, const int *__restrict__ constraint,
    const double *__restrict__ weights, double *__restrict__ out_constraints,
    double *__restrict__ out_objective) {
  int column = blockIdx.x;
  if (column >= k)
    return;
  double sum = 0.0;
  for (int rhs = threadIdx.x; rhs < nrhs; rhs += blockDim.x) {
    double x = px[column + (size_t)rhs * k];
    double y = py ? py[column + (size_t)rhs * k] : x;
    sum += x * y;
  }
  __shared__ double shared[256];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double value = weights[column] * shared[0];
    int row = constraint[column];
    if (row >= 0) {
      if (out_constraints)
        low_rank_atomic_add(out_constraints + row, value);
    } else if (out_objective) {
      low_rank_atomic_add(out_objective, value);
    }
  }
}

static __global__ void scale_low_rank_projection_kernel(
    double *__restrict__ projection, int k, int nrhs,
    const int *__restrict__ constraint, const double *__restrict__ weights,
    const double *__restrict__ q, int include_objective, double alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = k * nrhs;
  if (idx >= n)
    return;
  int column = idx % k;
  int row = constraint[column];
  double coeff = 0.0;
  if (row >= 0)
    coeff = q ? q[row] : 0.0;
  else if (include_objective)
    coeff = 1.0;
  projection[idx] *= alpha * weights[column] * coeff;
}

static __device__ __forceinline__ double low_rank_warp_sum(double value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

template <bool SAME_INPUTS>
static __global__ void batched_low_rank_eval_kernel(
    int total_columns, int dim, const int *__restrict__ column_cone,
    const int *__restrict__ cone_offsets, const int *__restrict__ constraints,
    const double *__restrict__ weights, double *const *__restrict__ factors,
    const int *__restrict__ ranks, const double *__restrict__ X,
    const long long *__restrict__ X_offsets, const double *__restrict__ Y,
    const long long *__restrict__ Y_offsets,
    double *__restrict__ out_constraints, double *__restrict__ out_objective) {
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int column = blockIdx.x * (blockDim.x >> 5) + warp;
  if (column >= total_columns)
    return;

  int constraint = constraints[column];
  if ((constraint >= 0 && out_constraints == NULL) ||
      (constraint < 0 && out_objective == NULL))
    return;

  int cone = column_cone[column];
  int local_column = column - cone_offsets[cone];
  int rank = ranks[cone];
  const double *u = factors[cone] + (long long)local_column * dim;
  const double *Xc = X + X_offsets[cone];
  const double *Yc = SAME_INPUTS ? Xc : Y + Y_offsets[cone];

  double value = 0.0;
  for (int rhs = 0; rhs < rank; rhs++) {
    double x_dot = 0.0;
    double y_dot = 0.0;
    for (int row = lane; row < dim; row += 32) {
      double u_row = u[row];
      x_dot += u_row * Xc[row + (long long)rhs * dim];
      if (!SAME_INPUTS)
        y_dot += u_row * Yc[row + (long long)rhs * dim];
    }
    x_dot = low_rank_warp_sum(x_dot);
    if (SAME_INPUTS) {
      if (lane == 0)
        value += x_dot * x_dot;
    } else {
      y_dot = low_rank_warp_sum(y_dot);
      if (lane == 0)
        value += x_dot * y_dot;
    }
  }

  if (lane == 0) {
    value *= weights[column];
    if (constraint >= 0)
      low_rank_atomic_add(out_constraints + constraint, value);
    else
      low_rank_atomic_add(out_objective, value);
  }
}

static __global__ void batched_low_rank_operator_kernel(
    int n_cones, int dim, int max_rank, const int *__restrict__ cone_offsets,
    const int *__restrict__ constraints, const double *__restrict__ weights,
    double *const *__restrict__ factors, const int *__restrict__ ranks,
    const double *__restrict__ q, int include_objective,
    const double *__restrict__ X, const long long *__restrict__ X_offsets,
    double alpha, double *__restrict__ out,
    const long long *__restrict__ out_offsets) {
  int cone = blockIdx.x;
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int rhs = blockIdx.y * (blockDim.x >> 5) + warp;
  if (cone >= n_cones || rhs >= max_rank || rhs >= ranks[cone])
    return;

  constexpr int ROWS_PER_LANE = (CARDAL_SMALL_CONE_DIM_THRESHOLD + 31) / 32;
  double accum[ROWS_PER_LANE];
#pragma unroll
  for (int i = 0; i < ROWS_PER_LANE; i++)
    accum[i] = 0.0;

  const double *Xc = X + X_offsets[cone];
  int start = cone_offsets[cone];
  int end = cone_offsets[cone + 1];
  for (int column = start; column < end; column++) {
    const double *u = factors[cone] + (long long)(column - start) * dim;
    double dot = 0.0;
    for (int row = lane; row < dim; row += 32)
      dot += u[row] * Xc[row + (long long)rhs * dim];
    dot = low_rank_warp_sum(dot);
    dot = __shfl_sync(0xffffffff, dot, 0);

    int constraint = constraints[column];
    double coeff = 0.0;
    if (constraint >= 0)
      coeff = q ? q[constraint] : 0.0;
    else if (include_objective)
      coeff = 1.0;
    double scaled_dot = weights[column] * coeff * dot;
    int slot = 0;
    for (int row = lane; row < dim; row += 32)
      accum[slot++] += scaled_dot * u[row];
  }

  double *out_c = out + out_offsets[cone];
  int slot = 0;
  for (int row = lane; row < dim; row += 32)
    out_c[row + (long long)rhs * dim] += alpha * accum[slot++];
}

static void ensure_projection_capacity(low_rank_block_data_t *lr, int nrhs) {
  if (!lr || nrhs <= lr->projection_rank_capacity)
    return;
  if (lr->projection)
    CUDA_CHECK(cudaFree(lr->projection));
  if (lr->projection_aux)
    CUDA_CHECK(cudaFree(lr->projection_aux));
  size_t count = (size_t)lr->num_columns * (size_t)nrhs;
  CUDA_CHECK(cudaMalloc(&lr->projection, count * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lr->projection_aux, count * sizeof(double)));
  lr->projection_rank_capacity = nrhs;
}

void initialize_low_rank_blocks(cardal_sdp_solver_state_t *state,
                                const compressed_sdp_problem_t *problem) {
  if (!state || !problem || !problem->low_rank_data)
    return;
  const symmetric_low_rank_data_t *src = problem->low_rank_data;
  int *counts = (int *)safe_calloc(
      (size_t)(state->n_blks > 0 ? state->n_blks : 1), sizeof(int));
  for (int j = 0; j < src->num_columns; j++) {
    int cone = src->cone_ind[j];
    if (cone >= 0 && cone < state->n_blks)
      counts[cone]++;
  }

  for (int cone = 0; cone < state->n_blks; cone++) {
    int k = counts[cone];
    if (k == 0)
      continue;
    block_low_rank_state_t *blk = state->block_low_rank_state[cone];
    low_rank_block_data_t *lr =
        (low_rank_block_data_t *)safe_calloc(1, sizeof(low_rank_block_data_t));
    lr->num_columns = k;
    lr->dim = blk->dim;

    int *h_constraint = (int *)safe_malloc((size_t)k * sizeof(int));
    double *h_weights = (double *)safe_malloc((size_t)k * sizeof(double));
    double *h_factors =
        (double *)safe_malloc((size_t)blk->dim * (size_t)k * sizeof(double));
    int write = 0;
    for (int j = 0; j < src->num_columns; j++) {
      if (src->cone_ind[j] != cone)
        continue;
      h_constraint[write] = src->constraint_ind[j];
      h_weights[write] = src->weights[j];
      memcpy(h_factors + (size_t)write * blk->dim,
             src->factor_values + src->factor_ptr[j],
             (size_t)blk->dim * sizeof(double));
      write++;
    }
    CUDA_CHECK(cudaMalloc(&lr->column_constraint, (size_t)k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lr->weights, (size_t)k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lr->factors,
                          (size_t)blk->dim * (size_t)k * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(lr->column_constraint, h_constraint,
                          (size_t)k * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lr->weights, h_weights, (size_t)k * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lr->factors, h_factors,
                          (size_t)blk->dim * (size_t)k * sizeof(double),
                          cudaMemcpyHostToDevice));
    free(h_constraint);
    free(h_weights);
    free(h_factors);
    blk->lr_data = lr;
  }
  free(counts);
}

void initialize_batched_low_rank(cardal_sdp_solver_state_t *state,
                                 block_low_rank_state_t *batch) {
  if (!state || !batch || batch->kind != CONE_BATCH_KIND_CUSTOM ||
      !batch->bdata)
    return;
  cone_batch_data_t *bd = batch->bdata;
  int n = batch->n_cones;
  int *h_offsets = (int *)safe_malloc((size_t)(n + 1) * sizeof(int));
  h_offsets[0] = 0;
  for (int c = 0; c < n; c++) {
    block_low_rank_state_t *blk = state->block_low_rank_state[bd->blk_idx_h[c]];
    int count = blk->lr_data ? blk->lr_data->num_columns : 0;
    h_offsets[c + 1] = h_offsets[c] + count;
  }
  bd->total_lr_columns = h_offsets[n];
  if (bd->total_lr_columns == 0) {
    free(h_offsets);
    return;
  }

  int total = bd->total_lr_columns;
  int *h_column_cone = (int *)safe_malloc((size_t)total * sizeof(int));
  double **h_factors = (double **)safe_malloc((size_t)n * sizeof(double *));

  CUDA_CHECK(cudaMalloc(&bd->d_lr_cone_offsets, (size_t)(n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&bd->d_lr_column_cone, (size_t)total * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&bd->d_lr_constraints, (size_t)total * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&bd->d_lr_weights, (size_t)total * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&bd->d_lr_factors, (size_t)n * sizeof(double *)));

  for (int c = 0; c < n; c++) {
    block_low_rank_state_t *blk = state->block_low_rank_state[bd->blk_idx_h[c]];
    low_rank_block_data_t *lr = blk->lr_data;
    h_factors[c] = lr ? lr->factors : NULL;
    for (int column = h_offsets[c]; column < h_offsets[c + 1]; column++)
      h_column_cone[column] = c;
    int count = h_offsets[c + 1] - h_offsets[c];
    if (count > 0) {
      CUDA_CHECK(cudaMemcpy(bd->d_lr_constraints + h_offsets[c],
                            lr->column_constraint, (size_t)count * sizeof(int),
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(bd->d_lr_weights + h_offsets[c], lr->weights,
                            (size_t)count * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    }
  }

  CUDA_CHECK(cudaMemcpy(bd->d_lr_cone_offsets, h_offsets,
                        (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(bd->d_lr_column_cone, h_column_cone,
                        (size_t)total * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(bd->d_lr_factors, h_factors,
                        (size_t)n * sizeof(double *), cudaMemcpyHostToDevice));
  free(h_factors);
  free(h_column_cone);
  free(h_offsets);
}

void free_low_rank_block(low_rank_block_data_t *data) {
  if (!data)
    return;
  if (data->column_constraint)
    CUDA_CHECK(cudaFree(data->column_constraint));
  if (data->factors)
    CUDA_CHECK(cudaFree(data->factors));
  if (data->weights)
    CUDA_CHECK(cudaFree(data->weights));
  if (data->projection)
    CUDA_CHECK(cudaFree(data->projection));
  if (data->projection_aux)
    CUDA_CHECK(cudaFree(data->projection_aux));
  free(data);
}

static void project_factors(cardal_sdp_solver_state_t *state,
                            low_rank_block_data_t *lr, const double *d_X,
                            int nrhs, double *projection) {
  double alpha = 1.0, beta = 0.0;
  CUBLAS_CHECK(cublasDgemm(state->blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           lr->num_columns, nrhs, lr->dim, &alpha, lr->factors,
                           lr->dim, d_X, lr->dim, &beta, projection,
                           lr->num_columns));
}

static void eval_common(cardal_sdp_solver_state_t *state,
                        block_low_rank_state_t *blk, const double *d_X,
                        const double *d_Y, int nrhs, double *d_out_constraints,
                        double *d_out_objective) {
  low_rank_block_data_t *lr = blk ? blk->lr_data : NULL;
  if (!lr || nrhs <= 0)
    return;
  cublasPointerMode_t saved_mode;
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetPointerMode(state->blas_handle, &saved_mode));
  CUBLAS_CHECK(cublasGetStream(state->blas_handle, &stream));
  CUBLAS_CHECK(
      cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));
  ensure_projection_capacity(lr, nrhs);
  project_factors(state, lr, d_X, nrhs, lr->projection);
  if (d_Y)
    project_factors(state, lr, d_Y, nrhs, lr->projection_aux);
  reduce_low_rank_projection_kernel<<<lr->num_columns, 256, 0, stream>>>(
      lr->projection, d_Y ? lr->projection_aux : NULL, lr->num_columns, nrhs,
      lr->column_constraint, lr->weights, d_out_constraints, d_out_objective);
  CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, saved_mode));
}

void low_rank_eval_self(cardal_sdp_solver_state_t *state,
                        block_low_rank_state_t *blk, const double *d_X,
                        int nrhs, double *d_out_constraints,
                        double *d_out_objective) {
  eval_common(state, blk, d_X, NULL, nrhs, d_out_constraints, d_out_objective);
}

void low_rank_eval_cross(cardal_sdp_solver_state_t *state,
                         block_low_rank_state_t *blk, const double *d_X,
                         const double *d_Y, int nrhs, double *d_out_constraints,
                         double *d_out_objective) {
  eval_common(state, blk, d_X, d_Y, nrhs, d_out_constraints, d_out_objective);
}

void low_rank_add_operator_times_matrix(cardal_sdp_solver_state_t *state,
                                        block_low_rank_state_t *blk,
                                        const double *d_q,
                                        int include_objective,
                                        const double *d_X, int nrhs,
                                        double alpha, double *d_out) {
  low_rank_block_data_t *lr = blk ? blk->lr_data : NULL;
  if (!lr || nrhs <= 0)
    return;
  cublasPointerMode_t saved_mode;
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetPointerMode(state->blas_handle, &saved_mode));
  CUBLAS_CHECK(cublasGetStream(state->blas_handle, &stream));
  CUBLAS_CHECK(
      cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));
  ensure_projection_capacity(lr, nrhs);
  project_factors(state, lr, d_X, nrhs, lr->projection);
  int n = lr->num_columns * nrhs;
  int blocks = (n + 255) / 256;
  scale_low_rank_projection_kernel<<<blocks, 256, 0, stream>>>(
      lr->projection, lr->num_columns, nrhs, lr->column_constraint, lr->weights,
      d_q, include_objective, alpha);
  double one = 1.0;
  CUBLAS_CHECK(cublasDgemm(state->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           lr->dim, nrhs, lr->num_columns, &one, lr->factors,
                           lr->dim, lr->projection, lr->num_columns, &one,
                           d_out, lr->dim));
  CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, saved_mode));
}

void low_rank_add_operator_times_vector(cardal_sdp_solver_state_t *state,
                                        block_low_rank_state_t *blk,
                                        const double *d_q,
                                        int include_objective,
                                        const double *d_x, double alpha,
                                        double *d_out) {
  low_rank_add_operator_times_matrix(state, blk, d_q, include_objective, d_x, 1,
                                     alpha, d_out);
}

void batched_low_rank_eval_self(const block_low_rank_state_t *batch,
                                const double *d_X, const long long *d_X_offsets,
                                double *d_out_constraints,
                                double *d_out_objective, cudaStream_t stream) {
  if (!batch || !batch->bdata || batch->bdata->total_lr_columns <= 0)
    return;
  int warps_per_block = 8;
  int blocks =
      (batch->bdata->total_lr_columns + warps_per_block - 1) / warps_per_block;
  batched_low_rank_eval_kernel<true><<<blocks, 256, 0, stream>>>(
      batch->bdata->total_lr_columns, batch->dim,
      batch->bdata->d_lr_column_cone, batch->bdata->d_lr_cone_offsets,
      batch->bdata->d_lr_constraints, batch->bdata->d_lr_weights,
      batch->bdata->d_lr_factors, batch->bdata->d_ranks, d_X, d_X_offsets, d_X,
      d_X_offsets, d_out_constraints, d_out_objective);
}

void batched_low_rank_eval_cross(const block_low_rank_state_t *batch,
                                 const double *d_X,
                                 const long long *d_X_offsets,
                                 const double *d_Y,
                                 const long long *d_Y_offsets,
                                 double *d_out_constraints,
                                 double *d_out_objective, cudaStream_t stream) {
  if (!batch || !batch->bdata || batch->bdata->total_lr_columns <= 0)
    return;
  int warps_per_block = 8;
  int blocks =
      (batch->bdata->total_lr_columns + warps_per_block - 1) / warps_per_block;
  batched_low_rank_eval_kernel<false><<<blocks, 256, 0, stream>>>(
      batch->bdata->total_lr_columns, batch->dim,
      batch->bdata->d_lr_column_cone, batch->bdata->d_lr_cone_offsets,
      batch->bdata->d_lr_constraints, batch->bdata->d_lr_weights,
      batch->bdata->d_lr_factors, batch->bdata->d_ranks, d_X, d_X_offsets, d_Y,
      d_Y_offsets, d_out_constraints, d_out_objective);
}

void batched_low_rank_add_operator_times_matrix(
    const block_low_rank_state_t *batch, const double *d_q,
    int include_objective, const double *d_X, const long long *d_X_offsets,
    double alpha, double *d_out, const long long *d_out_offsets,
    cudaStream_t stream) {
  if (!batch || !batch->bdata || batch->bdata->total_lr_columns <= 0 ||
      batch->bdata->max_rank <= 0)
    return;
  int warps_per_block = 8;
  dim3 grid(batch->n_cones,
            (batch->bdata->max_rank + warps_per_block - 1) / warps_per_block);
  batched_low_rank_operator_kernel<<<grid, 256, 0, stream>>>(
      batch->n_cones, batch->dim, batch->bdata->max_rank,
      batch->bdata->d_lr_cone_offsets, batch->bdata->d_lr_constraints,
      batch->bdata->d_lr_weights, batch->bdata->d_lr_factors,
      batch->bdata->d_ranks, d_q, include_objective, d_X, d_X_offsets, alpha,
      d_out, d_out_offsets);
}
