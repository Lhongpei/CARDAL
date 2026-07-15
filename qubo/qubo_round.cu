/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "qubo.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__global__ void k_round_signs(int n, int T,
                              const int *__restrict__ home_cone,
                              const int *__restrict__ home_lidx,
                              const int *__restrict__ blk_dims,
                              const long long *__restrict__ P_off,
                              const double *__restrict__ P,
                              char *__restrict__ X) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y;
  if (t >= T || i >= n)
    return;
  int b = home_cone[i];
  int l = home_lidx[i];
  int ld = blk_dims[b];
  long long base = P_off[b] + (long long)t * ld;
  double pl = P[base + l];
  double p0 = P[base];
  X[(long long)t * n + i] = ((pl >= 0.0) == (p0 >= 0.0)) ? (char)1 : (char)0;
}

__global__ void k_threshold_round(int n,
                                  const int *__restrict__ home_cone,
                                  const int *__restrict__ home_lidx,
                                  const int *__restrict__ blk_dims,
                                  const int *__restrict__ rank_list,
                                  const long long *__restrict__ R_off,
                                  const double *__restrict__ R,
                                  char *__restrict__ X_row) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int b = home_cone[i];
  int l = home_lidx[i];
  int n_b = blk_dims[b];
  int r_b = rank_list[b];
  long long base = R_off[b];
  double sum = 0.0;
  for (int c = 0; c < r_b; c++) {
    sum += R[base + (long long)c * n_b + 0] *
           R[base + (long long)c * n_b + l];
  }
  X_row[i] = (sum >= 0.5) ? (char)1 : (char)0;
}

__global__ void k_local_search_1flip(int n, int max_iters,
                                     const int *__restrict__ q_rowptr,
                                     const int *__restrict__ q_col,
                                     const double *__restrict__ q_val,
                                     const double *__restrict__ q_diag,
                                     char *__restrict__ X) {
  __shared__ double s_delta[256];
  __shared__ int s_idx[256];

  int t = blockIdx.x;
  char *x = X + (long long)t * n;
  int tid = threadIdx.x;
  int bsz = blockDim.x;

  for (int iter = 0; iter < max_iters; iter++) {
    double tb_delta = 0.0;
    int tb_idx = -1;
    for (int i = tid; i < n; i += bsz) {
      int xi = (int)x[i];
      int si = 1 - 2 * xi;
      double cross = 0.0;
      int rs = q_rowptr[i], re = q_rowptr[i + 1];
      for (int k = rs; k < re; k++) {
        cross += q_val[k] * (double)(int)x[q_col[k]];
      }
      double delta = (double)si * (q_diag[i] + 2.0 * cross);
      if (delta < tb_delta) {
        tb_delta = delta;
        tb_idx = i;
      }
    }
    s_delta[tid] = tb_delta;
    s_idx[tid] = tb_idx;
    __syncthreads();
    for (int off = bsz >> 1; off > 0; off >>= 1) {
      if (tid < off) {
        if (s_delta[tid + off] < s_delta[tid]) {
          s_delta[tid] = s_delta[tid + off];
          s_idx[tid] = s_idx[tid + off];
        }
      }
      __syncthreads();
    }
    if (s_delta[0] >= 0.0 || s_idx[0] < 0)
      break;
    if (tid == 0)
      x[s_idx[0]] = (char)(1 - (int)x[s_idx[0]]);
    __syncthreads();
  }
}

// Tabu search (Glover-style). One block per trial. Block size MUST be 256.
// At each iter:
//   - Compute Δ_i for every var i (same as 1-flip).
//   - Pick the var with the smallest Δ subject to:
//       * var not currently tabu (tabu_until[i] < iter), OR
//       * "aspiration": flipping it strictly improves the best-so-far obj.
//   - Flip that var (even when Δ >= 0, so we walk out of local minima).
//   - Mark the var tabu for `tabu_tenure` iters.
// We keep a snapshot of the best-x seen so far per trial; on exit, the trial's
// X row is restored to that best. `obj_init[t]` is f(x_init[t]) coming from
// the post-rounding eval.
__global__ void k_local_search_tabu(int n, int max_iters, int tabu_tenure,
                                    int stall_limit,
                                    const int *__restrict__ q_rowptr,
                                    const int *__restrict__ q_col,
                                    const double *__restrict__ q_val,
                                    const double *__restrict__ q_diag,
                                    char *__restrict__ X,
                                    char *__restrict__ best_X,
                                    int *__restrict__ tabu_until,
                                    const double *__restrict__ obj_init) {
  __shared__ double s_delta[256];
  __shared__ int s_idx[256];
  __shared__ double s_curr;
  __shared__ double s_best;
  __shared__ int s_no_improve;
  __shared__ int s_improved;

  int t = blockIdx.x;
  int tid = threadIdx.x;
  int bsz = blockDim.x;
  char *x = X + (long long)t * n;
  char *bx = best_X + (long long)t * n;
  int *tu = tabu_until + (long long)t * n;

  for (int i = tid; i < n; i += bsz) {
    bx[i] = x[i];
    tu[i] = -1;
  }
  if (tid == 0) {
    s_curr = obj_init[t];
    s_best = s_curr;
    s_no_improve = 0;
  }
  __syncthreads();

  for (int iter = 0; iter < max_iters; iter++) {
    double tb_delta = INFINITY;
    int tb_idx = -1;
    for (int i = tid; i < n; i += bsz) {
      int xi = (int)x[i];
      int si = 1 - 2 * xi;
      double cross = 0.0;
      int rs = q_rowptr[i], re = q_rowptr[i + 1];
      for (int k = rs; k < re; k++)
        cross += q_val[k] * (double)(int)x[q_col[k]];
      double delta = (double)si * (q_diag[i] + 2.0 * cross);
      bool is_tabu = (tu[i] >= iter);
      bool aspire = (s_curr + delta < s_best);
      if (is_tabu && !aspire)
        continue;
      if (delta < tb_delta) {
        tb_delta = delta;
        tb_idx = i;
      }
    }
    s_delta[tid] = tb_delta;
    s_idx[tid] = tb_idx;
    __syncthreads();
    for (int off = bsz >> 1; off > 0; off >>= 1) {
      if (tid < off) {
        if (s_delta[tid + off] < s_delta[tid]) {
          s_delta[tid] = s_delta[tid + off];
          s_idx[tid] = s_idx[tid + off];
        }
      }
      __syncthreads();
    }
    int idx_best = s_idx[0];
    if (idx_best < 0)
      break;
    double d_best = s_delta[0];
    if (tid == 0) {
      x[idx_best] = (char)(1 - (int)x[idx_best]);
      tu[idx_best] = iter + tabu_tenure;
      s_curr += d_best;
      if (s_curr < s_best) {
        s_best = s_curr;
        s_no_improve = 0;
        s_improved = 1;
      } else {
        s_no_improve += 1;
        s_improved = 0;
      }
    }
    __syncthreads();
    if (s_improved) {
      for (int i = tid; i < n; i += bsz)
        bx[i] = x[i];
    }
    __syncthreads();
    if (s_no_improve >= stall_limit)
      break;
  }

  for (int i = tid; i < n; i += bsz)
    x[i] = bx[i];
}

__global__ void k_eval_obj(int n, int T, int nnz,
                           const int *__restrict__ q_row,
                           const int *__restrict__ q_col,
                           const double *__restrict__ q_val,
                           const double *__restrict__ linear,
                           const char *__restrict__ X,
                           double *__restrict__ out_obj) {
  __shared__ double s_warp[8];
  int t = blockIdx.x;
  if (t >= T)
    return;
  int tid = threadIdx.x;
  const char *xt = X + (long long)t * n;
  double sum = 0.0;

  for (int k = tid; k < nnz; k += blockDim.x) {
    int i = q_row[k], j = q_col[k];
    double v = q_val[k];
    int xi = (int)xt[i];
    if (i == j) {
      sum += v * (double)xi;
    } else {
      int xj = (int)xt[j];
      sum += 2.0 * v * (double)(xi * xj);
    }
  }
  if (linear != NULL) {
    for (int i = tid; i < n; i += blockDim.x) {
      sum += linear[i] * (double)xt[i];
    }
  }

  unsigned mask = 0xFFFFFFFFu;
  for (int off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(mask, sum, off);
  int warp_id = tid >> 5;
  int lane = tid & 31;
  if (lane == 0)
    s_warp[warp_id] = sum;
  __syncthreads();

  if (warp_id == 0) {
    sum = (lane < 8) ? s_warp[lane] : 0.0;
    for (int off = 4; off > 0; off >>= 1)
      sum += __shfl_down_sync(mask, sum, off);
    if (lane == 0)
      out_obj[t] = sum;
  }
}

extern "C" void free_qubo_round_result(qubo_round_result_t *r) {
  if (!r)
    return;
  free(r->x);
  free(r);
}

extern "C" qubo_round_result_t *
qubo_round_gpu(const qubo_problem_t *q, const qubo_layout_t *layout,
               const int *blk_dims, const int *rank_list,
               const double *R_host, long long R_length, int num_trials,
               int max_ls_iters, uint64_t seed) {
  if (!q || !layout || !blk_dims || !rank_list || !R_host ||
      num_trials <= 0) {
    fprintf(stderr, "qubo_round_gpu: invalid args\n");
    return NULL;
  }
  if (q->n != layout->n) {
    fprintf(stderr, "qubo_round_gpu: q->n (%d) != layout->n (%d)\n", q->n,
            layout->n);
    return NULL;
  }
  int n = q->n;
  int K = layout->n_cones;
  int T = num_trials;

  clock_t t_start = clock();

  long long *h_R_off = (long long *)safe_malloc((K + 1) * sizeof(long long));
  long long *h_P_off = (long long *)safe_malloc((K + 1) * sizeof(long long));
  long long *h_G_off = (long long *)safe_malloc((K + 1) * sizeof(long long));
  h_R_off[0] = h_P_off[0] = h_G_off[0] = 0;
  for (int b = 0; b < K; b++) {
    h_R_off[b + 1] = h_R_off[b] + (long long)blk_dims[b] * rank_list[b];
    h_P_off[b + 1] = h_P_off[b] + (long long)blk_dims[b] * T;
    h_G_off[b + 1] = h_G_off[b] + (long long)rank_list[b] * T;
  }
  long long total_R = h_R_off[K];
  long long total_P = h_P_off[K];
  long long total_G = h_G_off[K];
  long long g_alloc = total_G + (total_G & 1LL);

  if (total_R > R_length) {
    fprintf(stderr,
            "qubo_round_gpu: total_R (%lld) > R_length (%lld) - cone/rank "
            "mismatch?\n",
            total_R, R_length);
    free(h_R_off);
    free(h_P_off);
    free(h_G_off);
    return NULL;
  }

  double *d_R = NULL, *d_G = NULL, *d_P = NULL, *d_linear = NULL;
  int *d_home_cone = NULL, *d_home_lidx = NULL, *d_blk_dims = NULL;
  int *d_rank_list = NULL;
  long long *d_P_off = NULL, *d_R_off = NULL;
  char *d_X = NULL;
  double *d_obj = NULL;
  int *d_q_row = NULL, *d_q_col = NULL;
  double *d_q_val = NULL;

  int T_total = T + 1; // T random hyperplanes + 1 deterministic threshold

  CUDA_CHECK(cudaMalloc(&d_R, total_R * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_R, R_host, total_R * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_G, g_alloc * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_P, total_P * sizeof(double)));

  CUDA_CHECK(cudaMalloc(&d_home_cone, (size_t)n * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_home_cone, layout->home_cone, n * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_home_lidx, (size_t)n * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_home_lidx, layout->home_lidx, n * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_blk_dims, (size_t)K * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_blk_dims, blk_dims, K * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_P_off, (size_t)(K + 1) * sizeof(long long)));
  CUDA_CHECK(cudaMemcpy(d_P_off, h_P_off, (K + 1) * sizeof(long long),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_R_off, (size_t)(K + 1) * sizeof(long long)));
  CUDA_CHECK(cudaMemcpy(d_R_off, h_R_off, (K + 1) * sizeof(long long),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_rank_list, (size_t)K * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_rank_list, rank_list, K * sizeof(int),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_X, (size_t)T_total * n * sizeof(char)));
  CUDA_CHECK(cudaMalloc(&d_obj, (size_t)T_total * sizeof(double)));

  CUDA_CHECK(cudaMalloc(&d_q_row, (size_t)q->nnz * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_q_row, q->row, q->nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_q_col, (size_t)q->nnz * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_q_col, q->col, q->nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_q_val, (size_t)q->nnz * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_q_val, q->val, q->nnz * sizeof(double),
                        cudaMemcpyHostToDevice));

  if (q->linear) {
    CUDA_CHECK(cudaMalloc(&d_linear, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_linear, q->linear, n * sizeof(double),
                          cudaMemcpyHostToDevice));
  }

  curandGenerator_t gen;
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
      gen, (unsigned long long)(seed ? seed : 0xC0FFEE15600D7EAULL)));
  CURAND_CHECK(curandGenerateNormalDouble(gen, d_G, (size_t)g_alloc, 0.0, 1.0));
  CURAND_CHECK(curandDestroyGenerator(gen));

  cublasHandle_t blas;
  CUBLAS_CHECK(cublasCreate(&blas));
  const double alpha = 1.0, beta = 0.0;
  for (int b = 0; b < K; b++) {
    int n_b = blk_dims[b];
    int r_b = rank_list[b];
    if (r_b == 0)
      continue;
    const double *R_b = d_R + h_R_off[b];
    const double *G_b = d_G + h_G_off[b];
    double *P_b = d_P + h_P_off[b];
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n_b, T, r_b,
                             &alpha, R_b, n_b, G_b, r_b, &beta, P_b, n_b));
  }
  CUBLAS_CHECK(cublasDestroy(blas));

  dim3 round_grid((T + 127) / 128, n);
  k_round_signs<<<round_grid, 128>>>(n, T, d_home_cone, d_home_lidx,
                                     d_blk_dims, d_P_off, d_P, d_X);
  CUDA_CHECK(cudaGetLastError());

  int thr_grid = (n + 127) / 128;
  k_threshold_round<<<thr_grid, 128>>>(n, d_home_cone, d_home_lidx, d_blk_dims,
                                       d_rank_list, d_R_off, d_R,
                                       d_X + (size_t)T * n);
  CUDA_CHECK(cudaGetLastError());

  double *h_qdiag = (double *)safe_calloc((size_t)n, sizeof(double));
  if (q->linear)
    safe_memcpy(h_qdiag, q->linear, (size_t)n * sizeof(double));
  int *h_cnt = (int *)safe_calloc((size_t)n, sizeof(int));
  for (int k = 0; k < q->nnz; k++) {
    int i = q->row[k], j = q->col[k];
    if (i == j) {
      h_qdiag[i] += q->val[k];
    } else if (q->val[k] != 0.0) {
      h_cnt[i]++;
      h_cnt[j]++;
    }
  }
  int *h_rptr = (int *)safe_malloc((size_t)(n + 1) * sizeof(int));
  h_rptr[0] = 0;
  for (int i = 0; i < n; i++)
    h_rptr[i + 1] = h_rptr[i] + h_cnt[i];
  long long off_nnz = h_rptr[n];
  int *h_col_sym = (int *)safe_malloc((size_t)(off_nnz > 0 ? off_nnz : 1) *
                                      sizeof(int));
  double *h_val_sym = (double *)safe_malloc(
      (size_t)(off_nnz > 0 ? off_nnz : 1) * sizeof(double));
  int *h_pos = (int *)safe_calloc((size_t)n, sizeof(int));
  for (int k = 0; k < q->nnz; k++) {
    int i = q->row[k], j = q->col[k];
    if (i == j || q->val[k] == 0.0)
      continue;
    double v = q->val[k];
    int pi = h_rptr[i] + h_pos[i]++;
    h_col_sym[pi] = j;
    h_val_sym[pi] = v;
    int pj = h_rptr[j] + h_pos[j]++;
    h_col_sym[pj] = i;
    h_val_sym[pj] = v;
  }
  free(h_cnt);
  free(h_pos);

  int *d_qsym_rptr = NULL, *d_qsym_col = NULL;
  double *d_qsym_val = NULL, *d_qdiag = NULL;
  CUDA_CHECK(cudaMalloc(&d_qsym_rptr, (size_t)(n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_qsym_rptr, h_rptr, (n + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  if (off_nnz > 0) {
    CUDA_CHECK(cudaMalloc(&d_qsym_col, (size_t)off_nnz * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_qsym_col, h_col_sym, off_nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_qsym_val, (size_t)off_nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_qsym_val, h_val_sym, off_nnz * sizeof(double),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMalloc(&d_qdiag, (size_t)n * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_qdiag, h_qdiag, n * sizeof(double),
                        cudaMemcpyHostToDevice));
  free(h_qdiag);
  free(h_rptr);
  free(h_col_sym);
  free(h_val_sym);

  k_eval_obj<<<T_total, 256>>>(n, T_total, q->nnz, d_q_row, d_q_col, d_q_val,
                               d_linear, d_X, d_obj);
  CUDA_CHECK(cudaGetLastError());
  
  double *h_obj_pre = (double *)safe_malloc((size_t)T_total * sizeof(double));
  CUDA_CHECK(cudaMemcpy(h_obj_pre, d_obj, (size_t)T_total * sizeof(double),
                        cudaMemcpyDeviceToHost));
  double pre_hyperplane = h_obj_pre[0];
  for (int t = 1; t < T; t++)
    if (h_obj_pre[t] < pre_hyperplane)
      pre_hyperplane = h_obj_pre[t];
  double pre_threshold = h_obj_pre[T];
  double pre_round_only =
      (pre_hyperplane < pre_threshold) ? pre_hyperplane : pre_threshold;
  free(h_obj_pre);

  char *d_best_X = NULL;
  int *d_tabu = NULL;
  CUDA_CHECK(cudaMalloc(&d_best_X, (size_t)T_total * n * sizeof(char)));
  CUDA_CHECK(cudaMalloc(&d_tabu, (size_t)T_total * n * sizeof(int)));

  int tabu_tenure = (n / 5 < 8) ? 8 : n / 5;
  // Semantics of max_ls_iters:
  //   < 0 : use built-in default (~20*n).
  //   == 0: skip local search entirely (round-only).
  //   > 0 : cap iterations at this exact value.
  int tabu_max_iters;
  if (max_ls_iters < 0) {
    tabu_max_iters = 20 * n;
    if (tabu_max_iters < 4000)
      tabu_max_iters = 4000;
  } else {
    tabu_max_iters = max_ls_iters;
  }
  int stall_limit = (n < 200) ? 200 : 2 * n;
  if (stall_limit > tabu_max_iters)
    stall_limit = tabu_max_iters;
  k_local_search_tabu<<<T_total, 256>>>(
      n, tabu_max_iters, tabu_tenure, stall_limit, d_qsym_rptr, d_qsym_col,
      d_qsym_val, d_qdiag, d_X, d_best_X, d_tabu, d_obj);
  CUDA_CHECK(cudaGetLastError());

  k_eval_obj<<<T_total, 256>>>(n, T_total, q->nnz, d_q_row, d_q_col, d_q_val,
                               d_linear, d_X, d_obj);
  CUDA_CHECK(cudaGetLastError());

  cudaFree(d_best_X);
  cudaFree(d_tabu);
  cudaFree(d_qsym_rptr);
  if (d_qsym_col)
    cudaFree(d_qsym_col);
  if (d_qsym_val)
    cudaFree(d_qsym_val);
  cudaFree(d_qdiag);

  double *h_obj = (double *)safe_malloc((size_t)T_total * sizeof(double));
  CUDA_CHECK(cudaMemcpy(h_obj, d_obj, (size_t)T_total * sizeof(double),
                        cudaMemcpyDeviceToHost));
  int best_t = 0;
  double best_obj = h_obj[0];
  for (int t = 1; t < T_total; t++) {
    if (h_obj[t] < best_obj) {
      best_obj = h_obj[t];
      best_t = t;
    }
  }

  char *h_x_char = (char *)safe_malloc((size_t)n * sizeof(char));
  CUDA_CHECK(cudaMemcpy(h_x_char, d_X + (size_t)best_t * n,
                        (size_t)n * sizeof(char), cudaMemcpyDeviceToHost));

  qubo_round_result_t *res =
      (qubo_round_result_t *)safe_malloc(sizeof(qubo_round_result_t));
  res->n = n;
  res->x = (int *)safe_malloc((size_t)n * sizeof(int));
  for (int i = 0; i < n; i++)
    res->x[i] = (int)h_x_char[i];
  res->obj = best_obj;
  res->num_trials = T_total;
  res->best_trial = best_t;
  res->obj_hyperplane_pre_ls = pre_hyperplane;
  res->obj_threshold_pre_ls = pre_threshold;
  res->obj_round_only = pre_round_only;

  free(h_x_char);
  free(h_obj);
  free(h_R_off);
  free(h_P_off);
  free(h_G_off);

  cudaFree(d_R);
  cudaFree(d_G);
  cudaFree(d_P);
  cudaFree(d_home_cone);
  cudaFree(d_home_lidx);
  cudaFree(d_blk_dims);
  cudaFree(d_P_off);
  cudaFree(d_R_off);
  cudaFree(d_rank_list);
  cudaFree(d_X);
  cudaFree(d_obj);
  cudaFree(d_q_row);
  cudaFree(d_q_col);
  cudaFree(d_q_val);
  if (d_linear)
    cudaFree(d_linear);

  clock_t t_end = clock();
  res->time_sec = (double)(t_end - t_start) / CLOCKS_PER_SEC;

  LOG_DBG("[QUBO round] T=%d, best obj=%.6e (trial %d), time=%.3fs\n", T,
          res->obj, res->best_trial, res->time_sec);
  return res;
}
