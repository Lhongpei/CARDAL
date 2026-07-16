/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#ifndef SOLVER_CORE_OP_CUH
#define SOLVER_CORE_OP_CUH

#include "batched_cone_ops.h"
#include "solver_state.h"
#include "utils.h"
#include "sdp_op.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

// Helper: Allreduce-SUM a scalar across the Rank axis (comm_rank). Used by
// negative-curvature detection so per-rank dot products / squared norms sum
// to a globally-consistent value when the BM-column dimension is split.
static inline double rank_axis_sum_scalar(cardal_sdp_solver_state_t *state,
                                          double local) {
#ifdef IS_DISTRIBUTED
  if (state->grid_context != NULL && state->grid_context->dims[1] > 1) {
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM,
                  state->grid_context->comm_rank);
    return global;
  }
#else
  (void)state;
#endif
  return local;
}

#ifndef SOLVER_CORE_OP_KERNELS_DEFINED
#define SOLVER_CORE_OP_KERNELS_DEFINED
static __global__ void find_min_array_kernel(const double *arr, double *min_val,
                                             int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    if (arr[idx] < 0) {
      atomicMinDouble(min_val, arr[idx]);
    }
  }
}

// Single-block reduction kernel: *out = sum(in[0..n-1]). Launch <<<1, 256>>>.
// n can be 0..few thousand; we assume n_blks <= 1<<20.
static __global__ void sum_to_scalar_kernel(double *__restrict__ out,
                                            const double *__restrict__ in,
                                            int n) {
  __shared__ double sdata[256];
  int tid = threadIdx.x;
  double v = 0.0;
  for (int i = tid; i < n; i += blockDim.x)
    v += in[i];
  sdata[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    *out = sdata[0];
}

// Sum into *out: *out += sum(in[0..n-1])
static __global__ void sum_into_scalar_kernel(double *__restrict__ out,
                                              const double *__restrict__ in,
                                              int n) {
  __shared__ double sdata[256];
  int tid = threadIdx.x;
  double v = 0.0;
  for (int i = tid; i < n; i += blockDim.x)
    v += in[i];
  sdata[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    *out += sdata[0];
}

static __device__ double cubic_function_val(double a, double b, double c,
                                            double d, double x) {
  return a * x * x * x * x + b * x * x * x + c * x * x + d * x;
}

static __global__ void
solve_line_search_tau_kernel(double *__restrict__ scalars, double penalty,
                             int tau_slot, int tau_sq_slot,
                             double tau_max_in) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  double q2q2 = scalars[0];
  double q1q2 = scalars[1];
  double q1q1 = scalars[2];
  double yq2 = scalars[3];
  double q0q2 = scalars[4];
  double d_coef = scalars[5];
  double p2 = scalars[6] + scalars[7];

  double a = 0.5 * penalty * q2q2;
  double b = penalty * q1q2;
  double c = 0.5 * penalty * q1q1 + p2 + yq2 + penalty * q0q2;
  double dd = d_coef;

  // Solve 4a x^3 + 3b x^2 + 2c x + d = 0 (Shengjin formula). roots[3].
  double A = (3.0 * b) * (3.0 * b) - 3.0 * (4.0 * a) * (2.0 * c);
  double B = (3.0 * b) * (2.0 * c) - 9.0 * (4.0 * a) * dd;
  double C = (2.0 * c) * (2.0 * c) - 3.0 * (3.0 * b) * dd;
  double delta = B * B - 4.0 * A * C;

  double roots[3] = {0.0, 0.0, 0.0};
  int num_roots = 0;
  double aa = 4.0 * a, bb = 3.0 * b, cc = 2.0 * c;
  if (A == 0.0 && B == 0.0) {
    if (bb != 0.0) {
      roots[0] = -cc / bb;
      num_roots = 1;
    }
  } else if (delta > 0.0) {
    double Y1 = A * bb + 1.5 * aa * (-B + sqrt(delta));
    double Y2 = A * bb + 1.5 * aa * (-B - sqrt(delta));
    roots[0] = (-bb - cbrt(Y1) - cbrt(Y2)) / (3.0 * aa);
    num_roots = 1;
  } else if (delta == 0.0 && A != 0.0 && B != 0.0) {
    double K = B / A;
    roots[0] = -bb / aa + K;
    roots[1] = -K / 2.0;
    num_roots = 2;
  } else if (delta < 0.0) {
    double sqA = sqrt(A);
    double T = (A * bb - 1.5 * aa * B) / (A * sqA);
    if (T > 1.0)
      T = 1.0;
    if (T < -1.0)
      T = -1.0;
    double theta = acos(T);
    double csth = cos(theta / 3.0);
    double sn3th = sqrt(3.0) * sin(theta / 3.0);
    roots[0] = (-bb - 2.0 * sqA * csth) / (3.0 * aa);
    roots[1] = (-bb + sqA * (csth + sn3th)) / (3.0 * aa);
    roots[2] = (-bb + sqA * (csth - sn3th)) / (3.0 * aa);
    num_roots = 3;
  }

  double tau_max = tau_max_in > 0.0 ? tau_max_in : 1.0;
  double f0 = cubic_function_val(a, b, c, dd, 0.0);
  double f1 = cubic_function_val(a, b, c, dd, tau_max);
  double best_tau = 0.0;
  double min_fval = f0;
  if (f1 < min_fval) {
    min_fval = f1;
    best_tau = tau_max;
  }
  for (int i = 0; i < num_roots; i++) {
    double r = roots[i];
    if (r > 1e-20 && r <= tau_max) {
      double fr = cubic_function_val(a, b, c, dd, r);
      if (fr < min_fval) {
        min_fval = fr;
        best_tau = r;
      }
    }
  }
  if (best_tau < 1e-10) {
    best_tau = 1.0;
  }
  scalars[tau_slot] = best_tau;
  scalars[tau_sq_slot] = best_tau * best_tau;
}

#endif // SOLVER_CORE_OP_KERNELS_DEFINED

static inline double COMPUTE_LP_MIN_SLACK(cardal_sdp_solver_state_t *state) {
  if (state->lp_dim <= 0)
    return 1e9;

  double *d_lp_slack = state->lp_slack_buffer;

  CUBLAS_CHECK(cublasDcopy(state->blas_handle, state->lp_dim,
                           state->lp_objective_vector, 1, d_lp_slack, 1));

  double alpha = 1.0;
  CUBLAS_CHECK(cublasDaxpy(state->blas_handle, state->lp_dim, &alpha,
                           state->dual_product + state->lp_start_active_idx, 1,
                           d_lp_slack, 1));

#ifdef IS_DISTRIBUTED
  if (state->grid_context != NULL && state->grid_context->dims[0] > 1) {
    NCCL_CHECK(ncclAllReduce(d_lp_slack, d_lp_slack, state->lp_dim, ncclDouble,
                             ncclSum, state->grid_context->nccl_row, 0));
  }
#endif

  if (state->lp_variable_rescaling != NULL) {
    int blocks_s =
        (state->lp_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t stream;
    CUBLAS_CHECK(cublasGetStream(state->blas_handle, &stream));
    double inv_obj = (state->objective_vector_rescaling > 0.0) ? (1.0 / state->objective_vector_rescaling) : 1.0;
    elementwise_multiply_scaled_kernel<<<blocks_s, THREADS_PER_BLOCK, 0, stream>>>(
        d_lp_slack, state->lp_variable_rescaling, inv_obj, state->lp_dim);
  }

  double h_min_slack = 1e9;
  double *d_min_slack = state->lp_min_slack_buf;
  CUDA_CHECK(cudaMemcpyAsync(d_min_slack, &h_min_slack, sizeof(double),
                             cudaMemcpyHostToDevice, 0));

  int blocks = (state->lp_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  find_min_array_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_lp_slack, d_min_slack,
                                                       state->lp_dim);

  CUDA_CHECK(cudaMemcpy(&h_min_slack, d_min_slack, sizeof(double),
                        cudaMemcpyDeviceToHost));

  return h_min_slack;
}

static inline int
COMPUTE_NEGATIVE_EIGEN_POWER_ITER(cardal_sdp_solver_state_t *state, int blk_idx,
                                  double *out_min_eigval,
                                  double **out_neg_eigenvector) {
  block_low_rank_state_t *blk = state->block_low_rank_state[blk_idx];
  int dim = blk->dim;
  int MAX_ITER_PHASE2 = 500;

  double *d_q, *d_Sq, *d_v;
  CUDA_CHECK(cudaMalloc(&d_q, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_Sq, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_v, dim * sizeof(double)));

  curandGenerator_t local_gen;
  CURAND_CHECK(curandCreateGenerator(&local_gen, CURAND_RNG_PSEUDO_DEFAULT));

  unsigned long long base_seed =
      1234ULL + (unsigned long long)state->num_outer_iteration * 1000003ULL +
      (unsigned long long)blk_idx * 2654435761ULL;
      
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(local_gen, base_seed));
  {
    int rand_len = (dim % 2 == 0) ? dim : dim + 1;
    if (rand_len == dim) {
      CURAND_CHECK(curandGenerateNormalDouble(local_gen, d_q, dim, 0.0, 1.0));
    } else {
      double *d_temp_rand;
      CUDA_CHECK(cudaMalloc(&d_temp_rand, rand_len * sizeof(double)));
      CURAND_CHECK(
          curandGenerateNormalDouble(local_gen, d_temp_rand, rand_len, 0.0, 1.0));
      CUDA_CHECK(cudaMemcpy(d_q, d_temp_rand, dim * sizeof(double),
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaFree(d_temp_rand));
    }
  }

  cusparseDnVecDescr_t vec_q, vec_Sq;
  CUSPARSE_CHECK(cusparseCreateDnVec(&vec_q, dim, d_q, CUDA_R_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vec_Sq, dim, d_Sq, CUDA_R_64F));

  size_t bufferSize = 0;
  void *dBuffer = NULL;
  double spmv_alpha = 1.0, spmv_beta = 0.0;
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha,
      blk->matSpS, vec_q, &spmv_beta, vec_Sq, CUDA_R_64F,
      CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

  double norm_q;
  CUBLAS_CHECK(cublasDnrm2(state->blas_handle, dim, d_q, 1, &norm_q));
  double inv_norm = 1.0 / norm_q;
  CUBLAS_CHECK(cublasDscal(state->blas_handle, dim, &inv_norm, d_q, 1));

  double mu = 0.0;
  double prev_mu = 0.0;
  int phase1_iters = 0;

  for (int i = 0; i < 150; i++) {
    phase1_iters++;

    unscaled_dual_spmv(state, blk, d_q, d_Sq, d_v, vec_q, vec_Sq, dBuffer);

#ifdef IS_DISTRIBUTED
    NCCL_CHECK(ncclAllReduce(d_Sq, d_Sq, dim, ncclDouble, ncclSum,
                             state->grid_context->nccl_row, 0));
#endif

    CUBLAS_CHECK(cublasDdot(state->blas_handle, dim, d_q, 1, d_Sq, 1, &mu));

    if (i > 0 && fabs(mu - prev_mu) < 1e-3 * fabs(mu) + 1e-6) {
      break;
    }
    prev_mu = mu;

    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, dim, d_Sq, 1, &norm_q));
    inv_norm = 1.0 / norm_q;
    CUBLAS_CHECK(cublasDcopy(state->blas_handle, dim, d_Sq, 1, d_q, 1));
    CUBLAS_CHECK(cublasDscal(state->blas_handle, dim, &inv_norm, d_q, 1));
  }

  double shift = fabs(mu) * 1.5 + 1.0;

  {
    int rand_len = (dim % 2 == 0) ? dim : dim + 1;
    if (rand_len == dim) {
      CURAND_CHECK(curandGenerateNormalDouble(local_gen, d_q, dim, 0.0, 1.0));
    } else {
      double *d_temp_rand;
      CUDA_CHECK(cudaMalloc(&d_temp_rand, rand_len * sizeof(double)));
      CURAND_CHECK(
          curandGenerateNormalDouble(local_gen, d_temp_rand, rand_len, 0.0, 1.0));
      CUDA_CHECK(cudaMemcpy(d_q, d_temp_rand, dim * sizeof(double),
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaFree(d_temp_rand));
    }
  }
  curandDestroyGenerator(local_gen);

  CUBLAS_CHECK(cublasDnrm2(state->blas_handle, dim, d_q, 1, &norm_q));
  inv_norm = 1.0 / norm_q;
  CUBLAS_CHECK(cublasDscal(state->blas_handle, dim, &inv_norm, d_q, 1));

  double prev_lambda_B = 0.0;

  for (int iter = 0; iter < MAX_ITER_PHASE2; iter++) {
    unscaled_dual_spmv(state, blk, d_q, d_Sq, d_v, vec_q, vec_Sq, dBuffer);

#ifdef IS_DISTRIBUTED
    NCCL_CHECK(ncclAllReduce(d_Sq, d_Sq, dim, ncclDouble, ncclSum,
                             state->grid_context->nccl_row, 0));
#endif

    // d_v = shift * d_q - d_Sq
    CUBLAS_CHECK(
        cublasDcopy(state->blas_handle, dim, d_q, 1, d_v, 1)); // d_v = d_q
    CUBLAS_CHECK(cublasDscal(state->blas_handle, dim, &shift, d_v,
                             1)); // d_v = shift * d_q
    double minus_one = -1.0;
    CUBLAS_CHECK(cublasDaxpy(state->blas_handle, dim, &minus_one, d_Sq, 1, d_v,
                             1)); // d_v -= d_Sq

    double lambda_B;
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, dim, d_v, 1, &lambda_B));

    if (fabs(lambda_B - prev_lambda_B) < 1e-6) {
      break;
    }
    prev_lambda_B = lambda_B;

    inv_norm = 1.0 / lambda_B;
    CUBLAS_CHECK(cublasDcopy(state->blas_handle, dim, d_v, 1, d_q, 1));
    CUBLAS_CHECK(cublasDscal(state->blas_handle, dim, &inv_norm, d_q, 1));
  }

  double min_eigval = 0.0;
  unscaled_dual_spmv(state, blk, d_q, d_Sq, d_v, vec_q, vec_Sq, dBuffer);

#ifdef IS_DISTRIBUTED
  NCCL_CHECK(ncclAllReduce(d_Sq, d_Sq, dim, ncclDouble, ncclSum,
                           state->grid_context->nccl_row, 0));
#endif

  CUBLAS_CHECK(
      cublasDdot(state->blas_handle, dim, d_q, 1, d_Sq, 1, &min_eigval));

  if (state->objective_vector_rescaling > 0.0)
    min_eigval /= state->objective_vector_rescaling;

  if (out_min_eigval != NULL) {
    *out_min_eigval = min_eigval;
  }

  int neg_count = 0;
  double threshold = -1e-4;

  if (min_eigval < threshold) {
    neg_count = 1;

    double scale = sqrt(fabs(min_eigval));
    double *d_U_neg;
    CUDA_CHECK(cudaMalloc(&d_U_neg, dim * sizeof(double)));

    CUBLAS_CHECK(cublasDcopy(state->blas_handle, dim, d_q, 1, d_U_neg, 1));
    CUBLAS_CHECK(cublasDscal(state->blas_handle, dim, &scale, d_U_neg, 1));

    *out_neg_eigenvector = d_U_neg;
  } else {
    *out_neg_eigenvector = NULL;
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_Sq));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(dBuffer));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vec_q));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vec_Sq));

  return neg_count;
}


static inline void
COMPUTE_BM_HVP_PERCONE(cardal_sdp_solver_state_t *state,
                       block_low_rank_state_t *blk,
                       cusparseDnMatDescr_t matV_descr,
                       cusparseDnMatDescr_t matHV_descr,
                       double *d_HV,
                       double *d_saved_S_val) {
#ifndef IS_DISTRIBUTED
  (void)d_HV;  // Only referenced from nccl_row/nccl_rank Allreduce calls
               // under IS_DISTRIBUTED; unused in single-GPU builds.
#endif
  int nnz_A = blk->constraint_sparse_pattern->num_nonzeros;
  int nnz_Union =
      blk->objective_union_constraint_sparse_pattern->num_nonzeros;
  double rho = state->penalty_coef;

  CUDA_CHECK(cudaMemcpyAsync(
      d_saved_S_val, blk->objective_union_constraint_sparse_pattern->val,
      nnz_Union * sizeof(double), cudaMemcpyDeviceToDevice, 0));


  double t1_alpha = 2.0, t1_beta = 0.0;
  CUSPARSE_CHECK(cusparseSpMM(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &t1_alpha, blk->matSpS, matV_descr,
      &t1_beta, matHV_descr, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
      blk->spmm_buffer_S));

  if (nnz_A == 0) {
    CUDA_CHECK(cudaMemcpyAsync(
        blk->objective_union_constraint_sparse_pattern->val, d_saved_S_val,
        nnz_Union * sizeof(double), cudaMemcpyDeviceToDevice, 0));
#ifdef IS_DISTRIBUTED
    if (state->grid_context != NULL && state->grid_context->dims[0] > 1) {
      NCCL_CHECK(ncclAllReduce(d_HV, d_HV,
                               (size_t)blk->dim * (size_t)blk->rank,
                               ncclDouble, ncclSum,
                               state->grid_context->nccl_row, 0));
    }
#endif
    return;
  }

  // (a) SDDMM(R, V^T) -> matSpA->val[(i,j)] = <R[i,:], V[j,:]>
  double sd_a = 1.0, sd_b = 0.0;
  CUSPARSE_CHECK(cusparseSDDMM(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &sd_a, blk->matR, matV_descr, &sd_b,
      blk->matSpA, CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT,
      blk->sddmm_buffer_A));
  // (b) accumulate SDDMM(V, R^T) -> matSpA = (R V^T + V R^T) at A's pattern.
  double sd_b1 = 1.0;
  CUSPARSE_CHECK(cusparseSDDMM(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &sd_a, matV_descr, blk->matR, &sd_b1,
      blk->matSpA, CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT,
      blk->sddmm_buffer_A));

#ifdef IS_DISTRIBUTED
  // Rank-axis: each peer holds slab R^(j), V^(j); the two SDDMMs above
  // produced R^(j) V^(j)^T + V^(j) R^(j)^T at A's pattern. The true global D
  // = sum_j (R^(j) V^(j)^T + V^(j) R^(j)^T) requires SUM over rank-axis peers
  // before scatter into the active-var domain. Without this, every downstream
  // step (q1, dual_product, matSpS_pert, Term-2 SpMM) uses a per-slab partial.
  // matSpA wraps blk->constraint_sparse_pattern->val (see solver_state.cu).
  // TODO: optimize by Allreducing the final dim*k_local HV instead of nnz_A
  // matSpA (smaller comm volume) — requires Allgather R first.
  if (state->grid_context != NULL && state->grid_context->dims[1] > 1) {
    NCCL_CHECK(ncclAllReduce(blk->constraint_sparse_pattern->val,
                             blk->constraint_sparse_pattern->val,
                             (size_t)nnz_A, ncclDouble, ncclSum,
                             state->grid_context->nccl_rank, 0));
  }
#endif

  // (c) scatter matSpA->val into primal_solution
  CUDA_CHECK(cudaMemsetAsync(state->primal_solution, 0,
                             state->n_active_vars * sizeof(double), 0));
  int blocks_A = (nnz_A + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  scatter_sddmm_to_global_kernel<<<blocks_A, THREADS_PER_BLOCK>>>(
      blk->constraint_sparse_pattern->val, blk->compat_mapping,
      state->primal_solution, nnz_A);

  // (d) q1 = matA_local * primal_solution (local m-slice of A * D)
  double spmv_a = 1.0, spmv_b = 0.0;
  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_primal_sol, state->primal_solution));
  CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_q1, state->q1));
  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_a,
      state->matA, state->vec_primal_sol, &spmv_b, state->vec_q1, CUDA_R_64F,
      CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

  // (e) dual_product = matAt_local * q1_local (partial across n_active)
  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_a,
      state->matAt, state->vec_q1, &spmv_b, state->vec_dual_prod, CUDA_R_64F,
      CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));


  // (f) matSpS_pert = add_Ay(dual_product_full) at LOCAL compat_mapping.
  CUDA_CHECK(cudaMemsetAsync(
      blk->objective_union_constraint_sparse_pattern->val, 0,
      nnz_Union * sizeof(double), 0));
  add_Ay_to_S_kernel<<<blocks_A, THREADS_PER_BLOCK>>>(
      blk->objective_union_constraint_sparse_pattern->val,
      state->dual_product, blk->compat_mapping,
      blk->constraint_to_union_mapping, nnz_A);


  // (g) HV += 2 * rho * matSpS_pert_full * R (full Term 2 per rank).
  double t2_alpha = 2.0 * rho, t2_beta = 1.0;
  CUSPARSE_CHECK(cusparseSpMM(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &t2_alpha, blk->matSpS, blk->matR,
      &t2_beta, matHV_descr, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
      blk->spmm_buffer_S));

  // (h) restore matSpS partial.
  CUDA_CHECK(cudaMemcpyAsync(
      blk->objective_union_constraint_sparse_pattern->val, d_saved_S_val,
      nnz_Union * sizeof(double), cudaMemcpyDeviceToDevice, 0));

#ifdef IS_DISTRIBUTED
  // Sum row-partial matHV across nccl_row. Uniform size = dim × rank.
  if (state->grid_context != NULL && state->grid_context->dims[0] > 1) {
    NCCL_CHECK(ncclAllReduce(d_HV, d_HV,
                             (size_t)blk->dim * (size_t)blk->rank,
                             ncclDouble, ncclSum,
                             state->grid_context->nccl_row, 0));
  }
#endif
}

// Lanczos on BM HVP; out_V_neg owned by caller if non-NULL. PERCONE only.
#ifndef LANCZOS_K
#define LANCZOS_K 30
#endif
#ifndef NEGATIVE_CURVATURE_LANCZOS_K
#define NEGATIVE_CURVATURE_LANCZOS_K 15
#endif
static inline int
FIND_AL_NEGATIVE_CURVATURE(cardal_sdp_solver_state_t *state, int blk_idx,
                          double *out_min_eigval, double **out_V_neg) {
  *out_min_eigval = 0.0;
  *out_V_neg = NULL;

  block_low_rank_state_t *blk = state->block_low_rank_state[blk_idx];
  if (blk->kind == CONE_BATCH_KIND_CUSTOM) return 0;  // batched: skip
  int dim = blk->dim;
  int r = blk->rank;
  if (r <= 0 || dim <= 0) return 0;

  size_t N = (size_t)dim * (size_t)r;
  int k_target = (N < (size_t)NEGATIVE_CURVATURE_LANCZOS_K)
                     ? (int)N
                     : NEGATIVE_CURVATURE_LANCZOS_K;
  int k = k_target;

  int nnz_Union =
      blk->objective_union_constraint_sparse_pattern->num_nonzeros;

  double *d_Q = NULL, *d_v = NULL, *d_saved_S = NULL;
  CUDA_CHECK(cudaMalloc(&d_Q, N * (size_t)k * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_saved_S, (size_t)nnz_Union * sizeof(double)));

  double *alpha_arr = (double *)safe_malloc(k * sizeof(double));
  double *beta_arr = (double *)safe_malloc(k * sizeof(double));

  // Init random q_0. Rank-axis peers must use DIFFERENT seeds so the per-rank
  // slabs together form a single random vector in the full BM column space
  // (dim x total_rank). Each peer's slab is its share; the global initial
  // vector norm is assembled via Allreduce below.
  unsigned long long q0_seed = 5678ULL + (unsigned long long)blk_idx;
#ifdef IS_DISTRIBUTED
  if (state->grid_context != NULL) {
    q0_seed += (unsigned long long)state->grid_context->coords[1] * 7919ULL;
  }
#endif
  curandGenerator_t gen;
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, q0_seed));
  size_t rand_len = (N % 2 == 0) ? N : (N + 1);
  if (rand_len <= N * (size_t)k) {
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_Q, rand_len, 0.0, 1.0));
  } else {
    double *tmp;
    CUDA_CHECK(cudaMalloc(&tmp, rand_len * sizeof(double)));
    CURAND_CHECK(curandGenerateNormalDouble(gen, tmp, rand_len, 0.0, 1.0));
    CUDA_CHECK(cudaMemcpy(d_Q, tmp, N * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(tmp));
  }
  curandDestroyGenerator(gen);

  double q0_local_nrm = 0.0;
  CUBLAS_CHECK(cublasDnrm2(state->blas_handle, (int)N, d_Q, 1, &q0_local_nrm));
  double q0_nrm =
      sqrt(rank_axis_sum_scalar(state, q0_local_nrm * q0_local_nrm));
  if (q0_nrm < 1e-30) {
    free(alpha_arr); free(beta_arr);
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_saved_S));
    return 0;
  }
  double inv0 = 1.0 / q0_nrm;
  CUBLAS_CHECK(cublasDscal(state->blas_handle, (int)N, &inv0, d_Q, 1));

  beta_arr[0] = 0.0;

  // Reusable Dense descriptors that we re-point each Lanczos step.
  cusparseDnMatDescr_t matV_descr, matHV_descr;
  CUSPARSE_CHECK(cusparseCreateDnMat(&matV_descr, dim, r, dim, d_Q,
                                     CUDA_R_64F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&matHV_descr, dim, r, dim, d_v,
                                     CUDA_R_64F, CUSPARSE_ORDER_COL));

  for (int j = 0; j < k; j++) {
    double *q_j = d_Q + (size_t)j * N;
    CUSPARSE_CHECK(cusparseDnMatSetValues(matV_descr, q_j));
    CUSPARSE_CHECK(cusparseDnMatSetValues(matHV_descr, d_v));

    // d_v = H[q_j]
    COMPUTE_BM_HVP_PERCONE(state, blk, matV_descr, matHV_descr, d_v,
                           d_saved_S);

    if (j > 0) {
      double mb = -beta_arr[j];
      double *qprev = d_Q + (size_t)(j - 1) * N;
      CUBLAS_CHECK(
          cublasDaxpy(state->blas_handle, (int)N, &mb, qprev, 1, d_v, 1));
    }
    double alpha_local = 0.0;
    CUBLAS_CHECK(cublasDdot(state->blas_handle, (int)N, q_j, 1, d_v, 1,
                            &alpha_local));
    alpha_arr[j] = rank_axis_sum_scalar(state, alpha_local);
    double ma = -alpha_arr[j];
    CUBLAS_CHECK(
        cublasDaxpy(state->blas_handle, (int)N, &ma, q_j, 1, d_v, 1));

    // Full reorth against all previous q_i. proj must be a global dot across
    // rank-axis peers so every peer subtracts the same multiple of q_i.
    for (int i = 0; i <= j; i++) {
      double *q_i = d_Q + (size_t)i * N;
      double proj_local = 0.0;
      CUBLAS_CHECK(
          cublasDdot(state->blas_handle, (int)N, q_i, 1, d_v, 1, &proj_local));
      double proj = rank_axis_sum_scalar(state, proj_local);
      double mp = -proj;
      CUBLAS_CHECK(
          cublasDaxpy(state->blas_handle, (int)N, &mp, q_i, 1, d_v, 1));
    }

    if (j < k - 1) {
      double beta_local = 0.0;
      CUBLAS_CHECK(
          cublasDnrm2(state->blas_handle, (int)N, d_v, 1, &beta_local));
      beta_arr[j + 1] =
          sqrt(rank_axis_sum_scalar(state, beta_local * beta_local));
      if (beta_arr[j + 1] < 1e-12) {
        k = j + 1;
        break;
      }
      double *q_next = d_Q + (size_t)(j + 1) * N;
      CUBLAS_CHECK(
          cublasDcopy(state->blas_handle, (int)N, d_v, 1, q_next, 1));
      double inv_b = 1.0 / beta_arr[j + 1];
      CUBLAS_CHECK(
          cublasDscal(state->blas_handle, (int)N, &inv_b, q_next, 1));
    }
  }

  // Solve tridiagonal eigenproblem
  double *d_diag = (double *)safe_malloc(k * sizeof(double));
  double *d_off = (double *)safe_malloc(k * sizeof(double));
  for (int i = 0; i < k; i++) d_diag[i] = alpha_arr[i];
  d_off[0] = 0.0;
  for (int i = 0; i < k - 1; i++) d_off[i + 1] = beta_arr[i + 1];

  double *Z = (double *)safe_malloc((size_t)k * (size_t)k * sizeof(double));
  memset(Z, 0, (size_t)k * (size_t)k * sizeof(double));
  for (int i = 0; i < k; i++) Z[i + i * k] = 1.0;

  int info = tridiagonal_eigen_solver(k, d_diag, d_off, Z);
  (void)info;

  double min_eig = 1e30;
  int min_idx = -1;
  for (int i = 0; i < k; i++) {
    if (d_diag[i] < min_eig) {
      min_eig = d_diag[i];
      min_idx = i;
    }
  }
  *out_min_eigval = min_eig;

  int has_neg = 0;
  if (min_idx >= 0 && min_eig < 0.0) {
    double *d_V_neg = NULL;
    CUDA_CHECK(cudaMalloc(&d_V_neg, N * sizeof(double)));
    double *d_zmin = NULL;
    CUDA_CHECK(cudaMalloc(&d_zmin, k * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_zmin, Z + (size_t)min_idx * k,
                          k * sizeof(double), cudaMemcpyHostToDevice));
    double ga = 1.0, gb = 0.0;
    CUBLAS_CHECK(cublasDgemv(state->blas_handle, CUBLAS_OP_N, (int)N, k, &ga,
                             d_Q, (int)N, d_zmin, 1, &gb, d_V_neg, 1));
    double vn_local = 0.0;
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, (int)N, d_V_neg, 1, &vn_local));
    double vn = sqrt(rank_axis_sum_scalar(state, vn_local * vn_local));
    if (vn > 1e-30) {
      double inv = 1.0 / vn;
      CUBLAS_CHECK(cublasDscal(state->blas_handle, (int)N, &inv, d_V_neg, 1));
    }
    CUDA_CHECK(cudaFree(d_zmin));
    *out_V_neg = d_V_neg;
    has_neg = 1;
  }

  CUSPARSE_CHECK(cusparseDestroyDnMat(matV_descr));
  CUSPARSE_CHECK(cusparseDestroyDnMat(matHV_descr));
  free(alpha_arr); free(beta_arr);
  free(d_diag); free(d_off); free(Z);
  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_saved_S));
  return has_neg;
}

static inline int
COMPUTE_NEGATIVE_EIGEN_LANCZOS(cardal_sdp_solver_state_t *state, int blk_idx,
                               double *out_min_eigval,
                               double **out_neg_eigenvectors,
                               double **out_neg_eigenvalues) {
  block_low_rank_state_t *blk = state->block_low_rank_state[blk_idx];
  int dim = blk->dim;
  int k = (dim < LANCZOS_K) ? dim : LANCZOS_K;

  double *d_Q, *d_v, *d_qscratch = NULL;
  CUDA_CHECK(cudaMalloc(&d_Q, dim * k * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_v, dim * sizeof(double)));

  if (blk->psd_cone_rescaling != NULL)
    CUDA_CHECK(cudaMalloc(&d_qscratch, dim * sizeof(double)));
  double *alpha = (double *)safe_malloc(k * sizeof(double));
  double *beta = (double *)safe_malloc(k * sizeof(double));

  curandGenerator_t gen;
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  int rand_len = (dim % 2 == 0) ? dim : dim + 1;

  if (rand_len <= dim * k) {
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_Q, rand_len, 0.0, 1.0));
  } else {
    double *d_temp_rand;
    CUDA_CHECK(cudaMalloc(&d_temp_rand, rand_len * sizeof(double)));
    CURAND_CHECK(
        curandGenerateNormalDouble(gen, d_temp_rand, rand_len, 0.0, 1.0));
    CUDA_CHECK(cudaMemcpy(d_Q, d_temp_rand, dim * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(d_temp_rand));
  }
  curandDestroyGenerator(gen);

  double norm_q1;
  CUBLAS_CHECK(cublasDnrm2(state->blas_handle, dim, d_Q, 1, &norm_q1));
  double inv_norm = 1.0 / norm_q1;
  CUBLAS_CHECK(cublasDscal(state->blas_handle, dim, &inv_norm, d_Q, 1));

  beta[0] = 0.0;
  cusparseDnVecDescr_t vec_q, vec_v;
  CUSPARSE_CHECK(cusparseCreateDnVec(&vec_q, dim, d_Q, CUDA_R_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vec_v, dim, d_v, CUDA_R_64F));

  size_t bufferSize = 0;
  void *dBuffer = NULL;
  double spmv_alpha = 1.0, spmv_beta = 0.0;
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha,
      blk->matSpS, vec_q, &spmv_beta, vec_v, CUDA_R_64F,
      CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

  for (int j = 0; j < k; j++) {
    double *q_j = d_Q + j * dim;
    CUSPARSE_CHECK(cusparseDnVecSetValues(vec_q, q_j));

    unscaled_dual_spmv(state, blk, q_j, d_v, d_qscratch, vec_q, vec_v, dBuffer);

#ifdef IS_DISTRIBUTED
    NCCL_CHECK(ncclAllReduce(d_v, d_v, dim, ncclDouble, ncclSum,
                             state->grid_context->nccl_row, 0));
#endif

    if (j > 0) {
      double *q_prev = d_Q + (j - 1) * dim;
      double minus_beta = -beta[j];
      CUBLAS_CHECK(
          cublasDaxpy(state->blas_handle, dim, &minus_beta, q_prev, 1, d_v, 1));
    }
    CUBLAS_CHECK(
        cublasDdot(state->blas_handle, dim, q_j, 1, d_v, 1, &alpha[j]));

    double minus_alpha = -alpha[j];
    CUBLAS_CHECK(
        cublasDaxpy(state->blas_handle, dim, &minus_alpha, q_j, 1, d_v, 1));

    for (int i = 0; i <= j; i++) {
      double *q_i = d_Q + i * dim;
      double proj;
      CUBLAS_CHECK(cublasDdot(state->blas_handle, dim, q_i, 1, d_v, 1, &proj));
      double minus_proj = -proj;
      CUBLAS_CHECK(
          cublasDaxpy(state->blas_handle, dim, &minus_proj, q_i, 1, d_v, 1));
    }

    if (j < k - 1) {
      CUBLAS_CHECK(cublasDnrm2(state->blas_handle, dim, d_v, 1, &beta[j + 1]));
      if (beta[j + 1] < 1e-12) {
        k = j + 1;
        break;
      }
      double *q_next = d_Q + (j + 1) * dim;
      CUBLAS_CHECK(cublasDcopy(state->blas_handle, dim, d_v, 1, q_next, 1));
      double inv_beta = 1.0 / beta[j + 1];
      CUBLAS_CHECK(cublasDscal(state->blas_handle, dim, &inv_beta, q_next, 1));
    }
  }

  double *d = (double *)safe_malloc(k * sizeof(double));
  double *e = (double *)safe_malloc(k * sizeof(double));
  for (int i = 0; i < k; i++)
    d[i] = alpha[i];
  e[0] = 0.0;
  for (int i = 0; i < k - 1; i++)
    e[i + 1] = beta[i + 1];

  double *Z = (double *)safe_malloc(k * k * sizeof(double));
  memset(Z, 0, k * k * sizeof(double));
  for (int i = 0; i < k; i++)
    Z[i + i * k] = 1.0;

  int info = tridiagonal_eigen_solver(k, d, e, Z);
  if (info != 0)
    printf("Warning: Tridiagonal eigen solver failed to converge!\n");

  int neg_count = 0;
  double threshold = -1e-4;
  int *neg_indices = (int *)safe_malloc(k * sizeof(int));
  double min_eigval = 1e9;
  int min_idx = -1;

  for (int i = 0; i < k; i++) {
    if (d[i] < min_eigval) {
      min_eigval = d[i];
      min_idx = i;
    }
    if (d[i] < threshold) {
      neg_indices[neg_count] = i;
      neg_count++;
    }
  }

  if (state->objective_vector_rescaling > 0.0)
    min_eigval /= state->objective_vector_rescaling;

  if (out_min_eigval != NULL)
    *out_min_eigval = min_eigval;
  if (state->verbose >= 3)
    printf("  [Lanczos] Found %d negative eigenvalues (min lambda: %.4e, min "
           "idx: %d)\n",
           neg_count, min_eigval, min_idx);

  if (neg_count > 0) {
    double *h_Z_neg = (double *)safe_malloc(k * neg_count * sizeof(double));
    double *h_lambda_neg = (double *)safe_malloc(neg_count * sizeof(double));

    for (int i = 0; i < neg_count; i++) {
      int orig_idx = neg_indices[i];
      memcpy(h_Z_neg + i * k, Z + orig_idx * k, k * sizeof(double));
      h_lambda_neg[i] = d[orig_idx];
    }

    double *d_Z_neg, *d_U_neg;
    CUDA_CHECK(cudaMalloc(&d_Z_neg, k * neg_count * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_Z_neg, h_Z_neg, k * neg_count * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_U_neg, dim * neg_count * sizeof(double)));

    double alpha_gemm = 1.0, beta_gemm = 0.0;
    CUBLAS_CHECK(cublasDgemm(state->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, dim,
                             neg_count, k, &alpha_gemm, d_Q, dim, d_Z_neg, k,
                             &beta_gemm, d_U_neg, dim));

    *out_neg_eigenvectors = d_U_neg;
    if (out_neg_eigenvalues != NULL) {
      double *lam = (double *)safe_malloc(neg_count * sizeof(double));
      for (int i = 0; i < neg_count; i++)
        lam[i] = h_lambda_neg[i];
      *out_neg_eigenvalues = lam;
    }
    CUDA_CHECK(cudaFree(d_Z_neg));
    free(h_Z_neg);
    free(h_lambda_neg);
  } else {
    *out_neg_eigenvectors = NULL;
    if (out_neg_eigenvalues != NULL)
      *out_neg_eigenvalues = NULL;
  }

  free(neg_indices);
  free(d);
  free(e);
  free(Z);
  free(alpha);
  free(beta);
  CUDA_CHECK(cudaFree(dBuffer));
  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_v));
  if (d_qscratch != NULL)
    CUDA_CHECK(cudaFree(d_qscratch));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vec_q));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vec_v));
  return neg_count;
}

static inline void COMPUTE_GRADIENT(cardal_sdp_solver_state_t *state) {
  int blocks_m =
      (state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  compute_q1_kernel<<<blocks_m, THREADS_PER_BLOCK>>>(
      state->q1, state->dual_solution, state->q0, state->penalty_coef,
      state->num_constraints);

  double alpha_spmv = 1.0;
  double beta_spmv = 0.0;
  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_spmv,
      state->matAt, state->vec_q1, &beta_spmv, state->vec_dual_prod, CUDA_R_64F,
      CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

  double alpha_spmm = 2.0;
  double beta_spmm = 0.0;

  CUDA_CHECK(cudaStreamSynchronize(0));

  // Per-batch dispatch. CUSTOM batches do batched memcpy+add_Ay+SpMM on stream 0
  for (int bi = 0; bi < state->n_batches; bi++) {
    const block_low_rank_state_t *batch = state->block_low_rank_state[state->batch_leaders[bi]];
    if (batch->kind == CONE_BATCH_KIND_CUSTOM) {
      launch_batched_copy_objval_to_spS(batch, 0);
      launch_batched_add_Ay_to_spS(batch, state->dual_product, 0);
      launch_batched_spmm_grad(batch, state->low_rank_solution,
                               state->low_rank_gradient, alpha_spmm, 0);
      continue;
    }
    block_low_rank_state_t *blk =
        state->block_low_rank_state[state->batch_leaders[bi]];
    int nnz_Union =
        blk->objective_union_constraint_sparse_pattern->num_nonzeros;
    int nnz_A = blk->constraint_sparse_pattern->num_nonzeros;
    CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle, blk->stream));
    if (nnz_Union > 0) {
      CUDA_CHECK(
          cudaMemcpyAsync(blk->objective_union_constraint_sparse_pattern->val,
                          blk->objective_val, nnz_Union * sizeof(double),
                          cudaMemcpyDeviceToDevice, blk->stream));
      if (nnz_A > 0) {
        int blocks_A = (nnz_A + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        add_Ay_to_S_kernel<<<blocks_A, THREADS_PER_BLOCK, 0, blk->stream>>>(
            blk->objective_union_constraint_sparse_pattern->val,
            state->dual_product, blk->compat_mapping,
            blk->constraint_to_union_mapping, nnz_A);
      }
      CUSPARSE_CHECK(cusparseSpMM(state->sparse_handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_spmm,
                                  blk->matSpS, blk->matR, &beta_spmm,
                                  blk->matGrad, CUDA_R_64F,
                                  CUSPARSE_SPMM_ALG_DEFAULT,
                                  blk->spmm_buffer_S));
    } else {
      CUDA_CHECK(cudaMemsetAsync(blk->gradient, 0,
                                 blk->dim * blk->rank * sizeof(double),
                                 blk->stream));
    }
  }

  CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle, 0));
  for (int i = 0; i < state->cone_stream_pool_size; i++)
    CUDA_CHECK(cudaStreamSynchronize(state->cone_stream_pool[i]));

  if (state->lp_dim > 0) {
    int blocks = (state->lp_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_lp_gradient_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        state->low_rank_gradient + state->lp_solution_offset,
        state->lp_objective_vector,
        state->dual_product + state->lp_start_active_idx,
        state->low_rank_solution + state->lp_solution_offset, state->lp_dim);
  }

#ifdef IS_DISTRIBUTED
  if (state->grid_context != NULL && state->grid_context->dims[0] > 1) {
    NCCL_CHECK(ncclAllReduce(state->low_rank_gradient, state->low_rank_gradient,
                             state->length_low_rank_solution, ncclDouble,
                             ncclSum, state->grid_context->nccl_row, 0));
  }
#endif
}

static void COMPUTE_PRIMAL_RESIDUAL(cardal_sdp_solver_state_t *state) {
  CUDA_CHECK(cudaMemset(state->primal_solution, 0,
                        state->n_active_vars * sizeof(double)));

  for (int bi = 0; bi < state->n_batches; bi++) {
    const block_low_rank_state_t *batch = state->block_low_rank_state[state->batch_leaders[bi]];
    if (batch->kind == CONE_BATCH_KIND_CUSTOM) {
      launch_batched_sddmm_self_scatter(state->low_rank_solution, batch,
                                        state->primal_solution, 0);
    } else {
      compute_RR_block(state->batch_leaders[bi], state);
    }
  }

  CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle, 0));
  for (int i = 0; i < state->cone_stream_pool_size; i++)
    CUDA_CHECK(cudaStreamSynchronize(state->cone_stream_pool[i]));

  if (state->lp_dim > 0) {
    int blocks = (state->lp_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_lp_primal_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        state->primal_solution + state->lp_start_active_idx,
        state->low_rank_solution + state->lp_solution_offset, state->lp_dim);
  }

  // A * primal_solution
  double alpha_spmv = 1.0;
  double beta_spmv = 0.0;
  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_spmv,
      state->matA, state->vec_primal_sol, &beta_spmv, state->vec_primal_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));
  
#ifdef IS_DISTRIBUTED
  if (state->grid_context->dims[1] > 1) {
    NCCL_CHECK(ncclAllReduce(state->primal_product, state->primal_product,
                             state->num_constraints, ncclDouble, ncclSum,
                             state->grid_context->nccl_rank, 0));
  }
  if (state->grid_context->dims[2] > 1) {
    NCCL_CHECK(ncclAllReduce(state->primal_product, state->primal_product,
                             state->num_constraints, ncclDouble, ncclSum,
                             state->grid_context->nccl_cone, 0));
  }
#endif

  // A(RR^T) -> q0
  CUBLAS_CHECK(cublasDcopy(state->blas_handle, state->num_constraints,
                           state->primal_product, 1, state->q0, 1));

  // q0 = Ax - b
  double minus_one = -1.0;
  CUBLAS_CHECK(cublasDaxpy(state->blas_handle, state->num_constraints,
                           &minus_one, state->right_hand_side, 1, state->q0,
                           1));
}

static inline int CHECK_DUAL_INFEASIBILITY(cardal_sdp_solver_state_t *state) {
  double global_min_eigval = 1e9;
  int total_neg_count = 0;
  int *rank_incs = (int *)calloc(
      (size_t)(state->n_blks > 0 ? state->n_blks : 1), sizeof(int));

  for (int b = 0; b < state->n_blks; b++) {
    double blk_min_eigval = 0.0;
    double *blk_neg_eigenvectors = NULL;

    int neg_count = COMPUTE_NEGATIVE_EIGEN_POWER_ITER(state, b, &blk_min_eigval,
                                                      &blk_neg_eigenvectors);

    if (blk_min_eigval < global_min_eigval)
      global_min_eigval = blk_min_eigval;
    rank_incs[b] = neg_count;
    total_neg_count += neg_count;

    if (blk_neg_eigenvectors != NULL)
      CUDA_CHECK(cudaFree(blk_neg_eigenvectors));
  }

  if (state->lp_dim > 0) {
    double h_min_slack = COMPUTE_LP_MIN_SLACK(state);
    if (h_min_slack < global_min_eigval) {
      global_min_eigval = h_min_slack;
    }
  }

#ifdef IS_DISTRIBUTED
  if (state->grid_context != NULL && state->grid_context->dims[2] > 1) {
    double g_min = global_min_eigval;
    MPI_Allreduce(&global_min_eigval, &g_min, 1, MPI_DOUBLE, MPI_MIN,
                  state->grid_context->comm_cone);
    global_min_eigval = g_min;
    int g_total = total_neg_count;
    MPI_Allreduce(&total_neg_count, &g_total, 1, MPI_INT, MPI_SUM,
                  state->grid_context->comm_cone);
    total_neg_count = g_total;
  }
#endif

  if (global_min_eigval < 0) {
    state->absolute_dual_residual = fabs(global_min_eigval);
    state->relative_dual_residual =
        state->absolute_dual_residual / (1.0 + state->unscaled_objective_vector_norm);
  } else {
    state->absolute_dual_residual = 0.0;
    state->relative_dual_residual = 0.0;
  }

  free(rank_incs);
  return total_neg_count;
}

static inline void COMPUTE_OBJECTIVE_GAP(cardal_sdp_solver_state_t *state) {
  double p_obj = 0.0;
  double d_obj = 0.0;

  double alpha_sddmm = 1.0;
  double beta_sddmm = 0.0;

  int any_custom = 0;
  for (int bi = 0; bi < state->n_batches; bi++) {
    const block_low_rank_state_t *batch = state->block_low_rank_state[state->batch_leaders[bi]];
    if (batch->kind == CONE_BATCH_KIND_CUSTOM) {
      launch_batched_sddmm_self_to_spS(state->low_rank_solution, batch, 0);
      launch_batched_segmented_dot_p2(batch, batch->bdata->d_blk_idx,
                                      state->d_p2_per_cone, 0);
      any_custom = 1;
      continue;
    }
    block_low_rank_state_t *blk =
        state->block_low_rank_state[state->batch_leaders[bi]];
    int nnz_Union =
        blk->objective_union_constraint_sparse_pattern->num_nonzeros;
    if (nnz_Union > 0) {
      CUSPARSE_CHECK(cusparseSDDMM(state->sparse_handle,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   CUSPARSE_OPERATION_TRANSPOSE, &alpha_sddmm,
                                   blk->matR, blk->matR, &beta_sddmm,
                                   blk->matSpS, CUDA_R_64F,
                                   CUSPARSE_SDDMM_ALG_DEFAULT,
                                   blk->sddmm_buffer_C));
      double block_p_obj = 0.0;
      CUBLAS_CHECK(
          cublasDdot(state->blas_handle, nnz_Union,
                     blk->objective_union_constraint_sparse_pattern->val, 1,
                     blk->objective_val, 1, &block_p_obj));
      p_obj += block_p_obj;
    }
  }
  if (any_custom) {
    CUDA_CHECK(cudaMemcpy(state->h_p2_per_cone, state->d_p2_per_cone,
                          state->n_blks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    for (int bi = 0; bi < state->n_batches; bi++) {
      const block_low_rank_state_t *batch = state->block_low_rank_state[state->batch_leaders[bi]];
      if (batch->kind != CONE_BATCH_KIND_CUSTOM)
        continue;
      for (int c = 0; c < batch->n_cones; c++)
        p_obj += state->h_p2_per_cone[batch->bdata->blk_idx_h[c]];
    }
  }

  if (state->lp_dim > 0) {
    double lp_p_obj = 0.0;
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle, state->lp_dim, state->lp_objective_vector, 1,
        state->primal_solution + state->lp_start_active_idx, 1, &lp_p_obj));
    p_obj += lp_p_obj;
  }

  CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints,
                          state->right_hand_side, 1, state->dual_solution, 1,
                          &d_obj));

#ifdef IS_DISTRIBUTED
  if (state->grid_context->dims[1] > 1) {
    double global_p_obj = 0.0;
    MPI_Allreduce(&p_obj, &global_p_obj, 1, MPI_DOUBLE, MPI_SUM,
                  state->grid_context->comm_rank);
    p_obj = global_p_obj;
  }
  if (state->grid_context->dims[2] > 1) {
    double global_p_obj = 0.0;
    MPI_Allreduce(&p_obj, &global_p_obj, 1, MPI_DOUBLE, MPI_SUM,
                  state->grid_context->comm_cone);
    p_obj = global_p_obj;
  }

  double local_d_obj = -d_obj;
  double global_d_obj = local_d_obj;
  if (state->grid_context->dims[0] > 1) {
    MPI_Allreduce(&local_d_obj, &global_d_obj, 1, MPI_DOUBLE, MPI_SUM,
                  state->grid_context->comm_row);
  }

  if (state->grid_context->dims[0] > 1) {
    double local_p = p_obj;
    MPI_Allreduce(&local_p, &p_obj, 1, MPI_DOUBLE, MPI_SUM,
                  state->grid_context->comm_row);
  }

  {
    double denom = state->objective_vector_rescaling * state->right_hand_side_rescaling;
    if (denom > 0.0 && denom != 1.0) {
      p_obj /= denom;
      global_d_obj /= denom;
    }
  }

  state->primal_objective = p_obj;
  state->dual_objective = global_d_obj;
  state->objective_gap = fabs(p_obj - global_d_obj);
  state->relative_objective_gap =
      state->objective_gap / (1.0 + fabs(p_obj) + fabs(global_d_obj));
#else
  d_obj = -d_obj;
  {
    double denom = state->objective_vector_rescaling * state->right_hand_side_rescaling;
    if (denom > 0.0 && denom != 1.0) {
      p_obj /= denom;
      d_obj /= denom;
    }
  }
  state->primal_objective = p_obj;
  state->dual_objective = d_obj;
  state->objective_gap = fabs(p_obj - d_obj);
  state->relative_objective_gap =
      state->objective_gap / (1.0 + fabs(p_obj) + fabs(d_obj));
#endif
}

#ifndef ALORA_TRUST_C
#define ALORA_TRUST_C 2.0
#endif

static inline void
RUN_PER_CONE_LANCZOS(cardal_sdp_solver_state_t *state, int *rank_incs,
                     double **neg_eigvecs, double **neg_eigvals,
                     double *out_global_min, int *out_total_neg) {
  double global_min = 1e9;
  int total_neg = 0;
  for (int b = 0; b < state->n_blks; b++) {
    double blk_min = 0.0;
    double *blk_vecs = NULL;
    double *blk_vals = NULL;
    int neg_count = COMPUTE_NEGATIVE_EIGEN_LANCZOS(state, b, &blk_min,
                                                   &blk_vecs, &blk_vals);
    if (blk_min < global_min) global_min = blk_min;
    rank_incs[b] = neg_count;
    total_neg += neg_count;
    neg_eigvecs[b] = blk_vecs;
    neg_eigvals[b] = blk_vals;
  }
#ifdef IS_DISTRIBUTED
  if (state->grid_context != NULL && state->grid_context->dims[2] > 1) {
    double g_min = global_min;
    MPI_Allreduce(&global_min, &g_min, 1, MPI_DOUBLE, MPI_MIN,
                  state->grid_context->comm_cone);
    global_min = g_min;
  }
#endif
  *out_global_min = global_min;
  *out_total_neg = total_neg;
}

static inline void
FREE_LANCZOS_BUFFERS(cardal_sdp_solver_state_t *state,
                     double **neg_eigvecs, double **neg_eigvals) {
  (void)state;
  for (int b = 0; b < state->n_blks; b++) {
    if (neg_eigvecs[b] != NULL) {
      CUDA_CHECK(cudaFree(neg_eigvecs[b]));
      neg_eigvecs[b] = NULL;
    }
    if (neg_eigvals[b] != NULL) {
      free(neg_eigvals[b]);
      neg_eigvals[b] = NULL;
    }
  }
}

static inline int
CHECK_DUAL_INFEASIBILITY_AND_AUGMENT(cardal_sdp_solver_state_t *state) {
  int *rank_incs = (int *)calloc(state->n_blks, sizeof(int));
  double **neg_eigvecs = (double **)calloc(state->n_blks, sizeof(double *));
  double **neg_eigvals = (double **)calloc(state->n_blks, sizeof(double *));

  update_al_gradient_S(state);
  double al_min_eigval = 1e9;
  int total_neg_count = 0;
  RUN_PER_CONE_LANCZOS(state, rank_incs, neg_eigvecs, neg_eigvals,
                       &al_min_eigval, &total_neg_count);

  double perturb_fnorm = compute_penalty_perturbation_fnorm(state);
  int trust_al_sig = (fabs(al_min_eigval) > ALORA_TRUST_C * perturb_fnorm);
  int near_feasible = (state->relative_primal_residual < 1e-3);
  int trust_al = trust_al_sig || near_feasible;

  if (state->verbose >= 3)
    printf(">>> AL-gradient min eig: %.4e ; rho*||A*q0||_F: %.4e ; "
           "trust_AL=%d (sig=%d, near_feasible=%d, rel_pres=%.2e)\n",
           al_min_eigval, perturb_fnorm, trust_al, trust_al_sig,
           near_feasible, state->relative_primal_residual);

  if (!trust_al && total_neg_count > 0) {
    if (state->verbose >= 3)
      printf(">>> AL signal dominated by perturbation; switching to "
             "feasibility (rho*A*q0) eigenvectors.\n");
    FREE_LANCZOS_BUFFERS(state, neg_eigvecs, neg_eigvals);
    memset(rank_incs, 0, state->n_blks * sizeof(int));
    update_penalty_only_S(state);
    double feas_min_eigval = 1e9;
    int feas_total_neg = 0;
    RUN_PER_CONE_LANCZOS(state, rank_incs, neg_eigvecs, neg_eigvals,
                         &feas_min_eigval, &feas_total_neg);
    total_neg_count = feas_total_neg;
    if (state->verbose >= 3)
      printf(">>> Feasibility-operator min eig: %.4e ; total neg: %d\n",
             feas_min_eigval, feas_total_neg);
  }

  double global_min_eigval = al_min_eigval;
  if (state->lp_dim > 0) {
    double h_min_slack = COMPUTE_LP_MIN_SLACK(state);
    if (h_min_slack < global_min_eigval) {
      global_min_eigval = h_min_slack;
    }
  }
  if (global_min_eigval < 0) {
    state->absolute_dual_residual = fabs(global_min_eigval);
    state->relative_dual_residual =
        state->absolute_dual_residual /
        (1.0 + state->unscaled_objective_vector_norm);
  } else {
    state->absolute_dual_residual = 0.0;
    state->relative_dual_residual = 0.0;
  }

  if (total_neg_count > 0) {
    if (state->verbose >= 3)
      printf(">>> Augment: %d negative eigenvalues across blocks (signal: %s)\n",
             total_neg_count, trust_al ? "AL gradient" : "rho*A*q0");
    augment_system_rank(state, rank_incs, neg_eigvecs, neg_eigvals);
  }

  FREE_LANCZOS_BUFFERS(state, neg_eigvecs, neg_eigvals);
  free(neg_eigvecs);
  free(neg_eigvals);
  free(rank_incs);
  return total_neg_count;
}

static inline double COMPUTE_EXACT_STEP_SIZE_TAUMAX(
    cardal_sdp_solver_state_t *state, double tau_max);
static inline double COMPUTE_EXACT_STEP_SIZE(cardal_sdp_solver_state_t *state) {
  return COMPUTE_EXACT_STEP_SIZE_TAUMAX(state, 1.0);
}
static inline double COMPUTE_EXACT_STEP_SIZE_TAUMAX(
    cardal_sdp_solver_state_t *state, double tau_max) {
  CUBLAS_CHECK(
      cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_DEVICE));
  CUDA_CHECK(cudaMemset(state->primal_direct_solution_cross, 0,
                        state->n_active_vars * sizeof(double)));
  CUDA_CHECK(cudaMemset(state->primal_direct_double, 0,
                        state->n_active_vars * sizeof(double)));
  CUDA_CHECK(cudaMemset(state->q1, 0, state->num_constraints * sizeof(double)));
  CUDA_CHECK(cudaMemset(state->q2, 0, state->num_constraints * sizeof(double)));

  for (int bi = 0; bi < state->n_batches; bi++) {
    const block_low_rank_state_t *batch = state->block_low_rank_state[state->batch_leaders[bi]];
    if (batch->kind == CONE_BATCH_KIND_CUSTOM) {
      launch_batched_sddmm_cross_scatter(state->low_rank_direction,
                                         state->low_rank_direction, batch,
                                         state->primal_direct_double, 0);
      launch_batched_sddmm_cross_scatter(state->low_rank_solution,
                                         state->low_rank_direction, batch,
                                         state->primal_direct_solution_cross,
                                         0);
      launch_batched_sddmm_self_to_spS(state->low_rank_direction, batch, 0);
      launch_batched_segmented_dot_p2(batch, batch->bdata->d_blk_idx,
                                      state->d_p2_per_cone, 0);
    } else {
      compute_RD_DD_block(state->batch_leaders[bi], state);
    }
  }
  for (int i = 0; i < state->cone_stream_pool_size; i++)
    CUDA_CHECK(cudaStreamSynchronize(state->cone_stream_pool[i]));
  CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle, 0));
  CUBLAS_CHECK(cublasSetStream(state->blas_handle, 0));

  if (state->n_blks > 0) {
    sum_to_scalar_kernel<<<1, 256>>>(state->d_step_scalars + 6,
                                     state->d_p2_per_cone, state->n_blks);
  } else {
    CUDA_CHECK(cudaMemsetAsync(state->d_step_scalars + 6, 0, sizeof(double), 0));
  }

  if (state->lp_dim > 0) {
    int blocks = (state->lp_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_lp_line_search_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        state->low_rank_solution + state->lp_solution_offset,
        state->low_rank_direction + state->lp_solution_offset,
        state->primal_direct_solution_cross + state->lp_start_active_idx,
        state->primal_direct_double + state->lp_start_active_idx,
        state->lp_dim);
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle, state->lp_dim, state->lp_objective_vector, 1,
        state->primal_direct_double + state->lp_start_active_idx, 1,
        state->d_step_scalars + 7));
  } else {
    CUDA_CHECK(cudaMemsetAsync(state->d_step_scalars + 7, 0, sizeof(double), 0));
  }

  double alpha_spmv = 1.0;
  double beta_spmv = 0.0;
  CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol,
                                        state->primal_direct_double));
  CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->q2));
  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_spmv,
      state->matA, state->vec_primal_sol, &beta_spmv, state->vec_primal_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));
  CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol,
                                        state->primal_direct_solution_cross));
  CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->q1));
  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_spmv,
      state->matA, state->vec_primal_sol, &beta_spmv, state->vec_primal_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

#ifdef IS_DISTRIBUTED
  if (state->grid_context->dims[1] > 1) {
      NCCL_CHECK(ncclGroupStart());
      NCCL_CHECK(ncclAllReduce(state->q2, state->q2, state->num_constraints,
                               ncclDouble, ncclSum, state->grid_context->nccl_rank, 0));
      NCCL_CHECK(ncclAllReduce(state->q1, state->q1, state->num_constraints,
                               ncclDouble, ncclSum, state->grid_context->nccl_rank, 0));
      NCCL_CHECK(ncclGroupEnd());
  }
  if (state->grid_context->dims[2] > 1) {
      NCCL_CHECK(ncclGroupStart());
      NCCL_CHECK(ncclAllReduce(state->q2, state->q2, state->num_constraints,
                               ncclDouble, ncclSum, state->grid_context->nccl_cone, 0));
      NCCL_CHECK(ncclAllReduce(state->q1, state->q1, state->num_constraints,
                               ncclDouble, ncclSum, state->grid_context->nccl_cone, 0));
      NCCL_CHECK(ncclGroupEnd());
  }
#endif

  CUBLAS_CHECK(cublasDscal(state->blas_handle, state->num_constraints,
                           state->d_step_scalars + 10, state->q1, 1));

  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_primal_sol, state->primal_solution));
  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

  CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, state->q2,
                          1, state->q2, 1, &state->d_step_scalars[0]));
  CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, state->q1,
                          1, state->q2, 1, &state->d_step_scalars[1]));
  CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, state->q1,
                          1, state->q1, 1, &state->d_step_scalars[2]));
  CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints,
                          state->dual_solution, 1, state->q2, 1,
                          &state->d_step_scalars[3]));
  CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, state->q0,
                          1, state->q2, 1, &state->d_step_scalars[4]));
  CUBLAS_CHECK(cublasDdot(state->blas_handle, state->length_low_rank_solution,
                          state->low_rank_gradient, 1,
                          state->low_rank_direction, 1,
                          &state->d_step_scalars[5]));

#ifdef IS_DISTRIBUTED
  if (state->grid_context->dims[0] > 1) {
    NCCL_CHECK(ncclGroupStart());
    for (int s = 0; s < 5; s++)
      NCCL_CHECK(ncclAllReduce(state->d_step_scalars + s,
                               state->d_step_scalars + s, 1, ncclDouble,
                               ncclSum, state->grid_context->nccl_row, 0));
    NCCL_CHECK(ncclAllReduce(state->d_step_scalars + 6,
                             state->d_step_scalars + 6, 1, ncclDouble, ncclSum,
                             state->grid_context->nccl_row, 0));
    NCCL_CHECK(ncclAllReduce(state->d_step_scalars + 7,
                             state->d_step_scalars + 7, 1, ncclDouble, ncclSum,
                             state->grid_context->nccl_row, 0));
    NCCL_CHECK(ncclGroupEnd());
  }
  if (state->grid_context->dims[1] > 1) {
    NCCL_CHECK(ncclAllReduce(state->d_step_scalars + 5,
                             state->d_step_scalars + 5, 3, ncclDouble, ncclSum,
                             state->grid_context->nccl_rank, 0));
  }
  if (state->grid_context->dims[2] > 1) {
    NCCL_CHECK(ncclAllReduce(state->d_step_scalars + 5,
                             state->d_step_scalars + 5, 3, ncclDouble, ncclSum,
                             state->grid_context->nccl_cone, 0));
  }
#endif

  solve_line_search_tau_kernel<<<1, 1>>>(state->d_step_scalars,
                                         state->penalty_coef,
                                         /*tau_slot=*/8, /*tau_sq_slot=*/9,
                                         tau_max);

  CUBLAS_CHECK(
      cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));
  double tau_h = 0.0, tau_sq_h = 0.0;
  CUDA_CHECK(cudaMemcpy(&tau_h, state->d_step_scalars + 8, sizeof(double),
                        cudaMemcpyDeviceToHost));
  tau_sq_h = tau_h * tau_h;
  CUBLAS_CHECK(cublasDaxpy(state->blas_handle, state->num_constraints, &tau_h,
                           state->q1, 1, state->q0, 1));
  CUBLAS_CHECK(cublasDaxpy(state->blas_handle, state->num_constraints,
                           &tau_sq_h, state->q2, 1, state->q0, 1));
  return tau_h;
}

// Post-LBFGS gate; for each PERCONE block runs FIND_AL_NEGATIVE_CURVATURE,
// sign-aligns V_neg against gradient, applies R += tau*V_neg. Returns
// number of blocks that escaped. Caller must re-run LBFGS after.
static inline int
DETECT_NEGATIVE_CURVATURE_AND_ESCAPE(cardal_sdp_solver_state_t *state,
                                     double threshold_factor) {
  if (state->n_blks <= 0) return 0;
  double scale_factor = 1.0 + state->objective_vector_linf_norm;
  double curvature_threshold = threshold_factor * scale_factor;

  int escaped = 0;

  CUDA_CHECK(cudaMemset(state->low_rank_direction, 0,
                        state->length_low_rank_solution * sizeof(double)));

  int blk_offset = 0;
  for (int b = 0; b < state->n_blks; b++) {
    block_low_rank_state_t *blk = state->block_low_rank_state[b];
    int dim = blk->dim;
    int r = blk->rank;
    int blk_len = dim * r;

    if (blk->kind == CONE_BATCH_KIND_CUSTOM || r <= 0) {
      blk_offset += blk_len;
      continue;
    }

    double lambda_min;
    double *d_V_neg = NULL;
    FIND_AL_NEGATIVE_CURVATURE(state, b, &lambda_min, &d_V_neg);

    if (d_V_neg == NULL || lambda_min >= -curvature_threshold) {
      if (d_V_neg) CUDA_CHECK(cudaFree(d_V_neg));
      blk_offset += blk_len;
      continue;
    }

    double dot_local = 0.0;
    CUBLAS_CHECK(cublasDdot(state->blas_handle, blk_len, blk->gradient, 1,
                            d_V_neg, 1, &dot_local));
    // Sign-flip decision must be globally consistent across rank-axis peers;
    // otherwise some ranks flip V_neg and others don't, the per-rank slabs
    // become incoherent, and the step taken at line ~1591 disagrees.
    double dot = rank_axis_sum_scalar(state, dot_local);
    if (dot > 0.0) {
      double m1 = -1.0;
      CUBLAS_CHECK(
          cublasDscal(state->blas_handle, blk_len, &m1, d_V_neg, 1));
    }

    // Place V_neg into block b's slot of low_rank_direction.
    CUDA_CHECK(cudaMemcpy(state->low_rank_direction + blk_offset, d_V_neg,
                          blk_len * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    double tau = COMPUTE_EXACT_STEP_SIZE_TAUMAX(state, 1e6);

    if (fabs(tau) > 1e-30) {
      // R_b += tau * V_neg
      CUBLAS_CHECK(cublasDaxpy(state->blas_handle, blk_len, &tau, d_V_neg, 1,
                               state->low_rank_solution + blk_offset, 1));
      escaped++;
    }

    CUDA_CHECK(cudaMemset(state->low_rank_direction + blk_offset, 0,
                          blk_len * sizeof(double)));

    CUDA_CHECK(cudaFree(d_V_neg));
    blk_offset += blk_len;
  }
  return escaped;
}

#endif // SOLVER_CORE_OP_CUH
