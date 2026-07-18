/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "sdp_op.h"
#include "sdp_types.h"
#include "solver.h"
#include "solver_state.h"
#include "utils.h"
#include "batched_cone_ops.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

__device__ double atomicMinDouble(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed,
        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__global__ void compute_q1_kernel(double *__restrict__ q1,
                                  const double *__restrict__ lambda,
                                  const double *__restrict__ q0, double rho,
                                  int m) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    q1[idx] = lambda[idx] + rho * q0[idx];
  }
}
__global__ void compute_lp_primal_kernel(double *__restrict__ primal_sol,
                                         const double *__restrict__ lp_v,
                                         int lp_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < lp_dim) {
    primal_sol[idx] = lp_v[idx] * lp_v[idx];
  }
}

__global__ void compute_lp_gradient_kernel(double *__restrict__ lp_grad,
                                           const double *__restrict__ lp_C,
                                           const double *__restrict__ dual_prod,
                                           const double *__restrict__ lp_v,
                                           int lp_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < lp_dim) {
    lp_grad[idx] = 2.0 * (lp_C[idx] + dual_prod[idx]) * lp_v[idx];
  }
}
__global__ void add_Ay_to_S_kernel(double *__restrict__ S_val,
                                   const double *__restrict__ Ay_val_global,
                                   const int *__restrict__ compat_mapping,
                                   const int *__restrict__ A_to_Union_mapping,
                                   int nnz_A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nnz_A) {
    int global_var_idx = compat_mapping[idx];
    int union_idx = A_to_Union_mapping[idx];
    S_val[union_idx] += Ay_val_global[global_var_idx];
  }
}

__global__ void
scatter_sddmm_to_global_kernel(const double *__restrict__ local_vals,
                               const int *__restrict__ compat_mapping,
                               double *__restrict__ global_primal_solution,
                               int nnz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nnz) {
    int global_idx = compat_mapping[idx];
    global_primal_solution[global_idx] = local_vals[idx];
  }
}

// w (dim x 1 device, cone scaled space) -> out_m (num_constraints device).
// Clobbers blk->matSpA values and state->primal_solution.
static void as_ww_to_vec(cardal_sdp_solver_state_t *state,
                         block_low_rank_state_t *blk, double *d_w,
                         double *d_out_m) {
  int dim = blk->dim;
  CUDA_CHECK(cudaMemsetAsync(d_out_m, 0,
                             state->num_constraints * sizeof(double), 0));
  if (blk->constraint_sparse_pattern == NULL)
    return;
  int nnz_A = blk->constraint_sparse_pattern->num_nonzeros;
  if (nnz_A == 0)
    return;

  cusparseDnMatDescr_t matW;
  CUSPARSE_CHECK(cusparseCreateDnMat(&matW, dim, 1, dim, d_w, CUDA_R_64F,
                                     CUSPARSE_ORDER_COL));
  double alpha = 1.0, beta = 0.0;
  size_t bufSize = 0;
  CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &alpha, matW, matW, &beta, blk->matSpA,
      CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, &bufSize));
  void *buf = NULL;
  if (bufSize > 0)
    CUDA_CHECK(cudaMalloc(&buf, bufSize));
  CUSPARSE_CHECK(cusparseSDDMM(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &alpha, matW, matW, &beta, blk->matSpA,
      CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, buf));

  CUDA_CHECK(cudaMemset(state->primal_solution, 0,
                        state->n_active_vars * sizeof(double)));
  int blocks = (nnz_A + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  scatter_sddmm_to_global_kernel<<<blocks, THREADS_PER_BLOCK>>>(
      blk->constraint_sparse_pattern->val, blk->compat_mapping,
      state->primal_solution, nnz_A);

  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_primal_sol, state->primal_solution));
  cusparseDnVecDescr_t vec_out;
  CUSPARSE_CHECK(
      cusparseCreateDnVec(&vec_out, state->num_constraints, d_out_m, CUDA_R_64F));
  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      state->matA, state->vec_primal_sol, &beta, vec_out, CUDA_R_64F,
      CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vec_out));

  if (buf != NULL)
    CUDA_CHECK(cudaFree(buf));
  CUSPARSE_CHECK(cusparseDestroyDnMat(matW));
}

void compute_rank_lift_A_ww(cardal_sdp_solver_state_t *state,
                            block_low_rank_state_t *blk, double *d_w,
                            double *d_out_m) {
  as_ww_to_vec(state, blk, d_w, d_out_m);
}

void compute_rank_lift_A_uv(cardal_sdp_solver_state_t *state,
                            block_low_rank_state_t *blk, double *d_u,
                            double *d_v, double *d_out_m) {
  CUDA_CHECK(cudaMemsetAsync(d_out_m, 0,
                             state->num_constraints * sizeof(double), 0));
  if (blk->constraint_sparse_pattern == NULL)
    return;
  int nnz_A = blk->constraint_sparse_pattern->num_nonzeros;
  if (nnz_A == 0)
    return;

  int dim = blk->dim;
  cusparseDnMatDescr_t matU, matV;
  CUSPARSE_CHECK(cusparseCreateDnMat(&matU, dim, 1, dim, d_u, CUDA_R_64F,
                                     CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&matV, dim, 1, dim, d_v, CUDA_R_64F,
                                     CUSPARSE_ORDER_COL));

  double alpha = 1.0, beta = 0.0, accumulate = 1.0;
  size_t buffer_size_u = 0, buffer_size_v = 0;
  CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &alpha, matU, matV, &beta, blk->matSpA,
      CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, &buffer_size_u));
  CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &alpha, matV, matU, &accumulate,
      blk->matSpA, CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, &buffer_size_v));
  size_t buffer_size =
      buffer_size_u > buffer_size_v ? buffer_size_u : buffer_size_v;
  void *buffer = NULL;
  if (buffer_size > 0)
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

  CUSPARSE_CHECK(cusparseSDDMM(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &alpha, matU, matV, &beta, blk->matSpA,
      CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, buffer));
  CUSPARSE_CHECK(cusparseSDDMM(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &alpha, matV, matU, &accumulate,
      blk->matSpA, CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, buffer));

  CUDA_CHECK(cudaMemset(state->primal_solution, 0,
                        state->n_active_vars * sizeof(double)));
  int blocks = (nnz_A + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  scatter_sddmm_to_global_kernel<<<blocks, THREADS_PER_BLOCK>>>(
      blk->constraint_sparse_pattern->val, blk->compat_mapping,
      state->primal_solution, nnz_A);

  CUSPARSE_CHECK(
      cusparseDnVecSetValues(state->vec_primal_sol, state->primal_solution));
  cusparseDnVecDescr_t vec_out;
  CUSPARSE_CHECK(cusparseCreateDnVec(&vec_out, state->num_constraints, d_out_m,
                                     CUDA_R_64F));
  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      state->matA, state->vec_primal_sol, &beta, vec_out, CUDA_R_64F,
      CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

  CUSPARSE_CHECK(cusparseDestroyDnVec(vec_out));
  CUSPARSE_CHECK(cusparseDestroyDnMat(matU));
  CUSPARSE_CHECK(cusparseDestroyDnMat(matV));
  if (buffer != NULL)
    CUDA_CHECK(cudaFree(buffer));
}

double compute_Asuu_norm_sq_percone(cardal_sdp_solver_state_t *state,
                                    block_low_rank_state_t *blk, double *d_u) {
  int nnz_A = blk->constraint_sparse_pattern->num_nonzeros;
  if (nnz_A == 0)
    return 0.0;
  cublasPointerMode_t saved_mode;
  CUBLAS_CHECK(cublasGetPointerMode(state->blas_handle, &saved_mode));
  CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));
  as_ww_to_vec(state, blk, d_u, state->q1);
  double q = 0.0;
  CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, state->q1,
                          1, state->q1, 1, &q));
  CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, saved_mode));
  return q;
}
// Cyclic Jacobi on n x n symmetric A (row-major, destroyed). Outputs eigvals
// and eigvecs (col-major).
static void jacobi_eig_sym(double *A, int n, double *eigvals, double *eigvecs) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      eigvecs[i + j * n] = (i == j) ? 1.0 : 0.0;
  if (n == 1) {
    eigvals[0] = A[0];
    return;
  }
  for (int sweep = 0; sweep < 100; sweep++) {
    double off = 0.0;
    for (int p = 0; p < n; p++)
      for (int q = p + 1; q < n; q++)
        off += A[p * n + q] * A[p * n + q];
    if (off < 1e-28)
      break;
    for (int p = 0; p < n; p++) {
      for (int q = p + 1; q < n; q++) {
        double apq = A[p * n + q];
        if (fabs(apq) < 1e-300)
          continue;
        double app = A[p * n + p], aqq = A[q * n + q];
        double phi = 0.5 * atan2(2.0 * apq, aqq - app);
        double c = cos(phi), s = sin(phi);
        for (int k = 0; k < n; k++) {
          double akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; k++) {
          double apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; k++) {
          double vkp = eigvecs[k + p * n], vkq = eigvecs[k + q * n];
          eigvecs[k + p * n] = c * vkp - s * vkq;
          eigvecs[k + q * n] = s * vkp + c * vkq;
        }
      }
    }
  }
  for (int i = 0; i < n; i++)
    eigvals[i] = A[i * n + i];
}

// Pack/unpack symmetric r x r matrix S to/from svec s of length r(r+1)/2.
// Ordering: r diagonal entries first (i,i), then off-diagonals (i<j).
static inline int svec_len(int r) { return r * (r + 1) / 2; }

static void svec_unpack(const double *s, int r, double *S /*r x r row-major*/) {
  for (int i = 0; i < r; i++)
    S[i * r + i] = s[i];
  int idx = r;
  for (int i = 0; i < r; i++)
    for (int j = i + 1; j < r; j++) {
      S[i * r + j] = s[idx];
      S[j * r + i] = s[idx];
      idx++;
    }
}

static void svec_pack(const double *S, int r, double *s) {
  for (int i = 0; i < r; i++)
    s[i] = S[i * r + i];
  int idx = r;
  for (int i = 0; i < r; i++)
    for (int j = i + 1; j < r; j++)
      s[idx++] = S[i * r + j];
}

// Project symmetric S (r x r row-major) onto the PSD cone in place.
static void project_psd(double *S, int r, double *scratch_evec,
                        double *scratch_eval, double *scratch_A) {
  for (int i = 0; i < r * r; i++)
    scratch_A[i] = S[i];
  jacobi_eig_sym(scratch_A, r, scratch_eval, scratch_evec);
  for (int i = 0; i < r * r; i++)
    S[i] = 0.0;
  for (int e = 0; e < r; e++) {
    double lam = scratch_eval[e];
    if (lam <= 0.0)
      continue;
    for (int i = 0; i < r; i++)
      for (int j = 0; j < r; j++)
        S[i * r + j] += lam * scratch_evec[i + e * r] * scratch_evec[j + e * r];
  }
}

// ALORA augmentation SDP (per PERCONE cone): min_{S>=0} <L,S> + (rho/2)||A_s(U S U^T)||^2,
// U = D V, L = U^T G U. Writes W = U S^{1/2} into d_new_cols.
double solve_alora_sdp_percone(cardal_sdp_solver_state_t *state,
                               block_low_rank_state_t *blk, const double *d_V,
                               const double *h_lambda, int r, double rho,
                               double q0_norm_budget, double *d_new_cols) {
  int dim = blk->dim;
  int m = state->num_constraints;
  int P = svec_len(r);

  cublasPointerMode_t saved_mode;
  CUBLAS_CHECK(cublasGetPointerMode(state->blas_handle, &saved_mode));
  CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

  // U = D V
  double *d_U = NULL;
  CUDA_CHECK(cudaMalloc(&d_U, (size_t)dim * r * sizeof(double)));
  if (blk->psd_cone_rescaling != NULL) {
    CUBLAS_CHECK(cublasDdgmm(state->blas_handle, CUBLAS_SIDE_LEFT, dim, r, d_V,
                             dim, blk->psd_cone_rescaling, 1, d_U, dim));
  } else {
    CUDA_CHECK(cudaMemcpy(d_U, d_V, (size_t)dim * r * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  }

  // L = U^T (G U), G = blk->matSpS.
  double *d_GU = NULL, *d_L = NULL;
  CUDA_CHECK(cudaMalloc(&d_GU, (size_t)dim * r * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_L, (size_t)r * r * sizeof(double)));
  {
    cusparseDnMatDescr_t matU, matGU;
    CUSPARSE_CHECK(cusparseCreateDnMat(&matU, dim, r, dim, d_U, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matGU, dim, r, dim, d_GU, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    double a = 1.0, b = 0.0;
    size_t bsz = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &a, blk->matSpS, matU, &b, matGU,
        CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bsz));
    void *buf = NULL;
    if (bsz > 0)
      CUDA_CHECK(cudaMalloc(&buf, bsz));
    CUSPARSE_CHECK(cusparseSpMM(state->sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &a,
                                blk->matSpS, matU, &b, matGU, CUDA_R_64F,
                                CUSPARSE_SPMM_ALG_DEFAULT, buf));
    if (buf)
      CUDA_CHECK(cudaFree(buf));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matU));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matGU));
    // d_L = U^T GU (r x r), contraction over dim.
    CUBLAS_CHECK(cublasDgemm(state->blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, r, r,
                             dim, &a, d_U, dim, d_GU, dim, &b, d_L, r));
  }
  double *h_L = (double *)safe_malloc((size_t)r * r * sizeof(double));
  CUDA_CHECK(cudaMemcpy(h_L, d_L, (size_t)r * r * sizeof(double),
                        cudaMemcpyDeviceToHost));

  // Build C (m x P) and Q = C^T C (P x P).
  double *d_C = NULL;
  CUDA_CHECK(cudaMalloc(&d_C, (size_t)m * P * sizeof(double)));
  double *d_w = NULL;
  CUDA_CHECK(cudaMalloc(&d_w, (size_t)dim * sizeof(double)));
  // Diagonal columns c_ii (cols 0..r-1).
  for (int i = 0; i < r; i++) {
    as_ww_to_vec(state, blk, d_U + (size_t)i * dim, d_C + (size_t)i * m);
  }
  // Off-diagonal columns (cols r..P-1): 2*c_ij^sym = A((u_i+u_j)^2)-c_ii-c_jj.
  {
    int col = r;
    double one = 1.0, mone = -1.0;
    for (int i = 0; i < r; i++) {
      for (int j = i + 1; j < r; j++) {
        CUDA_CHECK(cudaMemcpy(d_w, d_U + (size_t)i * dim,
                              dim * sizeof(double), cudaMemcpyDeviceToDevice));
        CUBLAS_CHECK(cublasDaxpy(state->blas_handle, dim, &one,
                                 d_U + (size_t)j * dim, 1, d_w, 1));
        as_ww_to_vec(state, blk, d_w, d_C + (size_t)col * m);
        CUBLAS_CHECK(cublasDaxpy(state->blas_handle, m, &mone,
                                 d_C + (size_t)i * m, 1, d_C + (size_t)col * m, 1));
        CUBLAS_CHECK(cublasDaxpy(state->blas_handle, m, &mone,
                                 d_C + (size_t)j * m, 1, d_C + (size_t)col * m, 1));
        col++;
      }
    }
  }
  double *d_Q = NULL;
  CUDA_CHECK(cudaMalloc(&d_Q, (size_t)P * P * sizeof(double)));
  {
    double a = 1.0, b = 0.0;
    CUBLAS_CHECK(cublasDgemm(state->blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, P, P,
                             m, &a, d_C, m, d_C, m, &b, d_Q, P));
  }
  double *h_Q = (double *)safe_malloc((size_t)P * P * sizeof(double));
  CUDA_CHECK(cudaMemcpy(h_Q, d_Q, (size_t)P * P * sizeof(double),
                        cudaMemcpyDeviceToHost));

  // ---- Host SDP solve: min_{S>=0} l^T s + (rho/2) s^T Q s ----
  double *l = (double *)safe_calloc(P, sizeof(double));
  for (int i = 0; i < r; i++)
    l[i] = h_L[i * r + i];
  {
    int idx = r;
    for (int i = 0; i < r; i++)
      for (int j = i + 1; j < r; j++)
        l[idx++] = 2.0 * h_L[i * r + j];
  }
  // Estimate spectral norm of Q via power iteration (P small).
  double Lq = 0.0;
  {
    double *v = (double *)safe_malloc(P * sizeof(double));
    double *Av = (double *)safe_malloc(P * sizeof(double));
    for (int i = 0; i < P; i++)
      v[i] = 1.0;
    for (int it = 0; it < 50; it++) {
      for (int i = 0; i < P; i++) {
        double acc = 0.0;
        for (int j = 0; j < P; j++)
          acc += h_Q[i * P + j] * v[j];
        Av[i] = acc;
      }
      double nrm = 0.0;
      for (int i = 0; i < P; i++)
        nrm += Av[i] * Av[i];
      nrm = sqrt(nrm);
      if (nrm < 1e-30)
        break;
      Lq = nrm;
      for (int i = 0; i < P; i++)
        v[i] = Av[i] / nrm;
    }
    free(v);
    free(Av);
  }
  double step = 1.0 / (rho * Lq + 1e-30);
  double *s = (double *)safe_calloc(P, sizeof(double));
  double *grad = (double *)safe_malloc(P * sizeof(double));
  double *S = (double *)safe_malloc((size_t)r * r * sizeof(double));
  double *ev = (double *)safe_malloc((size_t)r * r * sizeof(double));
  double *ew = (double *)safe_malloc((size_t)r * sizeof(double));
  double *sa = (double *)safe_malloc((size_t)r * r * sizeof(double));
  for (int it = 0; it < 400; it++) {
    for (int i = 0; i < P; i++) {
      double acc = 0.0;
      for (int j = 0; j < P; j++)
        acc += h_Q[i * P + j] * s[j];
      grad[i] = l[i] + rho * acc;
    }
    for (int i = 0; i < P; i++)
      s[i] -= step * grad[i];
    svec_unpack(s, r, S);
    project_psd(S, r, ev, ew, sa);
    svec_pack(S, r, s);
  }

  double Cs_norm = 0.0;
  for (int i = 0; i < P; i++) {
    double acc = 0.0;
    for (int j = 0; j < P; j++)
      acc += h_Q[i * P + j] * s[j];
    Cs_norm += s[i] * acc;
  }
  Cs_norm = (Cs_norm > 0.0) ? sqrt(Cs_norm) : 0.0;
  double Cs_pre = Cs_norm;
  int capped = 0;
  if (q0_norm_budget > 0.0 && Cs_norm > q0_norm_budget) {
    double xi = q0_norm_budget / Cs_norm;
    for (int i = 0; i < P; i++)
      s[i] *= xi;
    Cs_norm = q0_norm_budget;
    capped = 1;
  }
  if (state->verbose >= 2)
    fprintf(stderr, "[augment-sdp] r=%d rho=%.2e q0=%.2e Cs_pre=%.2e Cs_post=%.2e %s\n",
            r, rho, q0_norm_budget, Cs_pre, Cs_norm, capped ? "(CAPPED)" : "");

  // S^{1/2} = E diag(sqrt sigma) E^T from final eigendecomp.
  svec_unpack(s, r, S);
  for (int i = 0; i < r * r; i++)
    sa[i] = S[i];
  jacobi_eig_sym(sa, r, ew, ev);
  double *Shalf = (double *)safe_calloc((size_t)r * r, sizeof(double));
  for (int e = 0; e < r; e++) {
    double lam = ew[e];
    if (lam <= 0.0)
      continue;
    double sq = sqrt(lam);
    for (int i = 0; i < r; i++)
      for (int j = 0; j < r; j++)
        Shalf[i * r + j] += sq * ev[i + e * r] * ev[j + e * r];
  }
  // W = U * Shalf  (dim x r). Shalf symmetric so col/row-major coincide.
  double *d_Shalf = NULL;
  CUDA_CHECK(cudaMalloc(&d_Shalf, (size_t)r * r * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_Shalf, Shalf, (size_t)r * r * sizeof(double),
                        cudaMemcpyHostToDevice));
  {
    double a = 1.0, b = 0.0;
    CUBLAS_CHECK(cublasDgemm(state->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, r,
                             r, &a, d_U, dim, d_Shalf, r, &b, d_new_cols, dim));
  }

  free(h_L);
  free(h_Q);
  free(l);
  free(s);
  free(grad);
  free(S);
  free(ev);
  free(ew);
  free(sa);
  free(Shalf);
  CUDA_CHECK(cudaFree(d_U));
  CUDA_CHECK(cudaFree(d_GU));
  CUDA_CHECK(cudaFree(d_L));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_Shalf));
  CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, saved_mode));
  return Cs_norm;
}

void compute_RR_block(int blk_idx, cardal_sdp_solver_state_t *state) {
  block_low_rank_state_t *blk = state->block_low_rank_state[blk_idx];
  int nnz_A = blk->constraint_sparse_pattern->num_nonzeros;

  if (nnz_A > 0) {
    double sddmm_alpha = 1.0;
    double sddmm_beta = 0.0;

    CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle, blk->stream));

    CUSPARSE_CHECK(
        cusparseSDDMM(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_TRANSPOSE, &sddmm_alpha, blk->matR,
                      blk->matR, &sddmm_beta, blk->matSpA, CUDA_R_64F,
                      CUSPARSE_SDDMM_ALG_DEFAULT, blk->sddmm_buffer_A));

    int blocks = (nnz_A + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scatter_sddmm_to_global_kernel<<<blocks, THREADS_PER_BLOCK, 0, blk->stream>>>(
        blk->constraint_sparse_pattern->val, blk->compat_mapping,
        state->primal_solution, nnz_A);
  }
}

// Shared implementation of the three matSpS-updating operators:
//   DUAL_SLACK   : q1 = y,           matSpS_percone = objective + Ay
//   AL_GRADIENT  : q1 = y + rho*q0,  matSpS_percone = objective + Ay
//   PENALTY_ONLY : q1 = rho*q0,      matSpS_percone = Ay only (batched skipped)
typedef enum {
  UPDATE_S_DUAL_SLACK,
  UPDATE_S_AL_GRADIENT,
  UPDATE_S_PENALTY_ONLY,
} update_S_mode_t;

static void update_S_impl(cardal_sdp_solver_state_t *state,
                          update_S_mode_t mode) {
  // 1. Prepare q1 according to mode.
  switch (mode) {
  case UPDATE_S_DUAL_SLACK:
    CUDA_CHECK(cudaMemcpyAsync(state->q1, state->dual_solution,
                               state->num_constraints * sizeof(double),
                               cudaMemcpyDeviceToDevice));
    break;
  case UPDATE_S_AL_GRADIENT: {
    int blocks_m =
        (state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_q1_kernel<<<blocks_m, THREADS_PER_BLOCK>>>(
        state->q1, state->dual_solution, state->q0, state->penalty_coef,
        state->num_constraints);
    break;
  }
  case UPDATE_S_PENALTY_ONLY:
    CUDA_CHECK(cudaMemcpyAsync(state->q1, state->q0,
                               state->num_constraints * sizeof(double),
                               cudaMemcpyDeviceToDevice));
    CUBLAS_CHECK(cublasDscal(state->blas_handle, state->num_constraints,
                             &state->penalty_coef, state->q1, 1));
    break;
  }

  // 2. dual_product = A^T q1 (common to all modes).
  double alpha_spmv = 1.0;
  double beta_spmv = 0.0;
  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_spmv,
      state->matAt, state->vec_q1, &beta_spmv, state->vec_dual_prod,
      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

  CUDA_CHECK(cudaStreamSynchronize(0));

  // 3. Per-batch: seed matSpS then add Ay contribution.
  for (int bi = 0; bi < state->n_batches; bi++) {
    const block_low_rank_state_t *batch =
        state->block_low_rank_state[state->batch_leaders[bi]];
    if (batch->kind == CONE_BATCH_KIND_CUSTOM) {
      // Batched cones are not overridden by the penalty-only path.
      if (mode == UPDATE_S_PENALTY_ONLY)
        continue;
      launch_batched_copy_objval_to_spS(batch, 0);
      launch_batched_add_Ay_to_spS(batch, state->dual_product, 0);
      continue;
    }
    block_low_rank_state_t *blk =
        state->block_low_rank_state[state->batch_leaders[bi]];
    int nnz_Union =
        blk->objective_union_constraint_sparse_pattern->num_nonzeros;
    int nnz_A = blk->constraint_sparse_pattern->num_nonzeros;
    if (nnz_Union == 0)
      continue;
    CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle, blk->stream));
    if (mode == UPDATE_S_PENALTY_ONLY) {
      // No objective contribution; matSpS starts at zero.
      CUDA_CHECK(cudaMemsetAsync(
          blk->objective_union_constraint_sparse_pattern->val, 0,
          nnz_Union * sizeof(double), blk->stream));
    } else {
      CUDA_CHECK(
          cudaMemcpyAsync(blk->objective_union_constraint_sparse_pattern->val,
                          blk->objective_val, nnz_Union * sizeof(double),
                          cudaMemcpyDeviceToDevice, blk->stream));
    }
    if (nnz_A > 0) {
      int blocks_A = (nnz_A + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      add_Ay_to_S_kernel<<<blocks_A, THREADS_PER_BLOCK, 0, blk->stream>>>(
          blk->objective_union_constraint_sparse_pattern->val,
          state->dual_product, blk->compat_mapping,
          blk->constraint_to_union_mapping, nnz_A);
    }
  }
  CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle, 0));
  for (int i = 0; i < state->cone_stream_pool_size; i++)
    CUDA_CHECK(cudaStreamSynchronize(state->cone_stream_pool[i]));
}

void update_dual_slack_S(cardal_sdp_solver_state_t *state) {
  update_S_impl(state, UPDATE_S_DUAL_SLACK);
}

void update_al_gradient_S(cardal_sdp_solver_state_t *state) {
  update_S_impl(state, UPDATE_S_AL_GRADIENT);
}

// matSpS = rho * A*(q0): penalty-only operator. PERCONE blocks only;
// batched blocks keep whatever update_al_gradient_S left in.
void update_penalty_only_S(cardal_sdp_solver_state_t *state) {
  update_S_impl(state, UPDATE_S_PENALTY_ONLY);
}

// rho * ||A*q0||_2. Clobbers q1 and dual_product.
double compute_penalty_perturbation_fnorm(cardal_sdp_solver_state_t *state) {
  cublasPointerMode_t saved_mode;
  CUBLAS_CHECK(cublasGetPointerMode(state->blas_handle, &saved_mode));
  CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));
  CUDA_CHECK(cudaMemcpyAsync(state->q1, state->q0,
                             state->num_constraints * sizeof(double),
                             cudaMemcpyDeviceToDevice));
  double alpha = 1.0, beta = 0.0;
  CUSPARSE_CHECK(cusparseSpMV(
      state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      state->matAt, state->vec_q1, &beta, state->vec_dual_prod, CUDA_R_64F,
      CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));
  double nrm = 0.0;
  CUBLAS_CHECK(cublasDnrm2(state->blas_handle, state->n_active_vars,
                           state->dual_product, 1, &nrm));
  CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, saved_mode));
  return state->penalty_coef * nrm;
}

void compute_RD_DD_block(int blk_idx, cardal_sdp_solver_state_t *state) {
  block_low_rank_state_t *blk = state->block_low_rank_state[blk_idx];
  double alpha = 1.0, beta = 0.0;

  CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle, blk->stream));
  CUBLAS_CHECK(cublasSetStream(state->blas_handle, blk->stream));

  int nnz_a = blk->constraint_sparse_pattern->num_nonzeros;
  if (nnz_a > 0) {
    int blocks = (nnz_a + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CUSPARSE_CHECK(
        cusparseSDDMM(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_TRANSPOSE, &alpha, blk->matD,
                      blk->matD, &beta, blk->matSpA, CUDA_R_64F,
                      CUSPARSE_SDDMM_ALG_DEFAULT, blk->sddmm_buffer_A));
    scatter_sddmm_to_global_kernel<<<blocks, THREADS_PER_BLOCK, 0, blk->stream>>>(
        blk->constraint_sparse_pattern->val, blk->compat_mapping,
        state->primal_direct_double, nnz_a);

    CUSPARSE_CHECK(
        cusparseSDDMM(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_TRANSPOSE, &alpha, blk->matR,
                      blk->matD, &beta, blk->matSpA, CUDA_R_64F,
                      CUSPARSE_SDDMM_ALG_DEFAULT, blk->sddmm_buffer_A));
    scatter_sddmm_to_global_kernel<<<blocks, THREADS_PER_BLOCK, 0, blk->stream>>>(
        blk->constraint_sparse_pattern->val, blk->compat_mapping,
        state->primal_direct_solution_cross, nnz_a);
  }

  int nnz_u = blk->objective_union_constraint_sparse_pattern->num_nonzeros;
  if (nnz_u > 0) {
    CUSPARSE_CHECK(
        cusparseSDDMM(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_TRANSPOSE, &alpha, blk->matD,
                      blk->matD, &beta, blk->matSpS, CUDA_R_64F,
                      CUSPARSE_SDDMM_ALG_DEFAULT, blk->sddmm_buffer_C));

    CUBLAS_CHECK(cublasDdot(state->blas_handle, nnz_u,
                            blk->objective_union_constraint_sparse_pattern->val,
                            1, blk->objective_val, 1,
                            &state->d_p2_per_cone[blk_idx]));
  } else {
    CUDA_CHECK(cudaMemsetAsync(&state->d_p2_per_cone[blk_idx], 0,
                               sizeof(double), blk->stream));
  }
}

__global__ void compute_lp_line_search_kernel(const double *__restrict__ v,
                                              const double *__restrict__ dv,
                                              double *__restrict__ primal_rd,
                                              double *__restrict__ primal_d,
                                              int lp_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < lp_dim) {
    primal_rd[idx] = v[idx] * dv[idx];
    primal_d[idx] = dv[idx] * dv[idx];
  }
}

__global__ void elementwise_multiply_kernel(const double *__restrict__ in,
                                            const double *__restrict__ scale,
                                            double *__restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = in[i] * scale[i];
}

__global__ void
elementwise_multiply_inplace_kernel(double *__restrict__ inout,
                                    const double *__restrict__ scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    inout[i] *= scale[i];
}

__global__ void
elementwise_multiply_scaled_kernel(double *__restrict__ inout,
                                   const double *__restrict__ scale,
                                   double scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    inout[i] = inout[i] * scale[i] * scalar;
}

void unscaled_dual_spmv(cardal_sdp_solver_state_t *state,
                        block_low_rank_state_t *blk, double *d_in,
                        double *d_out, double *d_scratch,
                        cusparseDnVecDescr_t vec_in,
                        cusparseDnVecDescr_t vec_out, void *dBuffer) {
  int dim = blk->dim;
  double spmv_alpha = 1.0, spmv_beta = 0.0;
  if (blk->psd_cone_rescaling != NULL) {
    int n_blocks_div = (dim + 255) / 256;
    cudaStream_t stream;
    CUBLAS_CHECK(cublasGetStream(state->blas_handle, &stream));
    elementwise_multiply_kernel<<<n_blocks_div, 256, 0, stream>>>(
        d_in, blk->psd_cone_rescaling, d_scratch, dim);
    CUSPARSE_CHECK(cusparseDnVecSetValues(vec_in, d_scratch));
    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha,
                                blk->matSpS, vec_in, &spmv_beta, vec_out,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                dBuffer));
    elementwise_multiply_inplace_kernel<<<n_blocks_div, 256, 0, stream>>>(
        d_out, blk->psd_cone_rescaling, dim);
    CUSPARSE_CHECK(cusparseDnVecSetValues(vec_in, d_in));
  } else {
    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha,
                                blk->matSpS, vec_in, &spmv_beta, vec_out,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                dBuffer));
  }
}
void populate_state_scaling_fields(cardal_sdp_solver_state_t *state,
                                   const rescale_info_t *info) {
  if (info == NULL) {
    state->constraint_rescaling = NULL;
    state->lp_variable_rescaling = NULL;
    state->q0_unscaled_buf = NULL;
    state->objective_vector_rescaling = 1.0;
    state->right_hand_side_rescaling = 1.0;
    state->unscaled_right_hand_side_norm = state->right_hand_side_norm;
    state->unscaled_objective_vector_norm = state->objective_vector_norm;
    return;
  }

  int m = state->num_constraints;
  if (m > 0 && info->constraint_rescaling != NULL) {
    CUDA_CHECK(
        cudaMalloc(&state->constraint_rescaling, (size_t)m * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(state->constraint_rescaling,
                          info->constraint_rescaling,
                          (size_t)m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMalloc(&state->q0_unscaled_buf, (size_t)m * sizeof(double)));
  } else {
    state->constraint_rescaling = NULL;
    state->q0_unscaled_buf = NULL;
  }

  int lp_dim = state->lp_dim;
  if (lp_dim > 0 && info->lp_variable_rescaling != NULL) {
    CUDA_CHECK(cudaMalloc(&state->lp_variable_rescaling,
                          (size_t)lp_dim * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(state->lp_variable_rescaling,
                          info->lp_variable_rescaling,
                          (size_t)lp_dim * sizeof(double),
                          cudaMemcpyHostToDevice));
  } else {
    state->lp_variable_rescaling = NULL;
  }

  state->objective_vector_rescaling = info->objective_vector_rescaling;
  state->right_hand_side_rescaling = info->right_hand_side_rescaling;
  state->unscaled_right_hand_side_norm = info->unscaled_right_hand_side_norm;
  state->unscaled_objective_vector_norm = info->unscaled_objective_vector_norm;
}

double compute_unscaled_q0_l2_norm(cardal_sdp_solver_state_t *state) {
  int m = state->num_constraints;
  const int threads = 256;
  int blocks = (m + threads - 1) / threads;

  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(state->blas_handle, &stream));
  elementwise_multiply_kernel<<<blocks, threads, 0, stream>>>(
      state->q0, state->constraint_rescaling, state->q0_unscaled_buf, m);

  double norm = 0.0;
  CUBLAS_CHECK(cublasDnrm2(state->blas_handle, m, state->q0_unscaled_buf, 1,
                           &norm));
  double inv_rhs = (state->right_hand_side_rescaling > 0.0)
                       ? (1.0 / state->right_hand_side_rescaling)
                       : 1.0;
  return inv_rhs * norm;
}

void populate_block_psd_cone_rescaling(block_low_rank_state_t *blk_state,
                                       int n_k, int nnz_Union,
                                       const int *h_row_ptr_U,
                                       const int *h_col_ind_U,
                                       const double *d_per_row) {
  if (d_per_row == NULL) {
    blk_state->psd_cone_rescaling = NULL;
    blk_state->vec_psd_cone_rescaling = NULL;
    return;
  }

  CUDA_CHECK(
      cudaMalloc(&blk_state->psd_cone_rescaling, (size_t)n_k * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(blk_state->psd_cone_rescaling, d_per_row,
                        (size_t)n_k * sizeof(double), cudaMemcpyHostToDevice));

  if (nnz_Union <= 0) {
    blk_state->vec_psd_cone_rescaling = NULL;
    return;
  }
  double *h_vec = (double *)safe_malloc((size_t)nnz_Union * sizeof(double));
  for (int r = 0; r < n_k; r++) {
    for (int idx = h_row_ptr_U[r]; idx < h_row_ptr_U[r + 1]; idx++) {
      int c = h_col_ind_U[idx];
      h_vec[idx] = d_per_row[r] * d_per_row[c];
    }
  }
  CUDA_CHECK(cudaMalloc(&blk_state->vec_psd_cone_rescaling,
                        (size_t)nnz_Union * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(blk_state->vec_psd_cone_rescaling, h_vec,
                        (size_t)nnz_Union * sizeof(double),
                        cudaMemcpyHostToDevice));
  free(h_vec);
}

void unscale_result(const rescale_info_t *info,
                    const cardal_sdp_solver_state_t *state,
                    sdp_result_t *result) {
  if (info == NULL || result == NULL || state == NULL)
    return;

  double tau_b = info->right_hand_side_rescaling; // ≤ 1
  double tau_c = info->objective_vector_rescaling; // ≤ 1
  double inv_sqrt_tau_b = (tau_b > 0.0) ? 1.0 / sqrt(tau_b) : 1.0;
  double inv_tau_b = (tau_b > 0.0) ? 1.0 / tau_b : 1.0;
  double inv_tau_c = (tau_c > 0.0) ? 1.0 / tau_c : 1.0;

  // PSD blocks: R column-major; R[i,j] at block_off + j*n_k + i.
  long long block_off = 0;
  int psd_offset = 0;
  if (result->low_rank_primal_solution != NULL) {
    for (int b = 0; b < state->n_blks; b++) {
      int n_k = state->blk_dims[b];
      int rank_b = state->rank_list[b];
      double *Rb = result->low_rank_primal_solution + block_off;
      for (int i = 0; i < n_k; i++) {
        double d_i = (info->psd_cone_rescaling != NULL)
                         ? info->psd_cone_rescaling[psd_offset + i]
                         : 1.0;
        double factor = inv_sqrt_tau_b / d_i;
        for (int j = 0; j < rank_b; j++)
          Rb[(long long)j * n_k + i] *= factor;
      }
      block_off += (long long)n_k * rank_b;
      psd_offset += n_k;
    }

    // LP variables sit at state->lp_solution_offset in the same buffer.
    if (state->lp_dim > 0) {
      double *lp = result->low_rank_primal_solution + state->lp_solution_offset;
      for (int i = 0; i < state->lp_dim; i++) {
        double d_i = (info->lp_variable_rescaling != NULL)
                         ? info->lp_variable_rescaling[i]
                         : 1.0;
        lp[i] *= inv_tau_b / d_i;
      }
    }
  }

  // Dual: λ_orig[i] = λ_scaled[i] / (D_con[i] · τ_c).
  if (result->dual_solution != NULL && info->constraint_rescaling != NULL) {
    for (int i = 0; i < state->num_constraints; i++) {
      double D_con_i = info->constraint_rescaling[i];
      double factor = (D_con_i > 0.0) ? (inv_tau_c / D_con_i) : inv_tau_c;
      result->dual_solution[i] *= factor;
    }
  } else if (result->dual_solution != NULL) {
    if (inv_tau_c != 1.0) {
      for (int i = 0; i < state->num_constraints; i++)
        result->dual_solution[i] *= inv_tau_c;
    }
  }
}
