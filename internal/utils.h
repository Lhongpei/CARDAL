/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once

#include "internal_types.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorName(err));                                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS Error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cublasGetStatusName(status));                                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUSPARSE_CHECK(call)                                                   \
  do {                                                                         \
    cusparseStatus_t status = call;                                            \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "cuSPARSE Error at %s:%d: %s\n", __FILE__, __LINE__,     \
              cusparseGetErrorName(status));                                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define THREADS_PER_BLOCK 256
#define ALLOC_AND_COPY(dest, src, bytes)                                       \
  CUDA_CHECK(cudaMalloc(&dest, bytes));                                        \
  CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice));

#define ALLOC_AND_COPY_CSR(dest_csr, src_csr, n_rows, nnz)                     \
  do {                                                                         \
    ALLOC_AND_COPY((dest_csr)->row_ptr, (src_csr)->row_ptr,                    \
                   ((n_rows) + 1) * sizeof(int));                              \
                                                                               \
    ALLOC_AND_COPY((dest_csr)->col_ind, (src_csr)->col_ind,                    \
                   (nnz) * sizeof(int));                                       \
                                                                               \
    ALLOC_AND_COPY((dest_csr)->val, (src_csr)->val, (nnz) * sizeof(double));   \
  } while (0)

#define ALLOC_ZERO(dest, bytes)                                                \
  CUDA_CHECK(cudaMalloc(&dest, bytes));                                        \
  CUDA_CHECK(cudaMemset(dest, 0, bytes));
#ifndef CURAND_CHECK
#define CURAND_CHECK(val)                                                      \
  do {                                                                         \
    curandStatus_t status = (val);                                             \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "CURAND Error at %s:%d\n", __FILE__, __LINE__);          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif
extern const double HOST_ONE;
extern const double HOST_ZERO;

// Process-wide log verbosity, settable from the cli.
//   0 = silent, 1 = banner + summary, 2 = + subtitle sections / iter table,
//   3 = + debug (parser details, mpi/grid info, augmentation, warnings, ...).
extern int g_log_verbose;
#define LOG_V(level) (g_log_verbose >= (level))
#define LOG_DBG(...)                                                           \
  do {                                                                         \
    if (LOG_V(3))                                                              \
      printf(__VA_ARGS__);                                                     \
  } while (0)

void *safe_malloc(size_t size);

void *safe_calloc(size_t num, size_t size);

void *safe_realloc(void *ptr, size_t new_size);

void *safe_memcpy(void *dest, const void *src, size_t n);
typedef struct {
  int row;
  int col;
  double val;
} coo_triplet_t;

compressed_sdp_problem_t *convert_to_compressed(basic_sdp_t *input);

// Pataki bound: ceil((sqrt(8 m + 1) - 1)/2)
int compute_theoretical_max_rank(int m);

void compute_per_block_max_rank(const compressed_sdp_problem_t *prob,
                                int user_cap, int *out_per_blk);

int compute_initial_rank(int m);

double
compute_initial_penalty_coef(const compressed_sdp_problem_t *sdp_problem);

void set_default_parameters(cardal_parameters_t *params);

int tridiagonal_eigen_solver(int n, double *d, double *e, double *z);

void free_compressed_sdp(compressed_sdp_problem_t *prob);
void print_header(void);
void print_log_entry(const cardal_sdp_solver_state_t *state,
                     int inner_this_iter);

void print_cardal_banner(void);
void print_subtitle(const char *title);
void print_kv_str(const char *key, const char *val);
void print_kv_int(const char *key, long long val);
void print_kv_dbl(const char *key, const char *fmt, double val);

const char *termination_reason_to_string(termination_reason_t r);

void free_sdp_result(sdp_result_t *result);

void print_runtime_environment_section(int verbose_floor, int is_distributed,
                                       int world_size, int row_dims,
                                       int rank_dims, int cone_dims);
void print_parameters_section(const cardal_parameters_t *params,
                              int verbose_floor);
void print_problem_statistics_section(const compressed_sdp_problem_t *prob,
                                      int verbose_floor);
void print_optimization_footer(const sdp_result_t *result,
                               const char *summary_file_path /* nullable */,
                               int verbose_floor);

#ifdef __cplusplus
}
#endif
