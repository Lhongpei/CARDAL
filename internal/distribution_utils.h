/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once
#include "internal_types.h"
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("NCCL failure %s:%d '%s'\n", __FILE__, __LINE__,                  \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


grid_context_t initialize_parallel_context(int P_row, int P_rank, int P_cone);

compressed_sdp_problem_t *
partition_problem(const compressed_sdp_problem_t *global_prob,
                  const grid_context_t *grid);

rescale_info_t *
partition_rescale_info(const rescale_info_t *global_info,
                       const compressed_sdp_problem_t *global_prob,
                       const grid_context_t *grid);
void select_valid_grid_size(const cardal_parameters_t *params,
                            const compressed_sdp_problem_t *original_problem,
                            cardal_parameters_t *sub_params);

void big_bcast_bytes(void **buffer_ptr, size_t *size_ptr, int root,
                     MPI_Comm comm);
void serialize_compressed_sdp(const compressed_sdp_problem_t *prob,
                              void **out_buf, size_t *out_size);
compressed_sdp_problem_t *deserialize_compressed_sdp(void *buf);
void cleanup_parallel_context(grid_context_t *grid);
int *permute_global_problem_constraints(compressed_sdp_problem_t *prob,
                                        shuffle_type_t type, int block_size,
                                        int seed);
void unpermute_dual_solution(int m, const double *shuffled_global_dual,
                             double *orig_global_dual, const int *p);

void gather_sdp_result(sdp_result_t *result,
                                         cardal_sdp_solver_state_t *state);

void print_per_rank_workload(const grid_context_t *grid,
                             const compressed_sdp_problem_t *local_prob,
                             int m_total_global,
                             int initial_rank_global);

#ifdef __cplusplus
}
#endif