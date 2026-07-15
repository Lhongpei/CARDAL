/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once

#include "sdp_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int n;
  int nnz;
  int *row;
  int *col;
  double *val;
  double *linear;
  int sense;
  double obj_const;
} qubo_problem_t;

qubo_problem_t *generate_random_qubo(int n, double density, uint64_t seed);
void free_qubo_problem(qubo_problem_t *q);

basic_sdp_t *qubo_to_sdp_chordal(const qubo_problem_t *q);
basic_sdp_t *qubo_to_sdp_dense(const qubo_problem_t *q);

typedef struct {
  int n;
  int n_cones;
  int *home_cone;
  int *home_lidx;
} qubo_layout_t;

qubo_layout_t *qubo_compute_layout(const qubo_problem_t *q, int chordal);
void free_qubo_layout(qubo_layout_t *l);

typedef struct {
  int n;
  int *x;
  double obj;                    // best overall (post-LS if enabled)
  int num_trials;                // T+1 total candidates (T hyperplanes + 1 thr)
  int best_trial;                // 0..T-1 = hyperplane, T = threshold
  double time_sec;

  double obj_hyperplane_pre_ls;  // best across random hyperplanes only
  double obj_threshold_pre_ls;   // threshold (deterministic 0.5 cut) candidate
  double obj_round_only;         // best of hyperplane+threshold, NO LS applied
} qubo_round_result_t;

qubo_round_result_t *qubo_round_gpu(const qubo_problem_t *q,
                                    const qubo_layout_t *layout,
                                    const int *blk_dims, const int *rank_list,
                                    const double *R_host, long long R_length,
                                    int num_trials, int max_ls_iters,
                                    uint64_t seed);

void free_qubo_round_result(qubo_round_result_t *r);

typedef struct {
  int chordal;          // 1 = chordal multi-cone, 0 = single big cone
  int num_round_trials; // <=0 = default (4096)
  int num_ls_iters;     // <0 = default (~20*n); 0 = skip LS; >0 = exact cap
  uint64_t round_seed;  // 0 = built-in default
  int is_mpi;           // 1 = use distributed_optimize
  const char *output_dir;
  const char *instance_name;
} qubo_run_config_t;

int qubo_run_e2e(qubo_problem_t *q, const qubo_run_config_t *cfg,
                 cardal_parameters_t *params);

#ifdef __cplusplus
}
#endif
