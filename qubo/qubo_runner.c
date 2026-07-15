/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "qubo.h"

#include "distribution_solver.h"
#include "parser.h"
#include "solver.h"
#include "utils.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

static char *qubo_join_path(const char *dir, const char *base,
                            const char *suffix) {
  size_t need = strlen(dir) + 1 + strlen(base) + strlen(suffix) + 1;
  char *p = (char *)safe_malloc(need);
  snprintf(p, need, "%s/%s%s", dir, base, suffix);
  return p;
}

static int qubo_ensure_dir(const char *path) {
  if (!path || !*path)
    return -1;
  if (mkdir(path, 0755) == 0)
    return 0;
  if (errno == EEXIST)
    return 0;
  fprintf(stderr, "qubo: cannot create '%s': %s\n", path, strerror(errno));
  return -1;
}

static void qubo_write_cone_size_distribution(
    FILE *fp, const compressed_sdp_problem_t *prob) {
  if (!prob || prob->n_blks <= 0)
    return;
  int K = prob->n_blks;
  int *sizes = (int *)safe_malloc((size_t)K * sizeof(int));
  int *counts = (int *)safe_calloc((size_t)K, sizeof(int));
  int distinct = 0;
  for (int i = 0; i < K; i++) {
    int dim = prob->blk_dims[i];
    int found = 0;
    for (int j = 0; j < distinct; j++) {
      if (sizes[j] == dim) {
        counts[j]++;
        found = 1;
        break;
      }
    }
    if (!found) {
      sizes[distinct] = dim;
      counts[distinct] = 1;
      distinct++;
    }
  }
  fprintf(fp, "Cone Size Distribution:\n");
  for (int i = 0; i < distinct; i++)
    fprintf(fp, "  - %d cone(s) of size %d x %d\n", counts[i], sizes[i],
            sizes[i]);
  free(sizes);
  free(counts);
}

static inline double qubo_restore_obj(double internal, int sense,
                                      double obj_const) {
  return (double)sense * internal + obj_const;
}

static void qubo_write_summary(const char *path, const sdp_result_t *result,
                               const compressed_sdp_problem_t *prob,
                               const char *instance_name,
                               const qubo_round_result_t *rr, int sense,
                               double obj_const) {
  if (!path || !result)
    return;
  FILE *fp = fopen(path, "w");
  if (!fp) {
    fprintf(stderr, "qubo: cannot open summary '%s': %s\n", path,
            strerror(errno));
    return;
  }

  fprintf(fp, "Instance: %s\n", instance_name ? instance_name : "qubo");
  fprintf(fp, "Termination Reason: %s\n",
          termination_reason_to_string(result->termination_reason));
  fprintf(fp, "Runtime (sec): %e\n", result->cumulative_time_sec);
  if (result->rescaling_time_sec > 0.0)
    fprintf(fp, "Rescaling Time (sec): %e\n", result->rescaling_time_sec);
  fprintf(fp, "Iterations Count: %d\n", result->total_count);
  fprintf(fp, "Inner Iterations Count: %d\n", result->total_inner_count);
  fprintf(fp, "Primal Objective Value: %e\n", result->primal_objective_value);
  fprintf(fp, "Dual Objective Value: %e\n", result->dual_objective_value);
  fprintf(fp, "Absolute Primal Residual: %e\n",
          result->absolute_primal_residual);
  fprintf(fp, "Relative Primal Residual: %e\n",
          result->relative_primal_residual);
  fprintf(fp, "Absolute Dual Residual: %e\n", result->absolute_dual_residual);
  fprintf(fp, "Relative Dual Residual: %e\n", result->relative_dual_residual);
  fprintf(fp, "Absolute Objective Gap: %e\n", result->objective_gap);
  fprintf(fp, "Relative Objective Gap: %e\n", result->relative_objective_gap);
  fprintf(fp, "Rows (Constraints): %d\n", result->num_constraints);
  fprintf(fp, "Active Variables: %d\n", result->num_variables);
  fprintf(fp, "Constraint Matrix NNZ: %d\n", result->num_nonzeros);
  fprintf(fp, "Burer-Monteiro Rank: %d\n", result->rank);
  if (prob) {
    fprintf(fp, "LP Variables: %d\n", prob->lp_dim);
    fprintf(fp, "PSD Cones: %d\n", prob->n_blks);
    qubo_write_cone_size_distribution(fp, prob);
  }
  if (rr) {
    int ones = 0;
    for (int i = 0; i < rr->n; i++)
      ones += (rr->x[i] != 0);
    double rounded_orig = qubo_restore_obj(rr->obj, sense, obj_const);
    double sdp_bound_orig =
        qubo_restore_obj(result->primal_objective_value, sense, obj_const);
    fprintf(fp, "QUBO Sense: %s\n", sense < 0 ? "maximize" : "minimize");
    fprintf(fp, "QUBO Objective Constant: %e\n", obj_const);
    fprintf(fp, "QUBO Rounded Obj: %e\n", rounded_orig);
    fprintf(fp, "QUBO SDP %s Bound: %e\n",
            sense < 0 ? "Upper" : "Lower", sdp_bound_orig);
    fprintf(fp, "QUBO Integrality Gap: %e\n",
            fabs(sdp_bound_orig - rounded_orig));
    fprintf(fp, "QUBO Rounding Trials: %d\n", rr->num_trials);
    fprintf(fp, "QUBO Rounding Best Trial: %d\n", rr->best_trial);
    fprintf(fp, "QUBO Rounding Time (sec): %e\n", rr->time_sec);
    fprintf(fp, "QUBO Sum x_i: %d / %d\n", ones, rr->n);
  }
  fclose(fp);
}

static void qubo_print_rounding(const qubo_round_result_t *rr,
                                const sdp_result_t *result, int sense,
                                double obj_const) {
  if (!rr || !result)
    return;
  int ones = 0;
  for (int i = 0; i < rr->n; i++)
    ones += (rr->x[i] != 0);
  double rounded_orig = qubo_restore_obj(rr->obj, sense, obj_const);
  double bound_orig =
      qubo_restore_obj(result->primal_objective_value, sense, obj_const);
  double gap = fabs(bound_orig - rounded_orig);
  double rel = (fabs(rounded_orig) > 1e-12) ? gap / fabs(rounded_orig) : gap;
  const char *bound_kind = sense < 0 ? "upper" : "lower";

  printf("\n================================================================================\n");
  printf("                              QUBO Rounding\n");
  printf("================================================================================\n");
  printf("  Objective sense     : %s%s\n", sense < 0 ? "maximize" : "minimize",
         obj_const != 0.0 ? "  (+ obj_const)" : "");
  printf("  Trials              : %d (%d random hyperplanes + 1 threshold, GPU)\n",
         rr->num_trials, rr->num_trials - 1);
  if (rr->best_trial == rr->num_trials - 1)
    printf("  Best trial          : threshold (deterministic 0.5 cut)\n");
  else
    printf("  Best trial          : %d (random hyperplane)\n",
           rr->best_trial);
  printf("  Strategy breakdown (internal min-form, no sense/const applied):\n");
  printf("    Hyperplane only     : %.6e\n", rr->obj_hyperplane_pre_ls);
  printf("    Threshold only      : %.6e\n", rr->obj_threshold_pre_ls);
  printf("    Round-only (best)   : %.6e   (hyperplane+threshold, no LS)\n",
         rr->obj_round_only);
  printf("    + Tabu LS (final)   : %.6e\n", rr->obj);
  printf("  Rounded obj f(x*)   : %.6e\n", rounded_orig);
  printf("  SDP %s bound     : %.6e\n", bound_kind, bound_orig);
  printf("  Integrality gap     : %.3e (rel %.3e)\n", gap, rel);
  printf("  sum_i x_i*          : %d / %d\n", ones, rr->n);
  printf("  Rounding time (sec) : %.3f\n", rr->time_sec);
  printf("================================================================================\n");
}

int qubo_run_e2e(qubo_problem_t *q, const qubo_run_config_t *cfg,
                 cardal_parameters_t *params) {
  if (!cfg || !params) {
    fprintf(stderr, "qubo_run_e2e: NULL cfg/params\n");
    if (q)
      free_qubo_problem(q);
    return -1;
  }
  int rank = 0;
#ifdef USE_MPI
  if (cfg->is_mpi) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized)
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
#endif

  if (rank == 0 && !q) {
    fprintf(stderr, "qubo: no problem to solve (rank 0 got NULL)\n");
    return -1;
  }

  char *summary_path = NULL;
  if (rank == 0 && cfg->output_dir && cfg->instance_name) {
    if (qubo_ensure_dir(cfg->output_dir) == 0)
      summary_path =
          qubo_join_path(cfg->output_dir, cfg->instance_name, "_summary.txt");
  }
  params->summary_file_path = summary_path;

  int sense = (rank == 0 && q) ? q->sense : 1;
  double obj_const = (rank == 0 && q) ? q->obj_const : 0.0;

  basic_sdp_t *sdp = NULL;
  compressed_sdp_problem_t *prob = NULL;

  if (rank == 0) {
    sdp = cfg->chordal ? qubo_to_sdp_chordal(q) : qubo_to_sdp_dense(q);
    if (!sdp) {
      free_qubo_problem(q);
      free(summary_path);
      return -1;
    }
    LOG_DBG("[Compression] Converting to compressed CSR form...\n");
    prob = convert_to_compressed(sdp);
  }

  sdp_result_t *result = NULL;
#ifdef USE_MPI
  if (cfg->is_mpi) {
    MPI_Barrier(MPI_COMM_WORLD);
    result = distributed_optimize(prob, params);
  } else {
    result = optimize(prob, params);
  }
#else
  result = optimize(prob, params);
#endif

  qubo_round_result_t *rr = NULL;
  if (rank == 0 && result && result->low_rank_primal_solution && prob && q) {
    qubo_layout_t *layout = qubo_compute_layout(q, cfg->chordal);
    if (layout) {
      int trials = cfg->num_round_trials > 0 ? cfg->num_round_trials : 4096;
      uint64_t seed = cfg->round_seed ? cfg->round_seed : 42ULL;
      rr = qubo_round_gpu(q, layout, prob->blk_dims, result->rank_list,
                          result->low_rank_primal_solution,
                          result->low_rank_solution_length, trials,
                          cfg->num_ls_iters, seed);
      free_qubo_layout(layout);
    }
    if (rr)
      qubo_print_rounding(rr, result, sense, obj_const);
  }

  if (rank == 0 && summary_path && result)
    qubo_write_summary(summary_path, result, prob, cfg->instance_name, rr,
                       sense, obj_const);

  if (rr)
    free_qubo_round_result(rr);
  free_sdp_result(result);
  if (prob)
    free_compressed_sdp(prob);
  if (sdp)
    free_basic_sdp(sdp);
  if (q)
    free_qubo_problem(q);
  free(summary_path);
  return 0;
}

