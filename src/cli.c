/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "cli_common.h"
#ifdef USE_MPI
#include "distribution_solver.h"
#endif
#include "generator.h"
#include "parser.h"
#include "sdp_types.h"
#include "solver.h"
#include "utils.h"
#include <errno.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

typedef enum { GEN_MAXCUT, GEN_SNL, GEN_QORDER } gen_type_t;

typedef struct {
  gen_type_t type;
  int n;
  double density; // MaxCut
  int anchors;    // SNL
  int d;          // SNL
  double R;       // SNL
  double noise;   // SNL
  int qorder_k;   // Quantum Order: number of moment matrices (k)
} generator_config_t;

static char *get_output_path(const char *output_dir, const char *instance_name,
                             const char *suffix) {
  size_t path_len =
      strlen(output_dir) + strlen(instance_name) + strlen(suffix) + 2;
  char *full_path = (char *)safe_malloc(path_len);
  snprintf(full_path, path_len, "%s/%s%s", output_dir, instance_name, suffix);
  return full_path;
}

static char *make_generator_instance_name(const generator_config_t *cfg) {
  char buf[256];
  if (cfg->type == GEN_MAXCUT) {
    snprintf(buf, sizeof(buf), "maxcut_n%d_d%.2e", cfg->n, cfg->density);
  } else if (cfg->type == GEN_QORDER) {
    snprintf(buf, sizeof(buf), "qorder_n%d_k%d", cfg->n, cfg->qorder_k);
  } else {
    snprintf(buf, sizeof(buf), "snl_n%d_a%d_d%d_R%.3f_noise%.2e", cfg->n,
             cfg->anchors, cfg->d, cfg->R, cfg->noise);
  }
  return strdup(buf);
}

static int ensure_output_dir(const char *path) {
  if (path == NULL || *path == '\0')
    return -1;
  if (mkdir(path, 0755) == 0)
    return 0;
  if (errno == EEXIST)
    return 0;
  fprintf(stderr, "Error creating output directory '%s': %s\n", path,
          strerror(errno));
  return -1;
}

static void write_cone_size_distribution(FILE *outfile,
                                         const compressed_sdp_problem_t *prob) {
  if (prob == NULL || prob->n_blks <= 0)
    return;

  int n_cones = prob->n_blks;
  int *uniq_sizes = (int *)safe_malloc(n_cones * sizeof(int));
  int *uniq_counts = (int *)safe_calloc(n_cones, sizeof(int));
  int num_distinct = 0;

  for (int i = 0; i < n_cones; i++) {
    int dim = prob->blk_dims[i];
    int found = 0;
    for (int j = 0; j < num_distinct; j++) {
      if (uniq_sizes[j] == dim) {
        uniq_counts[j]++;
        found = 1;
        break;
      }
    }
    if (!found) {
      uniq_sizes[num_distinct] = dim;
      uniq_counts[num_distinct] = 1;
      num_distinct++;
    }
  }

  fprintf(outfile, "Cone Size Distribution:\n");
  for (int i = 0; i < num_distinct; i++) {
    fprintf(outfile, "  - %d cone(s) of size %d x %d\n", uniq_counts[i],
            uniq_sizes[i], uniq_sizes[i]);
  }

  free(uniq_sizes);
  free(uniq_counts);
}

static void save_solver_summary(const sdp_result_t *result,
                                const compressed_sdp_problem_t *prob,
                                const char *output_dir,
                                const char *instance_name) {
  if (result == NULL || output_dir == NULL || instance_name == NULL)
    return;

  char *file_path = get_output_path(output_dir, instance_name, "_summary.txt");
  if (file_path == NULL)
    return;

  FILE *outfile = fopen(file_path, "w");
  if (outfile == NULL) {
    fprintf(stderr, "Error opening summary file '%s': %s\n", file_path,
            strerror(errno));
    free(file_path);
    return;
  }

  fprintf(outfile, "Instance: %s\n", instance_name);
  fprintf(outfile, "Termination Reason: %s\n",
          termination_reason_to_string(result->termination_reason));

  fprintf(outfile, "Runtime (sec): %e\n", result->cumulative_time_sec);
  if (result->rescaling_time_sec > 0.0)
    fprintf(outfile, "Rescaling Time (sec): %e\n", result->rescaling_time_sec);

  fprintf(outfile, "Iterations Count: %d\n", result->total_count);
  fprintf(outfile, "Inner Iterations Count: %d\n", result->total_inner_count);

  fprintf(outfile, "Primal Objective Value: %e\n",
          result->primal_objective_value);
  fprintf(outfile, "Dual Objective Value: %e\n", result->dual_objective_value);

  fprintf(outfile, "Absolute Primal Residual: %e\n",
          result->absolute_primal_residual);
  fprintf(outfile, "Relative Primal Residual: %e\n",
          result->relative_primal_residual);
  fprintf(outfile, "Absolute Dual Residual: %e\n",
          result->absolute_dual_residual);
  fprintf(outfile, "Relative Dual Residual: %e\n",
          result->relative_dual_residual);
  fprintf(outfile, "Absolute Objective Gap: %e\n", result->objective_gap);
  fprintf(outfile, "Relative Objective Gap: %e\n",
          result->relative_objective_gap);

  fprintf(outfile, "Rows (Constraints): %d\n", result->num_constraints);
  fprintf(outfile, "Active Variables: %d\n", result->num_variables);
  fprintf(outfile, "Constraint Matrix NNZ: %d\n", result->num_nonzeros);
  fprintf(outfile, "Burer-Monteiro Rank: %d\n", result->rank);

  if (prob != NULL) {
    fprintf(outfile, "LP Variables: %d\n", prob->lp_dim);
    fprintf(outfile, "PSD Cones: %d\n", prob->n_blks);
    write_cone_size_distribution(outfile, prob);
  }

  fclose(outfile);
  free(file_path);
}

void print_usage(const char *prog_name) {
  static const char *rule80 =
      "==============================================================="
      "=================";
  printf("%s\n", rule80);
  printf("                                 CARDAL\n");
  printf("          A Burer-Monteiro Augmented Lagrangian Method for "
         "Large-Scale\n");
  printf("                         SDPs on Multi-GPU Systems\n");
  printf("\n");
  printf("                     Hongpei Li (ishongpeili@gmail.com)\n");
  printf("%s\n", rule80);
  printf("Usage: %s [OPTIONS]\n\n", prog_name);

  printf("Input Options (Choose one):\n");
  printf("  -f, --file <path>       Read problem from file (.dat-s, "
         ".dat-s.gz, .mat, or .npz;\n");
  printf("                          format auto-detected)\n");
  printf("  -g, --gen <string>      Generate random SDP problem.\n");
  printf("                          Format: type,param1,param2,...\n");
  printf("                          Examples:\n");
  printf("                            -g maxcut,30000,1e-3\n");
  printf("                              (type=maxcut, nodes=30000, "
         "density=1e-3)\n");
  printf("                            -g snl,10000,4,2,0.05,0.0\n");
  printf("                              (type=snl, nodes=10000, anchors=4, "
         "dim=2, R=0.05, noise=0.0)\n");
  printf("                            -g quantum_order,64,3\n");
  printf("                              (type=quantum_order, "
         "moment-matrix size N=64, levels k=3)\n");
  printf("\n");
  printf("                          (For QUBO problems use cardal_qubo.)\n\n");

  cli_print_solver_param_help();
  printf("%s\n", rule80);
}

int parse_arguments(int argc, char *argv[], char **filename,
                    char **output_dir, generator_config_t *gen_config,
                    cardal_parameters_t *params) {
  struct option long_options[] = {{"file", required_argument, 0, 'f'},
                                  {"gen", required_argument, 0, 'g'},
                                  {"rank", required_argument, 0, 'r'},
                                  {"max-rank", required_argument, 0, 'R'},
                                  {"eps", required_argument, 0, 'e'},
                                  {"inner-iters", required_argument, 0, 'i'},
                                  {"outer-iters", required_argument, 0, 'o'},
                                  {"penalty-fac", required_argument, 0, 'p'},
                                  {"init-penalty", required_argument, 0, 'c'},
                                  {"max-penalty", required_argument, 0, 'M'},
                                  {"time-limit", required_argument, 0, 'T'},
                                  {"lbfgs-hist", required_argument, 0, 'L'},
                                  {"verbose", required_argument, 0, 'v'},
                                  {"help", no_argument, 0, 'h'},
                                  {"grid_size", required_argument, 0, 'z'},
                                  {"output-dir", required_argument, 0, 'O'},
                                  {"l_inf_ruiz_iter", required_argument, 0,
                                   CLI_OPT_L_INF_RUIZ_ITER},
                                  {"no_pock_chambolle", no_argument, 0,
                                   CLI_OPT_NO_POCK_CHAMBOLLE},
                                  {"pock_chambolle_alpha", required_argument, 0,
                                   CLI_OPT_POCK_CHAMBOLLE_ALPHA},
                                  {"no_bound_obj_rescaling", no_argument, 0,
                                   CLI_OPT_NO_BOUND_OBJ_RESCALING},
                                  {"no_scaling", no_argument, 0,
                                   CLI_OPT_NO_SCALING},
                                  {"psd_scale_mode", required_argument, 0,
                                   CLI_OPT_PSD_SCALE_MODE},
                                  {"shuffle", required_argument, 0,
                                   CLI_OPT_SHUFFLE_MODE},
                                  {0, 0, 0, 0}};

  char opt_string[64];
  snprintf(opt_string, sizeof(opt_string), "f:g:hO:%s", CLI_SOLVER_OPT_STRING);

  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, opt_string, long_options,
                            &option_index)) != -1) {
    if (cli_apply_solver_opt(opt, optarg, params))
      continue;
    switch (opt) {
    case 'f':
      *filename = optarg;
      break;
    case 'g': {
      char *arg_copy = strdup(optarg);
      char *token = strtok(arg_copy, ",");
      if (token == NULL)
        break;

      if (strcmp(token, "maxcut") == 0) {
        gen_config->type = GEN_MAXCUT;
        token = strtok(NULL, ",");
        if (token)
          gen_config->n = atoi(token);
        token = strtok(NULL, ",");
        if (token)
          gen_config->density = atof(token);
      } else if (strcmp(token, "snl") == 0) {
        gen_config->type = GEN_SNL;
        token = strtok(NULL, ",");
        if (token)
          gen_config->n = atoi(token);
        token = strtok(NULL, ",");
        if (token)
          gen_config->anchors = atoi(token);
        token = strtok(NULL, ",");
        if (token)
          gen_config->d = atoi(token);
        token = strtok(NULL, ",");
        if (token)
          gen_config->R = atof(token);
        token = strtok(NULL, ",");
        if (token)
          gen_config->noise = atof(token);
      } else if (strcmp(token, "quantum_order") == 0 ||
                 strcmp(token, "qorder") == 0) {
        gen_config->type = GEN_QORDER;
        token = strtok(NULL, ",");
        if (token)
          gen_config->n = atoi(token);
        token = strtok(NULL, ",");
        if (token)
          gen_config->qorder_k = atoi(token);
        if (gen_config->n < 2 || gen_config->qorder_k < 2) {
          fprintf(stderr,
                  "quantum_order requires N>=2 and k>=2 (got N=%d, k=%d).\n",
                  gen_config->n, gen_config->qorder_k);
          free(arg_copy);
          return 1;
        }
      } else if (strcmp(token, "qubo") == 0) {
        fprintf(stderr,
                "QUBO problems are no longer handled by cardal; use "
                "cardal_qubo instead.\n");
        free(arg_copy);
        return 1;
      } else {
        fprintf(stderr, "Unknown generator type: %s\n", token);
        free(arg_copy);
        return 1;
      }
      free(arg_copy);
      break;
    }
    case 'h':
      print_usage(argv[0]);
      return 1;
    case 'O':
      *output_dir = optarg;
      break;
    default:
      print_usage(argv[0]);
      return 1;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
#ifdef USE_MPI
  int is_mpi = cli_is_mpi();
  int rank = 0;

  if (is_mpi) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized)
      MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
#else
  int rank = 0;
#endif

  char *filename = NULL;
  char *output_dir = NULL;
  generator_config_t gen_config = {.type = GEN_MAXCUT,
                                   .n = 30000,
                                   .density = 1e-3,
                                   .anchors = 4,
                                   .d = 2,
                                   .R = 0.05,
                                   .noise = 0.0,
                                   .qorder_k = 3};

  cardal_parameters_t *params =
      (cardal_parameters_t *)safe_malloc(sizeof(cardal_parameters_t));
  set_default_parameters(params);

  if (parse_arguments(argc, argv, &filename, &output_dir, &gen_config,
                      params) != 0) {
    free(params);
#ifdef USE_MPI
    if (is_mpi)
      MPI_Finalize();
#endif
    return EXIT_FAILURE;
  }

  g_log_verbose = params->verbose;
  char instance_label_buf[256] = "";
  if (filename) {
    snprintf(instance_label_buf, sizeof(instance_label_buf), "File  %s",
             filename);
  } else if (gen_config.type == GEN_MAXCUT) {
    snprintf(instance_label_buf, sizeof(instance_label_buf),
             "Gen Max-Cut  n=%d  density=%.2e", gen_config.n,
             gen_config.density);
  } else if (gen_config.type == GEN_QORDER) {
    snprintf(instance_label_buf, sizeof(instance_label_buf),
             "Gen Quantum-Order  N=%d  k=%d", gen_config.n,
             gen_config.qorder_k);
  } else {
    snprintf(instance_label_buf, sizeof(instance_label_buf),
             "Gen SNL  n=%d  anchors=%d  d=%d  R=%.3f  noise=%.2e",
             gen_config.n, gen_config.anchors, gen_config.d, gen_config.R,
             gen_config.noise);
  }
  params->instance_label = instance_label_buf;

  char *summary_path = NULL;
  char *instance_name = NULL;
  if (output_dir != NULL) {
    instance_name = filename ? cli_extract_instance_name(filename)
                             : make_generator_instance_name(&gen_config);
    if (instance_name != NULL && ensure_output_dir(output_dir) == 0)
      summary_path = get_output_path(output_dir, instance_name, "_summary.txt");
  }
  params->summary_file_path = summary_path;

  basic_sdp_t *input = NULL;
  compressed_sdp_problem_t *prob = NULL;

  if (rank == 0) {
    if (filename) {
      LOG_DBG("\n[Parsing] Loading from file: %s\n", filename);
      input = sdp_problem_parse(filename);
      if (!input) {
        fprintf(stderr, "Fatal Error: Failed to load SDP from %s\n", filename);
        free(params);
#ifdef USE_MPI
        if (is_mpi)
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
        return EXIT_FAILURE;
      }
    } else {
      LOG_DBG("\n[Generating] Building random SDP instance...\n");
      if (gen_config.type == GEN_MAXCUT) {
        input = generate_maxcut_sdp_huge(gen_config.n, gen_config.density);
      } else if (gen_config.type == GEN_SNL) {
        input = generate_snl_sdp(gen_config.n, gen_config.anchors, gen_config.d,
                                 gen_config.R, gen_config.noise);
      } else if (gen_config.type == GEN_QORDER) {
        input = generate_order_sdp(gen_config.n, gen_config.qorder_k);
      }
    }

    LOG_DBG("[Compression] Converting to compressed CSR form...\n");
    prob = convert_to_compressed(input);
  }

  sdp_result_t *result = NULL;
#ifdef USE_MPI
  if (is_mpi) {
    MPI_Barrier(MPI_COMM_WORLD);
    result = distributed_optimize(prob, params);
  } else {
    result = optimize(prob, params);
  }
#else
  result = optimize(prob, params);
#endif

  if (rank == 0 && result != NULL && summary_path != NULL &&
      instance_name != NULL) {
    save_solver_summary(result, prob, output_dir, instance_name);
  }

  free(summary_path);
  free(instance_name);

  free_sdp_result(result);

  if (prob != NULL) {
    free_compressed_sdp(prob);
  }
  if (input != NULL) {
    free_basic_sdp(input);
  }
  free(params);

#ifdef USE_MPI
  if (is_mpi) {
    MPI_Finalize();
  }
#endif

  return EXIT_SUCCESS;
}
