/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "cli_common.h"
#include "qubo.h"
#include "qubo_io.h"
#include "utils.h"

#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

static void print_usage(const char *prog) {
  static const char *rule80 =
      "==============================================================="
      "=================";
  printf("%s\n", rule80);
  printf("                              CARDAL QUBO\n");
  printf("       QUBO end-to-end: chordal SDP relaxation -> Burer-Monteiro\n");
  printf("                       -> GPU random-hyperplane rounding\n");
  printf("%s\n", rule80);
  printf("Usage: %s ( -f <file> | -n <size> -d <density> ) [OPTIONS]\n\n",
         prog);

  printf("Input (choose one):\n");
  printf("  -f, --file <path>       Read QUBO from file (auto-detects D-Wave "
         "qbsolv\n");
  printf("                          .qubo or simple triplet 'n nnz / i j v' "
         "format)\n");
  printf("  -n, --size <int>        Random QUBO: number of binary variables\n");
  printf("  -d, --density <float>   Random QUBO: off-diagonal density (0..1)\n");
  printf("\n");
  printf("QUBO Options:\n");
  printf("  -m, --mode <s>          'dense' (default) for single n+1 cone, "
         "'chordal' for multi-cone\n");
  printf("  -s, --seed <int>        Random-QUBO RNG seed (default 0)\n");
  printf("\n");
  printf("Rounding:\n");
  printf("  -t, --trials <int>      Number of random-hyperplane trials "
         "(default 4096)\n");
  printf("  -l, --ls-iters <int>    Cap on per-trial tabu/1-flip iterations "
         "(-1=default ~20*n, 0=skip LS)\n");
  printf("  -S, --round-seed <int>  Hyperplane Gaussian RNG seed "
         "(default 42)\n");
  printf("\n");
  cli_print_solver_param_help();
  printf("%s\n", rule80);
}

static char *qubo_default_instance_name(int n, double density, int chordal) {
  char buf[256];
  snprintf(buf, sizeof(buf), "qubo_n%d_d%.2e_%s", n, density,
           chordal ? "chordal" : "dense");
  return strdup(buf);
}

int main(int argc, char *argv[]) {
  int is_mpi = cli_is_mpi();

#ifdef USE_MPI
  if (is_mpi) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized)
      MPI_Init(&argc, &argv);
  }
#else
  (void)is_mpi;
#endif

  cardal_parameters_t *params =
      (cardal_parameters_t *)safe_malloc(sizeof(cardal_parameters_t));
  set_default_parameters(params);

  qubo_run_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.chordal = 0;
  cfg.num_round_trials = 4096;
  cfg.num_ls_iters = -1;
  cfg.round_seed = 42ULL;
  cfg.is_mpi = is_mpi;

  int gen_n = 0;
  double gen_density = 0.0;
  uint64_t gen_seed = 0ULL;

  char *output_dir = NULL;
  char *file_path = NULL;
  int have_n = 0, have_d = 0;

  struct option long_options[] = {
      {"file", required_argument, 0, 'f'},
      {"size", required_argument, 0, 'n'},
      {"density", required_argument, 0, 'd'},
      {"mode", required_argument, 0, 'm'},
      {"seed", required_argument, 0, 's'},
      {"trials", required_argument, 0, 't'},
      {"ls-iters", required_argument, 0, 'l'},
      {"round-seed", required_argument, 0, 'S'},
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
      {"grid_size", required_argument, 0, 'z'},
      {"output-dir", required_argument, 0, 'O'},
      {"help", no_argument, 0, 'h'},
      {"l_inf_ruiz_iter", required_argument, 0, CLI_OPT_L_INF_RUIZ_ITER},
      {"no_pock_chambolle", no_argument, 0, CLI_OPT_NO_POCK_CHAMBOLLE},
      {"pock_chambolle_alpha", required_argument, 0,
       CLI_OPT_POCK_CHAMBOLLE_ALPHA},
      {"no_bound_obj_rescaling", no_argument, 0,
       CLI_OPT_NO_BOUND_OBJ_RESCALING},
      {"no_scaling", no_argument, 0, CLI_OPT_NO_SCALING},
      {"psd_scale_mode", required_argument, 0, CLI_OPT_PSD_SCALE_MODE},
      {0, 0, 0, 0}};

  char opt_string[96];
  snprintf(opt_string, sizeof(opt_string), "f:n:d:m:s:t:l:S:hO:%s",
           CLI_SOLVER_OPT_STRING);

  int opt, idx = 0;
  while ((opt = getopt_long(argc, argv, opt_string, long_options, &idx)) !=
         -1) {
    if (cli_apply_solver_opt(opt, optarg, params))
      continue;
    switch (opt) {
    case 'f':
      file_path = optarg;
      break;
    case 'n':
      gen_n = atoi(optarg);
      have_n = 1;
      break;
    case 'd':
      gen_density = atof(optarg);
      have_d = 1;
      break;
    case 'm':
      if (strcmp(optarg, "chordal") == 0)
        cfg.chordal = 1;
      else if (strcmp(optarg, "dense") == 0)
        cfg.chordal = 0;
      else {
        fprintf(stderr, "Unknown mode '%s' (use 'chordal' or 'dense')\n",
                optarg);
        free(params);
        return EXIT_FAILURE;
      }
      break;
    case 's':
      gen_seed = (uint64_t)strtoull(optarg, NULL, 10);
      break;
    case 't':
      cfg.num_round_trials = atoi(optarg);
      break;
    case 'l':
      cfg.num_ls_iters = atoi(optarg);
      break;
    case 'S':
      cfg.round_seed = (uint64_t)strtoull(optarg, NULL, 10);
      break;
    case 'O':
      output_dir = optarg;
      break;
    case 'h':
      print_usage(argv[0]);
      free(params);
#ifdef USE_MPI
      if (is_mpi)
        MPI_Finalize();
#endif
      return EXIT_SUCCESS;
    default:
      print_usage(argv[0]);
      free(params);
#ifdef USE_MPI
      if (is_mpi)
        MPI_Finalize();
#endif
      return EXIT_FAILURE;
    }
  }

  if (file_path && (have_n || have_d)) {
    fprintf(stderr,
            "Error: --file (-f) is mutually exclusive with --size/--density.\n");
    free(params);
#ifdef USE_MPI
    if (is_mpi)
      MPI_Finalize();
#endif
    return EXIT_FAILURE;
  }
  if (!file_path && (!have_n || !have_d)) {
    fprintf(stderr, "Error: provide --file (-f) OR both --size (-n) and "
                    "--density (-d).\n\n");
    print_usage(argv[0]);
    free(params);
#ifdef USE_MPI
    if (is_mpi)
      MPI_Finalize();
#endif
    return EXIT_FAILURE;
  }

  g_log_verbose = params->verbose;
  char label_buf[256];
  if (file_path)
    snprintf(label_buf, sizeof(label_buf), "QUBO File  %s  mode=%s", file_path,
             cfg.chordal ? "chordal" : "dense");
  else
    snprintf(label_buf, sizeof(label_buf),
             "Gen QUBO  n=%d  density=%.2e  mode=%s", gen_n, gen_density,
             cfg.chordal ? "chordal" : "dense");
  params->instance_label = label_buf;

  char *instance_name = NULL;
  if (output_dir != NULL) {
    if (file_path)
      instance_name = cli_extract_instance_name(file_path);
    else
      instance_name =
          qubo_default_instance_name(gen_n, gen_density, cfg.chordal);
  }
  cfg.output_dir = output_dir;
  cfg.instance_name = instance_name;

  int rank = 0;
#ifdef USE_MPI
  if (is_mpi) {
    int mi = 0;
    MPI_Initialized(&mi);
    if (mi)
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
#endif

  qubo_problem_t *q = NULL;
  if (rank == 0) {
    if (file_path) {
      LOG_DBG("\n[Reading] QUBO from %s (mode=%s)\n", file_path,
              cfg.chordal ? "chordal" : "dense");
      q = qubo_read_file(file_path, QUBO_FMT_AUTO);
    } else {
      LOG_DBG("\n[Generating] Random QUBO (n=%d, density=%.2e, mode=%s)\n",
              gen_n, gen_density, cfg.chordal ? "chordal" : "dense");
      q = generate_random_qubo(gen_n, gen_density, gen_seed);
    }
    if (!q) {
      free(instance_name);
      free(params);
#ifdef USE_MPI
      if (is_mpi)
        MPI_Finalize();
#endif
      return EXIT_FAILURE;
    }
  }

  int rc = qubo_run_e2e(q, &cfg, params);

  free(instance_name);
  free(params);

#ifdef USE_MPI
  if (is_mpi)
    MPI_Finalize();
#endif
  return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
