/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "cli_common.h"
#include <libgen.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *CLI_SOLVER_OPT_STRING = "r:R:e:i:o:p:c:M:L:T:v:z:";

int cli_is_mpi(void) {
  if (getenv("OMPI_COMM_WORLD_RANK") != NULL)
    return 1;
  if (getenv("PMI_RANK") != NULL)
    return 1;
  if (getenv("PMI_SIZE") != NULL)
    return 1;
  if (getenv("I_MPI_RANK") != NULL)
    return 1;
  return 0;
}

int cli_apply_solver_opt(int opt, const char *optarg,
                         cardal_parameters_t *params) {
  switch (opt) {
  case 'r':
    params->initial_rank = atoi(optarg);
    return 1;
  case 'R':
    params->max_rank = atoi(optarg);
    return 1;
  case 'e':
    params->termination_criteria.eps_feasible_relative = atof(optarg);
    params->termination_criteria.eps_optimal_relative = atof(optarg);
    return 1;
  case 'i':
    params->inner_iterations_limit = atof(optarg);
    return 1;
  case 'o':
    params->termination_criteria.iteration_limit = atoi(optarg);
    return 1;
  case 'p':
    params->penalty_factor = atof(optarg);
    return 1;
  case 'c':
    params->initial_penalty_coef = atof(optarg);
    return 1;
  case 'M':
    params->max_penalty_coef = atof(optarg);
    return 1;
  case 'T':
    params->termination_criteria.time_sec_limit = atof(optarg);
    return 1;
  case 'L':
    params->lbfgs_history_size = atoi(optarg);
    if (params->lbfgs_history_size < 1)
      params->lbfgs_history_size = 1;
    return 1;
  case 'v':
    params->verbose = atoi(optarg);
    if (params->verbose < 0)
      params->verbose = 0;
    if (params->verbose > 3)
      params->verbose = 3;
    return 1;
  case 'z': {
    char *arg_copy = strdup(optarg);
    char *token = strtok(arg_copy, ",");
    if (token) {
      params->grid_size.row_dims = atoi(token);
      token = strtok(NULL, ",");
      if (token) {
        params->grid_size.rank_dims = atoi(token);
        token = strtok(NULL, ",");
        params->grid_size.cone_dims = token ? atoi(token) : 1;
        if (params->grid_size.cone_dims < 1)
          params->grid_size.cone_dims = 1;
        params->grid_size.decided = true;
      }
    }
    free(arg_copy);
    return 1;
  }
  case CLI_OPT_L_INF_RUIZ_ITER:
    params->l_inf_ruiz_iterations = atoi(optarg);
    if (params->l_inf_ruiz_iterations < 0)
      params->l_inf_ruiz_iterations = 0;
    return 1;
  case CLI_OPT_NO_POCK_CHAMBOLLE:
    params->has_pock_chambolle_alpha = false;
    return 1;
  case CLI_OPT_POCK_CHAMBOLLE_ALPHA:
    params->pock_chambolle_alpha = atof(optarg);
    params->has_pock_chambolle_alpha = true;
    return 1;
  case CLI_OPT_NO_BOUND_OBJ_RESCALING:
    params->bound_objective_rescaling = false;
    return 1;
  case CLI_OPT_NO_SCALING:
    // Master switch: turn off all three scaling stages.
    params->l_inf_ruiz_iterations = 0;
    params->has_pock_chambolle_alpha = false;
    params->bound_objective_rescaling = false;
    return 1;
  case CLI_OPT_PSD_SCALE_MODE:
    if (optarg != NULL) {
      if (strcmp(optarg, "per-element") == 0 || strcmp(optarg, "0") == 0)
        params->psd_scale_mode = PSD_SCALE_MODE_PER_ELEMENT;
      else if (strcmp(optarg, "per-cone") == 0 || strcmp(optarg, "1") == 0)
        params->psd_scale_mode = PSD_SCALE_MODE_PER_CONE;
      else {
        fprintf(stderr, "Unknown --psd-scale-mode value: %s (use "
                "per-element or per-cone)\n", optarg);
        return 1;
      }
    }
    return 1;
  case CLI_OPT_SHUFFLE_MODE:
    if (optarg != NULL) {
      if (strcmp(optarg, "none") == 0)
        params->shuffle_mode = SHUFFLE_NONE;
      else if (strcmp(optarg, "uniform") == 0)
        params->shuffle_mode = SHUFFLE_UNIFORM;
      else if (strcmp(optarg, "block") == 0)
        params->shuffle_mode = SHUFFLE_BLOCK;
      else if (strcmp(optarg, "col") == 0 ||
               strcmp(optarg, "col-locality") == 0)
        params->shuffle_mode = SHUFFLE_COL_LOCALITY;
      else {
        fprintf(stderr, "Unknown --shuffle value: %s "
                "(use none | uniform | block | col)\n", optarg);
        return 1;
      }
    }
    return 1;
  }
  return 0;
}

void cli_print_solver_param_help(void) {
  printf("Solver Parameters:\n");
  printf("  -r, --rank <int>        Initial rank for Burer-Monteiro "
         "(Default: Auto 2*log(m))\n");
  printf("  -R, --max-rank <int>    Hard upper bound on per-block rank "
         "(Default: Auto ceil((sqrt(8m+1)-1)/2))\n");
  printf("  -e, --eps <float>       Target tolerance (Feas & Opt) "
         "(Default: 1e-4)\n");
  printf("  -i, --inner-iters <int> Inner loop iteration limit "
         "(Default: 1000)\n");
  printf("  -o, --outer-iters <int> Outer loop iteration limit "
         "(Default: 500)\n");
  printf("  -p, --penalty-fac <flt> Penalty multiplier factor "
         "(Default: 1.5)\n");
  printf("  -M, --max-penalty <flt> Max penalty coefficient "
         "(Default: 5e5)\n");
  printf("  -T, --time-limit <sec>  Wall-time budget in seconds "
         "(0 = no limit, default)\n");
  printf("  -L, --lbfgs-hist <int>  LBFGS history depth m (Default: 5)\n");
  printf("  -c, --init-penalty <flt>Initial penalty coefficient "
         "(Default: Auto 2/sqrt(N))\n");
  printf("  -v, --verbose <int>     Log level: 0=silent, 1=banner+summary, "
         "2=+sections+iter, 3=debug (default 3)\n");
  printf("  -z, --grid_size <r,n[,k]>  Grid topology for MPI: "
         "r=row (constraint) axis, n=rank (BM-col) axis, "
         "k=cone axis (default 1)\n");
  printf("  -O, --output-dir <path> Directory to write <instance>_summary.txt "
         "into (created if missing)\n");
  printf("  -h, --help              Print this help message and exit\n");
  printf("\n");
  printf("Scaling (preconditioning):\n");
  printf("      --l_inf_ruiz_iter <int>     Iterations for L-inf Ruiz "
         "rescaling (default 10; 0 disables)\n");
  printf("      --no_pock_chambolle         Disable Pock-Chambolle "
         "rescaling (default enabled)\n");
  printf("      --pock_chambolle_alpha <flt> Value for Pock-Chambolle "
         "alpha (default 1.0)\n");
  printf("      --no_bound_obj_rescaling    Disable bound-objective "
         "rescaling (default enabled)\n");
  printf("      --no_scaling                Disable ALL scaling stages "
         "(equivalent to the three flags above + ruiz=0)\n");
  printf("      --psd_scale_mode <mode>     PSD cone scaling style: "
         "per-element (D=diag(d_k), default) or per-cone (uniform s_k*I)\n");
  printf("      --shuffle <mode>            Pre-partition constraint reorder "
         "(distributed only): none | uniform | block | col (default col)\n");
}

char *cli_extract_instance_name(const char *filename) {
  if (!filename || !*filename)
    return NULL;
  char *copy = strdup(filename);
  if (!copy)
    return NULL;
  char *base = basename(copy);
  char *dot = strchr(base, '.');
  if (dot)
    *dot = '\0';
  char *name = strdup(base);
  free(copy);
  return name;
}
