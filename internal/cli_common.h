/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once

#include "sdp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int cli_is_mpi(void);

int cli_apply_solver_opt(int opt, const char *optarg,
                         cardal_parameters_t *params);

void cli_print_solver_param_help(void);

extern const char *CLI_SOLVER_OPT_STRING;

enum {
  CLI_OPT_L_INF_RUIZ_ITER = 1000,
  CLI_OPT_NO_POCK_CHAMBOLLE,
  CLI_OPT_POCK_CHAMBOLLE_ALPHA,
  CLI_OPT_NO_BOUND_OBJ_RESCALING,
  CLI_OPT_NO_SCALING,
  CLI_OPT_PSD_SCALE_MODE,
  CLI_OPT_SHUFFLE_MODE,
  CLI_OPT_EPS_PRIMAL,
  CLI_OPT_EPS_DUAL,
  CLI_OPT_EPS_GAP,
  CLI_OPT_AUGMENTATION_MODE,
};

char *cli_extract_instance_name(const char *filename);

#ifdef __cplusplus
}
#endif
