/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once

#include "internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int *rank_incs;
  double **d_columns;
} rank_lift_correction_t;

void initialize_rank_lift_correction(const cardal_sdp_solver_state_t *state,
                                     rank_lift_correction_t *correction);

void free_rank_lift_correction(const cardal_sdp_solver_state_t *state,
                               rank_lift_correction_t *correction);

// Construct augmentation columns from retained dual-slack directions using
// the configured random, joint QP, shared closed-form, or joint SDP backend.
// Returned device columns are already scaled and numerically compressed.
int solve_joint_rank_lift(cardal_sdp_solver_state_t *state,
                          const int *direction_counts,
                          double *const *d_directions,
                          double *const *eigenvalues,
                          rank_lift_correction_t *correction);

#ifdef __cplusplus
}
#endif
