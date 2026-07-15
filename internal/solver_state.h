/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once
#include "internal_types.h"
#include "sdp_types.h"
#ifdef __cplusplus
extern "C" {
#endif

cardal_sdp_solver_state_t *
initialize_solver_state(const compressed_sdp_problem_t *sdp_problem,
                        const cardal_parameters_t *params,
                        int *theoretical_max_rank_per_blk,
                        const rescale_info_t *rescale_info);

void free_solver_state(cardal_sdp_solver_state_t *state);

void augment_system_rank(cardal_sdp_solver_state_t *state,
                         const int *rank_incs,
                         double *const *neg_eigvecs,
                         double *const *neg_eigvals);

void build_cone_batches(cardal_sdp_solver_state_t *state);
void refresh_cone_batches(cardal_sdp_solver_state_t *state);

sdp_result_t *
create_result_from_state(cardal_sdp_solver_state_t *state,
                      const compressed_sdp_problem_t *sdp_problem);
#ifdef __cplusplus
}
#endif
