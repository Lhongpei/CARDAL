/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once

#include "internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

rescale_info_t *rescale_problem(const cardal_parameters_t *params,
                                const compressed_sdp_problem_t *original_problem);

void free_rescale_info(rescale_info_t *info);

#ifdef __cplusplus
}
#endif
