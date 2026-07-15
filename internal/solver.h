/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once

#include "sdp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

sdp_result_t *optimize(const compressed_sdp_problem_t *sdp_problem,
                       const cardal_parameters_t *params);

#ifdef __cplusplus
}
#endif
