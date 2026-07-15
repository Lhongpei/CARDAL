/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once

#include "sdp_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
basic_sdp_t *parse_sdpa_from_memory(char *data, size_t size);
basic_sdp_t *handle_mat_from_memory(const char *original_filename, char *data,
                                    size_t size);
basic_sdp_t *read_pdsdp_npz(const char *filename);
typedef enum {
  FORMAT_UNKNOWN = 0,
  FORMAT_SDPA,
  FORMAT_MAT,
  FORMAT_CBF,
} sdp_format_t;

basic_sdp_t *sdp_problem_parse(const char *filename);

void free_basic_sdp(basic_sdp_t *sdp);

#ifdef __cplusplus
}
#endif