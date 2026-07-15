/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once

#include "sdp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

basic_sdp_t *generate_maxcut_sdp_huge(int n, double edge_prob);
basic_sdp_t *generate_snl_sdp(int n, int num_anchors, int d, double radio_range,
                              double noise_std);
basic_sdp_t *generate_order_sdp(int n_size, int k);
#ifdef __cplusplus
}
#endif
