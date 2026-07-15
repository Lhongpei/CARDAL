/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "generator.h"
#include "utils.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint64_t snl_rng_state = 0xDEADBEEFCAFEBABEULL;
static inline double snl_rand_double() {
  snl_rng_state ^= snl_rng_state << 13;
  snl_rng_state ^= snl_rng_state >> 7;
  snl_rng_state ^= snl_rng_state << 17;
  return ((snl_rng_state >> 11) + 1.0) / 9007199254740992.0;
}
static inline double snl_rand_normal(double std_dev) {
  double u1 = snl_rand_double();
  double u2 = snl_rand_double();
  if (u1 <= 1e-10)
    u1 = 1e-10;
  double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
  return z * std_dev;
}

// Sensor Network Localization (SNL) SDP generator.
//   n            : number of sensors (unknown nodes)
//   num_anchors  : number of anchors (known coordinates; >=4 recommended)
//   d            : spatial dimension (typically 2 or 3)
//   radio_range  : sensor detection radius, in [0, 1]
//   noise_std    : standard deviation of the distance measurement noise
//                  (0.0 for a noise-free complete graph)
basic_sdp_t *generate_snl_sdp(int n, int num_anchors, int d, double radio_range,
                              double noise_std) {
  LOG_DBG("\nGenerating SNL SDP for n=%d, anchors=%d, d=%d, R=%.3f, "
         "noise=%.1e...\n",
         n, num_anchors, d, radio_range, noise_std);

  snl_rng_state = (uint64_t)time(NULL) ^ 0x123456789ABCDEFULL;

  double *sensors = (double *)safe_malloc(n * d * sizeof(double));
  double *anchors = (double *)safe_malloc(num_anchors * d * sizeof(double));

  for (int i = 0; i < n * d; i++)
    sensors[i] = snl_rand_double();
  for (int i = 0; i < num_anchors * d; i++)
    anchors[i] = snl_rand_double();

  int m_ss = 0;
  int m_sa = 0;
  double r_sq = radio_range * radio_range;

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dist2 = 0;
      for (int l = 0; l < d; l++) {
        double diff = sensors[i * d + l] - sensors[j * d + l];
        dist2 += diff * diff;
      }
      if (dist2 <= r_sq)
        m_ss++;
    }

    for (int k = 0; k < num_anchors; k++) {
      double dist2 = 0;
      for (int l = 0; l < d; l++) {
        double diff = sensors[i * d + l] - anchors[k * d + l];
        dist2 += diff * diff;
      }
      if (dist2 <= r_sq)
        m_sa++;
    }
  }

  int mat_dim = d + n;
  int m_identity = d * (d + 1) / 2;
  int total_m = m_identity + m_ss + m_sa;

  // Nonzero counts (upper-triangle storage, mirror nodes excluded).
  int nnz_identity = d * (d + 1) / 2;
  int nnz_ss = 3 * m_ss;       // 2 diagonal + 1 upper-triangle
  int nnz_sa = (1 + d) * m_sa; // 1 diagonal + d upper-triangle
  int total_nnz = nnz_identity + nnz_ss + nnz_sa;

  LOG_DBG("  -> Graph parsed: %d SS edges, %d SA edges.\n", m_ss, m_sa);
  LOG_DBG("  -> Total constraints (m): %d\n", total_m);
  LOG_DBG("  -> Allocating %d non-zeros...\n", total_nnz);

  basic_sdp_t *input = (basic_sdp_t *)safe_malloc(sizeof(basic_sdp_t));
  input->m = total_m;
  input->n_cones = 1;
  input->blk_dims = (int *)safe_malloc(sizeof(int));
  input->blk_dims[0] = mat_dim;

  input->right_hand_side = (double *)safe_malloc(total_m * sizeof(double));

  input->nnz_psd_constr = total_nnz;
  input->psd_cone_constraints =
      (psd_cone_constraint_t *)safe_malloc(sizeof(psd_cone_constraint_t));
  input->psd_cone_constraints->constr_ind =
      (int *)safe_malloc(total_nnz * sizeof(int));
  input->psd_cone_constraints->cone_ind = (int *)calloc(total_nnz, sizeof(int));
  input->psd_cone_constraints->row_ind =
      (int *)safe_malloc(total_nnz * sizeof(int));
  input->psd_cone_constraints->col_ind =
      (int *)safe_malloc(total_nnz * sizeof(int));
  input->psd_cone_constraints->val =
      (double *)safe_malloc(total_nnz * sizeof(double));

  int c_idx = 0;
  int nnz_idx = 0;

  // Top-left identity-matrix constraints (upper triangle only).
  for (int i = 0; i < d; i++) {
    for (int j = i; j < d; j++) {
      if (i == j) {
        input->right_hand_side[c_idx] = 1.0;
        input->psd_cone_constraints->constr_ind[nnz_idx] = c_idx;
        input->psd_cone_constraints->row_ind[nnz_idx] = i;
        input->psd_cone_constraints->col_ind[nnz_idx] = i;
        input->psd_cone_constraints->val[nnz_idx] = 1.0;
        nnz_idx++;
      } else {
        input->right_hand_side[c_idx] = 0.0;
        input->psd_cone_constraints->constr_ind[nnz_idx] = c_idx;
        input->psd_cone_constraints->row_ind[nnz_idx] = i;
        input->psd_cone_constraints->col_ind[nnz_idx] = j;
        input->psd_cone_constraints->val[nnz_idx] = 0.5;
        nnz_idx++;
      }
      c_idx++;
    }
  }

  // Sensor-Sensor distance constraints (unique pairs only).
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dist2 = 0;
      for (int l = 0; l < d; l++) {
        double diff = sensors[i * d + l] - sensors[j * d + l];
        dist2 += diff * diff;
      }
      if (dist2 <= r_sq) {
        double measured_dist =
            sqrt(dist2) * fabs(1.0 + snl_rand_normal(noise_std));
        input->right_hand_side[c_idx] = measured_dist * measured_dist;

        int idx_i = d + i;
        int idx_j = d + j;

        input->psd_cone_constraints->constr_ind[nnz_idx] = c_idx;
        input->psd_cone_constraints->row_ind[nnz_idx] = idx_i;
        input->psd_cone_constraints->col_ind[nnz_idx] = idx_i;
        input->psd_cone_constraints->val[nnz_idx] = 1.0;
        nnz_idx++;

        input->psd_cone_constraints->constr_ind[nnz_idx] = c_idx;
        input->psd_cone_constraints->row_ind[nnz_idx] = idx_j;
        input->psd_cone_constraints->col_ind[nnz_idx] = idx_j;
        input->psd_cone_constraints->val[nnz_idx] = 1.0;
        nnz_idx++;

        // 注意：因为 j 循环从 i+1 开始，所以 idx_i 永远小于
        // idx_j，直接填入即可！
        input->psd_cone_constraints->constr_ind[nnz_idx] = c_idx;
        input->psd_cone_constraints->row_ind[nnz_idx] = idx_i;
        input->psd_cone_constraints->col_ind[nnz_idx] = idx_j;
        input->psd_cone_constraints->val[nnz_idx] = -1.0;
        nnz_idx++;

        c_idx++;
      }
    }
  }

  // Sensor-Anchor distance constraints (unique pairs only).
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < num_anchors; k++) {
      double dist2 = 0;
      double norm_a2 = 0;
      for (int l = 0; l < d; l++) {
        double diff = sensors[i * d + l] - anchors[k * d + l];
        dist2 += diff * diff;
        norm_a2 += anchors[k * d + l] * anchors[k * d + l];
      }
      if (dist2 <= r_sq) {
        double measured_dist =
            sqrt(dist2) * fabs(1.0 + snl_rand_normal(noise_std));
        input->right_hand_side[c_idx] = measured_dist * measured_dist - norm_a2;

        int idx_i = d + i;

        input->psd_cone_constraints->constr_ind[nnz_idx] = c_idx;
        input->psd_cone_constraints->row_ind[nnz_idx] = idx_i;
        input->psd_cone_constraints->col_ind[nnz_idx] = idx_i;
        input->psd_cone_constraints->val[nnz_idx] = 1.0;
        nnz_idx++;

        for (int l = 0; l < d; l++) {
          double a_val = -anchors[k * d + l];
          // 注意：l 最大是 d-1，而 idx_i = d + i。因此 l 永远严格小于 idx_i！
          input->psd_cone_constraints->constr_ind[nnz_idx] = c_idx;
          input->psd_cone_constraints->row_ind[nnz_idx] = l;
          input->psd_cone_constraints->col_ind[nnz_idx] = idx_i;
          input->psd_cone_constraints->val[nnz_idx] = a_val;
          nnz_idx++;
        }
        c_idx++;
      }
    }
  }

  int c_nnz = mat_dim;
  input->nnz_psd_obj = c_nnz;
  input->psd_cone_objective =
      (psd_cone_objective_t *)safe_malloc(sizeof(psd_cone_objective_t));
  input->psd_cone_objective->cone_ind = (int *)calloc(c_nnz, sizeof(int));
  input->psd_cone_objective->row_ind = (int *)safe_malloc(c_nnz * sizeof(int));
  input->psd_cone_objective->col_ind = (int *)safe_malloc(c_nnz * sizeof(int));
  input->psd_cone_objective->val =
      (double *)safe_malloc(c_nnz * sizeof(double));

  for (int i = 0; i < mat_dim; i++) {
    input->psd_cone_objective->row_ind[i] = i;
    input->psd_cone_objective->col_ind[i] = i;
    input->psd_cone_objective->val[i] = 1e-5;
  }

  input->lp_constraints = NULL;
  input->lp_objective = NULL;
  input->nnz_lp_constr = 0;
  input->nnz_lp_obj = 0;

  free(sensors);
  free(anchors);

  LOG_DBG("  -> SNL Generation Complete!\n\n");
  return input;
}