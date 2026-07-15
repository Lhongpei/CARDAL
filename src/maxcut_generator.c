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
static uint64_t rng_state = 123456789012345ULL;

static inline double next_rand_double() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return ((rng_state >> 11) + 1.0) / 9007199254740992.0;
}

basic_sdp_t *generate_maxcut_sdp_huge(int n, double edge_prob) {
  LOG_DBG("Generating HUGE Max-Cut SDP for n=%d, edge_prob=%e...\n", n,
         edge_prob);

  rng_state = 42; //(uint64_t)time(NULL) ^ 0xDEADBEEFCAFEBABULL;

  basic_sdp_t *input = (basic_sdp_t *)safe_malloc(sizeof(basic_sdp_t));
  input->m = n;
  input->n_cones = 1;
  input->blk_dims = (int *)safe_malloc(sizeof(int));
  input->blk_dims[0] = n;

  input->right_hand_side = (double *)safe_malloc(n * sizeof(double));
  for (int i = 0; i < n; i++)
    input->right_hand_side[i] = 1.0;

  input->nnz_psd_constr = n;
  input->psd_cone_constraints =
      (psd_cone_constraint_t *)safe_malloc(sizeof(psd_cone_constraint_t));
  input->psd_cone_constraints->constr_ind = (int *)safe_malloc(n * sizeof(int));
  input->psd_cone_constraints->cone_ind = (int *)calloc(n, sizeof(int));
  input->psd_cone_constraints->row_ind = (int *)safe_malloc(n * sizeof(int));
  input->psd_cone_constraints->col_ind = (int *)safe_malloc(n * sizeof(int));
  input->psd_cone_constraints->val = (double *)safe_malloc(n * sizeof(double));

  for (int i = 0; i < n; i++) {
    input->psd_cone_constraints->constr_ind[i] = i;
    input->psd_cone_constraints->row_ind[i] = i;
    input->psd_cone_constraints->col_ind[i] = i;
    input->psd_cone_constraints->val[i] = 1.0;
  }

  double expected_edges = (double)n * (double)n / 2.0 * edge_prob;
  int capacity = (int)expected_edges + n + 10000;

  int *c_row = (int *)safe_malloc(capacity * sizeof(int));
  int *c_col = (int *)safe_malloc(capacity * sizeof(int));
  double *c_val = (double *)safe_malloc(capacity * sizeof(double));
  int c_nnz = 0;

  double *degrees = (double *)calloc(n, sizeof(double));

  if (edge_prob > 0.0 && edge_prob < 1.0) {
    double log_p = log(1.0 - edge_prob);
    int v = 1;
    int w = -1;

    while (v < n) {
      double r = next_rand_double();
      int skip = (int)(log(r) / log_p);
      w = w + 1 + skip;

      while (w >= v && v < n) {
        w = w - v;
        v = v + 1;
      }

      if (v < n) {
        if (c_nnz >= capacity - 1) {
          capacity = (capacity * 3) / 2;
          c_row = (int *)realloc(c_row, capacity * sizeof(int));
          c_col = (int *)realloc(c_col, capacity * sizeof(int));
          c_val = (double *)realloc(c_val, capacity * sizeof(double));
        }

        c_row[c_nnz] = (int)w;
        c_col[c_nnz] = (int)v;
        c_val[c_nnz] = 0.25;
        c_nnz++;
        degrees[w] += 1.0;
        degrees[v] += 1.0;
      }
    }
  }

  for (int i = 0; i < n; i++) {
    if (degrees[i] > 0) {
      if (c_nnz >= capacity) {
        capacity += n;
        c_row = (int *)realloc(c_row, capacity * sizeof(int));
        c_col = (int *)realloc(c_col, capacity * sizeof(int));
        c_val = (double *)realloc(c_val, capacity * sizeof(double));
      }
      c_row[c_nnz] = i;
      c_col[c_nnz] = i;
      c_val[c_nnz] = -degrees[i] / 4.0;
      c_nnz++;
    }
  }

  input->nnz_psd_obj = (int)c_nnz;
  input->psd_cone_objective =
      (psd_cone_objective_t *)safe_malloc(sizeof(psd_cone_objective_t));
  input->psd_cone_objective->cone_ind = (int *)calloc(c_nnz, sizeof(int));
  input->psd_cone_objective->row_ind = c_row;
  input->psd_cone_objective->col_ind = c_col;
  input->psd_cone_objective->val = c_val;

  input->lp_constraints = NULL;
  input->lp_objective = NULL;
  input->nnz_lp_constr = 0;
  input->nnz_lp_obj = 0;

  free(degrees);
  return input;
}