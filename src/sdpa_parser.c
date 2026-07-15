/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "sdp_types.h"
#include "utils.h"
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

typedef struct {
  char *ptr;
  char *end;
} fast_scanner_t;

static inline void skip_junk(fast_scanner_t *s) {
  while (s->ptr < s->end) {
    char c = *s->ptr;
    if (isspace(c) || c == ',' || c == '{' || c == '}' || c == '(' ||
        c == ')' || c == '=' || c == ':') {
      s->ptr++;
      continue;
    }
    if (c == '"' || c == '*' || c == '#') {
      while (s->ptr < s->end && *s->ptr != '\n')
        s->ptr++;
      continue;
    }
    break;
  }
}

static inline int next_int(fast_scanner_t *s) {
  skip_junk(s);
  if (s->ptr >= s->end)
    return 0;
  int val = 0, neg = 0;
  if (*s->ptr == '-') {
    neg = 1;
    s->ptr++;
  } else if (*s->ptr == '+') {
    s->ptr++;
  }

  if (s->ptr < s->end && !isdigit(*s->ptr)) {
    s->ptr++;
    return 0;
  }

  while (s->ptr < s->end && isdigit(*s->ptr)) {
    val = val * 10 + (*s->ptr - '0');
    s->ptr++;
  }
  return neg ? -val : val;
}

static inline double next_double(fast_scanner_t *s) {
  skip_junk(s);
  if (s->ptr >= s->end)
    return 0.0;

  char buf[64];
  int len = 0;
  while (s->ptr < s->end && len < 63) {
    char c = *s->ptr;
    if (isspace(c) || c == ',' || c == '}' || c == '{' || c == '(' || c == ')')
      break;
    buf[len++] = c;
    s->ptr++;
  }
  buf[len] = '\0';

  char *next_ptr;
  return strtod(buf, &next_ptr);
}

basic_sdp_t *parse_sdpa_from_memory(char *data, size_t size) {
  fast_scanner_t s = {data, data + size};
  basic_sdp_t *sdp = (basic_sdp_t *)calloc(1, sizeof(basic_sdp_t));

  sdp->m = next_int(&s);
  int raw_n_blocks = next_int(&s);

  int actual_n_cones = 0;
  int lp_dim = 0;

  int *temp_blk_dims = (int *)safe_malloc(raw_n_blocks * sizeof(int));
  int *block_is_lp = (int *)safe_malloc(raw_n_blocks * sizeof(int));
  int *block_lp_offset = (int *)safe_malloc(raw_n_blocks * sizeof(int));
  int *psd_cone_mapping = (int *)safe_malloc(raw_n_blocks * sizeof(int));

  for (int i = 0; i < raw_n_blocks; i++) {
    int dim = next_int(&s);
    if (dim > 0) {
      temp_blk_dims[actual_n_cones] = dim;
      psd_cone_mapping[i] = actual_n_cones;
      block_is_lp[i] = 0;
      actual_n_cones++;
    } else if (dim < 0) {
      block_is_lp[i] = 1;
      block_lp_offset[i] = lp_dim;
      psd_cone_mapping[i] = -1;
      lp_dim += (-dim);
    } else {
      block_is_lp[i] = 0;
      psd_cone_mapping[i] = -1;
    }
  }

  sdp->n_cones = actual_n_cones;
  sdp->blk_dims = temp_blk_dims;
  sdp->lp_dim = lp_dim;

  sdp->right_hand_side = (double *)safe_malloc(sdp->m * sizeof(double));
  for (int i = 0; i < sdp->m; i++) {
    sdp->right_hand_side[i] = next_double(&s);
  }

  char *data_start = s.ptr;

  // Two-pass count: classify each entry by destination (PSD-constr, PSD-obj,
  // LP-constr, LP-obj) so we can allocate the four families at their exact
  // size instead of every one at total_lines. For huge SOS-SDPs (N>=100) the
  // file has ~2B PSD-constr entries and tiny everything else; the prior code
  // wasted ~12x that as obj+LP arrays and could exhaust contiguous host VA.
  long long psd_c_n = 0, psd_o_n = 0, lp_c_n = 0;
  while (s.ptr < s.end) {
    skip_junk(&s);
    if (s.ptr >= s.end) break;
    int mat_idx = next_int(&s);
    int cone_idx = next_int(&s) - 1;
    int row = next_int(&s) - 1;
    int col = next_int(&s) - 1;
    (void)next_double(&s);
    if (cone_idx < 0 || cone_idx >= raw_n_blocks) continue;
    if (block_is_lp[cone_idx]) {
      if (row != col) continue;
      if (mat_idx != 0) lp_c_n++;
      // lp_objective is dense (sized lp_dim), no counter needed.
    } else {
      if (psd_cone_mapping[cone_idx] == -1) continue;
      if (mat_idx == 0) psd_o_n++;
      else psd_c_n++;
    }
  }
  if (LOG_V(3)) {
    fprintf(stderr, "[Parser count] psd_constr=%lld psd_obj=%lld lp_constr=%lld\n",
            psd_c_n, psd_o_n, lp_c_n);
    fflush(stderr);
  }

  // Per-array exact sizes. Use `int` for the field type to match downstream;
  // sanity-check that counts fit. SDP parser would need a wider field for
  // problems with > INT_MAX entries of any one kind.
  if (psd_c_n > (long long)INT_MAX || psd_o_n > (long long)INT_MAX ||
      lp_c_n > (long long)INT_MAX) {
    fprintf(stderr, "Fatal: entry count exceeds INT_MAX "
            "(psd_constr=%lld, psd_obj=%lld, lp_constr=%lld). "
            "Widen nnz fields to long long.\n",
            psd_c_n, psd_o_n, lp_c_n);
    exit(EXIT_FAILURE);
  }

  sdp->psd_cone_constraints =
      (psd_cone_constraint_t *)calloc(1, sizeof(psd_cone_constraint_t));
  sdp->psd_cone_objective =
      (psd_cone_objective_t *)calloc(1, sizeof(psd_cone_objective_t));
  sdp->lp_constraints = (lp_constraint_t *)calloc(1, sizeof(lp_constraint_t));

  // Use max(count,1) to avoid 0-size malloc edge cases on platforms that
  // return NULL for size=0.
  size_t pc_n = (size_t)(psd_c_n > 0 ? psd_c_n : 1);
  size_t po_n = (size_t)(psd_o_n > 0 ? psd_o_n : 1);
  size_t lc_n = (size_t)(lp_c_n > 0 ? lp_c_n : 1);

  if (LOG_V(3)) {
    fprintf(stderr, "[Parser alloc] pc_n=%zu (%.2f GB int, %.2f GB double per array)\n",
            pc_n, pc_n * 4.0 / 1e9, pc_n * 8.0 / 1e9);
    fflush(stderr);
  }
  sdp->psd_cone_constraints->constr_ind = (int *)safe_malloc(pc_n * sizeof(int));
  sdp->psd_cone_constraints->cone_ind   = (int *)safe_malloc(pc_n * sizeof(int));
  sdp->psd_cone_constraints->row_ind    = (int *)safe_malloc(pc_n * sizeof(int));
  sdp->psd_cone_constraints->col_ind    = (int *)safe_malloc(pc_n * sizeof(int));
  sdp->psd_cone_constraints->val        = (double *)safe_malloc(pc_n * sizeof(double));
  if (LOG_V(3)) {
    fprintf(stderr, "[Parser alloc] done.\n");
    fflush(stderr);
  }

  sdp->psd_cone_objective->cone_ind = (int *)safe_malloc(po_n * sizeof(int));
  sdp->psd_cone_objective->row_ind  = (int *)safe_malloc(po_n * sizeof(int));
  sdp->psd_cone_objective->col_ind  = (int *)safe_malloc(po_n * sizeof(int));
  sdp->psd_cone_objective->val      = (double *)safe_malloc(po_n * sizeof(double));

  sdp->lp_constraints->row_ind = (int *)safe_malloc(lc_n * sizeof(int));
  sdp->lp_constraints->col_ind = (int *)safe_malloc(lc_n * sizeof(int));
  sdp->lp_constraints->val     = (double *)safe_malloc(lc_n * sizeof(double));

  sdp->lp_objective =
      (double *)calloc((lp_dim > 0 ? lp_dim : 1), sizeof(double));

  s.ptr = data_start;
  int c_count = 0, o_count = 0;
  int lp_c_count = 0, lp_o_count = 0;

  while (s.ptr < s.end) {
    skip_junk(&s);
    if (s.ptr >= s.end)
      break;

    int mat_idx = next_int(&s);
    int cone_idx = next_int(&s) - 1; // 0-based
    int row = next_int(&s) - 1;
    int col = next_int(&s) - 1;
    double val = next_double(&s);

    if (cone_idx < 0 || cone_idx >= raw_n_blocks)
      continue;

    if (block_is_lp[cone_idx]) {
      if (row != col)
        continue;

      int lp_var_idx = block_lp_offset[cone_idx] + row;

      if (mat_idx == 0) {
        sdp->lp_objective[lp_var_idx] = -val;
        lp_o_count++;
      } else {
        sdp->lp_constraints->row_ind[lp_c_count] = mat_idx - 1;
        sdp->lp_constraints->col_ind[lp_c_count] = lp_var_idx;
        sdp->lp_constraints->val[lp_c_count] = val;
        lp_c_count++;
      }
    } else {
      int mapped_cone_idx = psd_cone_mapping[cone_idx];
      if (mapped_cone_idx == -1)
        continue;

      if (mat_idx == 0) {
        sdp->psd_cone_objective->cone_ind[o_count] = mapped_cone_idx;
        sdp->psd_cone_objective->row_ind[o_count] = row;
        sdp->psd_cone_objective->col_ind[o_count] = col;
        sdp->psd_cone_objective->val[o_count] = -val;
        o_count++;
      } else {
        sdp->psd_cone_constraints->constr_ind[c_count] = mat_idx - 1;
        sdp->psd_cone_constraints->cone_ind[c_count] = mapped_cone_idx;
        sdp->psd_cone_constraints->row_ind[c_count] = row;
        sdp->psd_cone_constraints->col_ind[c_count] = col;
        sdp->psd_cone_constraints->val[c_count] = val;
        c_count++;
      }
    }
  }

  sdp->nnz_psd_constr = c_count;
  sdp->nnz_psd_obj = o_count;
  sdp->nnz_lp_constr = lp_c_count;
  sdp->nnz_lp_obj = lp_o_count;

  LOG_DBG("  -> Parsed successfully: m=%d, PSD cones=%d, LP dim=%d\n", sdp->m,
         sdp->n_cones, sdp->lp_dim);
  LOG_DBG("     PSD NNZ: A=%d, C=%d | LP NNZ: A=%d, C=%d\n", sdp->nnz_psd_constr,
         sdp->nnz_psd_obj, sdp->nnz_lp_constr, sdp->nnz_lp_obj);
  fflush(stdout);

  free(block_is_lp);
  free(block_lp_offset);
  free(psd_cone_mapping);

  return sdp;
}