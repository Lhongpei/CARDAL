/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "npz_reader.h"
#include "parser.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// PDSDP-style "spot npz" SDP interchange format.
//
// Expected NPZ entries (all 0-indexed in the file):
//   tI_size: int (n_cones,)                — PSD block dimensions
//   b      : float64 (m,)                   — RHS vector
//   A      : float64 (nnz_A, 5)             — [cons_idx, block, col, row, val]
//   C      : float64 (nnz_C, 4)  (optional) — [block, col, row, val]
//   c      : float64 (lp_dim,)   (optional) — linear objective coefficients
//   a      : float64 (nnz_a, 3)  (optional) — [cons_idx, var_idx, val] linear
//
// Each entry encodes a single symmetric-matrix value v at (row, col) for the
// given block (off-diag stored once; the file does NOT double or duplicate).
// We canonicalize to row<=col for basic_sdp_t's upper-triangular storage; no
// scaling needed since basic_sdp_t's trace convention already accounts for
// the symmetric off-diag factor.
// =============================================================================

static long long round_to_int(double v) { return (long long)(v + 0.5); }

basic_sdp_t *read_pdsdp_npz(const char *filename) {
  npz_archive_t *arc = npz_read(filename);
  if (!arc) {
    fprintf(stderr, "pdsdp: failed to read NPZ archive '%s'\n", filename);
    return NULL;
  }

  const npz_entry_t *e_tI = npz_find(arc, "tI_size");
  const npz_entry_t *e_b = npz_find(arc, "b");
  const npz_entry_t *e_A = npz_find(arc, "A");
  const npz_entry_t *e_C = npz_find(arc, "C");
  const npz_entry_t *e_c = npz_find(arc, "c");
  const npz_entry_t *e_a = npz_find(arc, "a");

  if (!e_tI || !e_b || !e_A) {
    fprintf(stderr,
            "pdsdp: required entries missing (tI_size=%p, b=%p, A=%p)\n",
            (const void *)e_tI, (const void *)e_b, (const void *)e_A);
    npz_free(arc);
    return NULL;
  }

  // ---- cone dims ----
  int n_cones = (int)e_tI->n_elements;
  if (e_tI->dtype != NPY_DTYPE_I64 && e_tI->dtype != NPY_DTYPE_I32) {
    fprintf(stderr, "pdsdp: tI_size must be integer dtype\n");
    npz_free(arc);
    return NULL;
  }
  int *blk_dims = (int *)safe_malloc((size_t)n_cones * sizeof(int));
  if (e_tI->dtype == NPY_DTYPE_I64) {
    const int64_t *src = (const int64_t *)e_tI->data;
    for (int k = 0; k < n_cones; k++)
      blk_dims[k] = (int)src[k];
  } else {
    const int32_t *src = (const int32_t *)e_tI->data;
    for (int k = 0; k < n_cones; k++)
      blk_dims[k] = src[k];
  }

  // ---- RHS ----
  int m = (int)e_b->n_elements;
  if (e_b->dtype != NPY_DTYPE_F64) {
    fprintf(stderr, "pdsdp: b must be float64\n");
    free(blk_dims);
    npz_free(arc);
    return NULL;
  }
  double *rhs = (double *)safe_malloc((size_t)m * sizeof(double));
  memcpy(rhs, e_b->data, (size_t)m * sizeof(double));

  // ---- PSD constraint nnz (A) ----
  if (e_A->dtype != NPY_DTYPE_F64 || e_A->n_dim != 2 || e_A->shape[1] != 5) {
    fprintf(stderr,
            "pdsdp: A must be float64 with shape (nnz, 5); got dim=%d shape=(%lld,%lld)\n",
            e_A->n_dim, e_A->shape[0], e_A->shape[1]);
    free(blk_dims);
    free(rhs);
    npz_free(arc);
    return NULL;
  }
  long long nnz_A = e_A->shape[0];
  int *A_constr = (int *)safe_malloc((size_t)nnz_A * sizeof(int));
  int *A_cone = (int *)safe_malloc((size_t)nnz_A * sizeof(int));
  int *A_row = (int *)safe_malloc((size_t)nnz_A * sizeof(int));
  int *A_col = (int *)safe_malloc((size_t)nnz_A * sizeof(int));
  double *A_val = (double *)safe_malloc((size_t)nnz_A * sizeof(double));
  const double *Asrc = (const double *)e_A->data;
  // NPY shape is (nnz, 5); numpy default order is C (row-major), so element
  // (i, k) is at index i*5 + k. fortran_order would put it at k*nnz + i.
  int A_fo = e_A->fortran_order;
  for (long long i = 0; i < nnz_A; i++) {
    double v_cons, v_cone, v_col, v_row, v_val;
    if (A_fo) {
      v_cons = Asrc[0 * nnz_A + i];
      v_cone = Asrc[1 * nnz_A + i];
      v_col = Asrc[2 * nnz_A + i];
      v_row = Asrc[3 * nnz_A + i];
      v_val = Asrc[4 * nnz_A + i];
    } else {
      v_cons = Asrc[i * 5 + 0];
      v_cone = Asrc[i * 5 + 1];
      v_col = Asrc[i * 5 + 2];
      v_row = Asrc[i * 5 + 3];
      v_val = Asrc[i * 5 + 4];
    }
    int r = (int)round_to_int(v_row);
    int c = (int)round_to_int(v_col);
    if (r > c) {
      int t = r;
      r = c;
      c = t;
    }
    A_constr[i] = (int)round_to_int(v_cons);
    A_cone[i] = (int)round_to_int(v_cone);
    A_row[i] = r;
    A_col[i] = c;
    A_val[i] = v_val;
  }

  // ---- PSD objective nnz (C) ----
  long long nnz_C = 0;
  int *C_cone = NULL, *C_row = NULL, *C_col = NULL;
  double *C_val = NULL;
  double *lp_obj = NULL;
  int *a_row = NULL, *a_col = NULL;
  double *a_val = NULL;
  if (e_C) {
    if (e_C->dtype != NPY_DTYPE_F64 || e_C->n_dim != 2 || e_C->shape[1] != 4) {
      fprintf(stderr, "pdsdp: C must be float64 with shape (nnz, 4)\n");
      goto fail;
    }
    nnz_C = e_C->shape[0];
    C_cone = (int *)safe_malloc((size_t)nnz_C * sizeof(int));
    C_row = (int *)safe_malloc((size_t)nnz_C * sizeof(int));
    C_col = (int *)safe_malloc((size_t)nnz_C * sizeof(int));
    C_val = (double *)safe_malloc((size_t)nnz_C * sizeof(double));
    const double *Csrc = (const double *)e_C->data;
    int C_fo = e_C->fortran_order;
    for (long long i = 0; i < nnz_C; i++) {
      double v_cone, v_col, v_row, v_val;
      if (C_fo) {
        v_cone = Csrc[0 * nnz_C + i];
        v_col = Csrc[1 * nnz_C + i];
        v_row = Csrc[2 * nnz_C + i];
        v_val = Csrc[3 * nnz_C + i];
      } else {
        v_cone = Csrc[i * 4 + 0];
        v_col = Csrc[i * 4 + 1];
        v_row = Csrc[i * 4 + 2];
        v_val = Csrc[i * 4 + 3];
      }
      int r = (int)round_to_int(v_row);
      int c = (int)round_to_int(v_col);
      if (r > c) {
        int t = r;
        r = c;
        c = t;
      }
      C_cone[i] = (int)round_to_int(v_cone);
      C_row[i] = r;
      C_col[i] = c;
      C_val[i] = v_val;
    }
  }

  // ---- LP objective (c) ----
  // PDSDP convention (load_problem_from_spot_npz_to_cpu):
  //   - if both 'C' (PSD obj) and 'c' (LP obj) are present: keep c as-is
  //   - if only 'c' (no 'C'): negate c. This is the SOS-dual / POP convention
  //     where the LP variable is being maximized; PDSDP's internal solver
  //     minimizes, so it stores -c. We mirror that here so the reported
  //     primal/dual objectives match PDSDP's.
  // c is sometimes stored as uint8 (0/1 indicator under SOS mode); cast to
  // double regardless of source dtype.
  int lp_dim = 0;
  int negate_lp_obj = (e_c && !e_C);
  if (e_c && e_c->dtype != NPY_DTYPE_UNKNOWN) {
    lp_dim = (int)e_c->n_elements;
    lp_obj = (double *)safe_malloc((size_t)lp_dim * sizeof(double));
    switch (e_c->dtype) {
    case NPY_DTYPE_F64:
      memcpy(lp_obj, e_c->data, (size_t)lp_dim * sizeof(double));
      break;
    case NPY_DTYPE_F32: {
      const float *src = (const float *)e_c->data;
      for (int k = 0; k < lp_dim; k++)
        lp_obj[k] = (double)src[k];
      break;
    }
    case NPY_DTYPE_I64: {
      const int64_t *src = (const int64_t *)e_c->data;
      for (int k = 0; k < lp_dim; k++)
        lp_obj[k] = (double)src[k];
      break;
    }
    case NPY_DTYPE_I32: {
      const int32_t *src = (const int32_t *)e_c->data;
      for (int k = 0; k < lp_dim; k++)
        lp_obj[k] = (double)src[k];
      break;
    }
    case NPY_DTYPE_U8: {
      const uint8_t *src = (const uint8_t *)e_c->data;
      for (int k = 0; k < lp_dim; k++)
        lp_obj[k] = (double)src[k];
      break;
    }
    case NPY_DTYPE_I8: {
      const int8_t *src = (const int8_t *)e_c->data;
      for (int k = 0; k < lp_dim; k++)
        lp_obj[k] = (double)src[k];
      break;
    }
    default:
      fprintf(stderr, "pdsdp: unsupported dtype for 'c'\n");
      free(lp_obj);
      lp_obj = NULL;
      lp_dim = 0;
      goto fail;
    }
    if (negate_lp_obj) {
      for (int k = 0; k < lp_dim; k++)
        lp_obj[k] = -lp_obj[k];
    }
  }

  // ---- LP constraint nnz (a) ----
  long long nnz_a = 0;
  if (e_a) {
    if (e_a->dtype != NPY_DTYPE_F64 || e_a->n_dim != 2 || e_a->shape[1] != 3) {
      fprintf(stderr, "pdsdp: a must be float64 with shape (nnz, 3)\n");
      goto fail;
    }
    nnz_a = e_a->shape[0];
    a_row = (int *)safe_malloc((size_t)nnz_a * sizeof(int));
    a_col = (int *)safe_malloc((size_t)nnz_a * sizeof(int));
    a_val = (double *)safe_malloc((size_t)nnz_a * sizeof(double));
    const double *asrc = (const double *)e_a->data;
    int a_fo = e_a->fortran_order;
    for (long long i = 0; i < nnz_a; i++) {
      double v_cons, v_var, v_val;
      if (a_fo) {
        v_cons = asrc[0 * nnz_a + i];
        v_var = asrc[1 * nnz_a + i];
        v_val = asrc[2 * nnz_a + i];
      } else {
        v_cons = asrc[i * 3 + 0];
        v_var = asrc[i * 3 + 1];
        v_val = asrc[i * 3 + 2];
      }
      a_row[i] = (int)round_to_int(v_cons);
      a_col[i] = (int)round_to_int(v_var);
      a_val[i] = v_val;
    }
  }

  // ---- Assemble basic_sdp_t ----
  basic_sdp_t *sdp = (basic_sdp_t *)safe_malloc(sizeof(basic_sdp_t));
  sdp->m = m;
  sdp->n_cones = n_cones;
  sdp->blk_dims = blk_dims;
  sdp->right_hand_side = rhs;
  sdp->lp_dim = lp_dim;

  sdp->nnz_psd_constr = (int)nnz_A;
  sdp->psd_cone_constraints =
      (psd_cone_constraint_t *)safe_malloc(sizeof(psd_cone_constraint_t));
  sdp->psd_cone_constraints->constr_ind = A_constr;
  sdp->psd_cone_constraints->cone_ind = A_cone;
  sdp->psd_cone_constraints->row_ind = A_row;
  sdp->psd_cone_constraints->col_ind = A_col;
  sdp->psd_cone_constraints->val = A_val;

  sdp->nnz_psd_obj = (int)nnz_C;
  if (nnz_C > 0) {
    sdp->psd_cone_objective =
        (psd_cone_objective_t *)safe_malloc(sizeof(psd_cone_objective_t));
    sdp->psd_cone_objective->cone_ind = C_cone;
    sdp->psd_cone_objective->row_ind = C_row;
    sdp->psd_cone_objective->col_ind = C_col;
    sdp->psd_cone_objective->val = C_val;
  } else {
    sdp->psd_cone_objective = NULL;
  }

  sdp->nnz_lp_obj = lp_dim;
  sdp->lp_objective = lp_obj;

  sdp->nnz_lp_constr = (int)nnz_a;
  if (nnz_a > 0) {
    sdp->lp_constraints =
        (lp_constraint_t *)safe_malloc(sizeof(lp_constraint_t));
    sdp->lp_constraints->row_ind = a_row;
    sdp->lp_constraints->col_ind = a_col;
    sdp->lp_constraints->val = a_val;
  } else {
    sdp->lp_constraints = NULL;
  }

  LOG_DBG("[pdsdp npz] %s: cones=%d  m=%d  lp_dim=%d  "
          "nnz_A=%lld  nnz_C=%lld  nnz_a=%lld\n",
          filename, n_cones, m, lp_dim, nnz_A, nnz_C, nnz_a);

  npz_free(arc);
  return sdp;

fail:
  free(blk_dims);
  free(rhs);
  free(A_constr);
  free(A_cone);
  free(A_row);
  free(A_col);
  free(A_val);
  free(C_cone);
  free(C_row);
  free(C_col);
  free(C_val);
  free(lp_obj);
  free(a_row);
  free(a_col);
  free(a_val);
  npz_free(arc);
  return NULL;
}
