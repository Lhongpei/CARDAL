/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "parser.h"
#include "sdp_types.h"
#include "utils.h"
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef USE_MATIO

#include <matio.h>

typedef enum { VAR_FREE, VAR_LP, VAR_SOCP, VAR_SDP, VAR_UNKNOWN } var_type_t;

static double read_typed_as_double(const void *base, enum matio_types t,
                                   size_t index) {
  switch (t) {
  case MAT_T_DOUBLE:
    return ((const double *)base)[index];
  case MAT_T_SINGLE:
    return ((const float *)base)[index];
  case MAT_T_INT8:
    return ((const signed char *)base)[index];
  case MAT_T_UINT8:
    return ((const unsigned char *)base)[index];
  case MAT_T_INT16:
    return ((const short *)base)[index];
  case MAT_T_UINT16:
    return ((const unsigned short *)base)[index];
  case MAT_T_INT32:
    return ((const int *)base)[index];
  case MAT_T_UINT32:
    return ((const unsigned int *)base)[index];
  case MAT_T_INT64:
    return (double)((const long long *)base)[index];
  case MAT_T_UINT64:
    return (double)((const unsigned long long *)base)[index];
  default:
    return 0.0;
  }
}

static int read_index(const void *base, size_t index) {
  return ((const int *)base)[index];
}

static double get_scalar(matvar_t *var, int index) {
  if (!var || !var->data)
    return 0.0;
  return read_typed_as_double(var->data, var->data_type, (size_t)index);
}

static size_t matvar_numel(const matvar_t *var) {
  if (!var || !var->data || var->rank <= 0)
    return 0;

  size_t count = 1;
  for (int i = 0; i < var->rank; i++)
    count *= var->dims[i];
  return count;
}

static bool matvar_contains_complex(const matvar_t *var) {
  if (!var)
    return false;
  if (var->isComplex)
    return true;
  if (var->class_type != MAT_C_CELL || !var->data)
    return false;

  matvar_t **cells = (matvar_t **)var->data;
  size_t count = matvar_numel(var);
  for (size_t i = 0; i < count; i++) {
    if (matvar_contains_complex(cells[i]))
      return true;
  }
  return false;
}

static bool matvar_contains_nonzero(const matvar_t *var) {
  if (!var || !var->data)
    return false;
  if (var->isComplex)
    return true;

  if (var->class_type == MAT_C_CELL) {
    matvar_t **cells = (matvar_t **)var->data;
    size_t count = matvar_numel(var);
    for (size_t i = 0; i < count; i++) {
      if (matvar_contains_nonzero(cells[i]))
        return true;
    }
    return false;
  }

  if (var->class_type == MAT_C_SPARSE) {
    mat_sparse_t *sparse = (mat_sparse_t *)var->data;
    for (size_t i = 0; i < sparse->ndata; i++) {
      if (read_typed_as_double(sparse->data, var->data_type, i) != 0.0)
        return true;
    }
    return false;
  }

  switch (var->class_type) {
  case MAT_C_DOUBLE:
  case MAT_C_SINGLE:
  case MAT_C_INT8:
  case MAT_C_UINT8:
  case MAT_C_INT16:
  case MAT_C_UINT16:
  case MAT_C_INT32:
  case MAT_C_UINT32:
  case MAT_C_INT64:
  case MAT_C_UINT64:
    break;
  default:
    return false;
  }

  size_t count = matvar_numel(var);
  for (size_t i = 0; i < count; i++) {
    if (read_typed_as_double(var->data, var->data_type, i) != 0.0)
      return true;
  }
  return false;
}

static double matvar_matrix_value(const matvar_t *var, int row, int col) {
  if (!var || !var->data || var->rank < 2 || row < 0 || col < 0 ||
      row >= (int)var->dims[0] || col >= (int)var->dims[1])
    return 0.0;
  if (var->class_type == MAT_C_SPARSE) {
    const mat_sparse_t *sp = (const mat_sparse_t *)var->data;
    const int *ir = (const int *)sp->ir;
    const int *jc = (const int *)sp->jc;
    for (int p = jc[col]; p < jc[col + 1]; p++)
      if (ir[p] == row)
        return read_typed_as_double(sp->data, var->data_type, (size_t)p);
    return 0.0;
  }
  return read_typed_as_double(var->data, var->data_type,
                              (size_t)col * var->dims[0] + row);
}

static void jacobi_eig_sym_cpu(double *a, int n, double *eigvals,
                               double *eigvecs) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      eigvecs[i + j * n] = i == j ? 1.0 : 0.0;
  for (int sweep = 0; sweep < 100 * n * n; sweep++) {
    int p = 0, q = 0;
    double max_off = 0.0;
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        double v = fabs(a[i * n + j]);
        if (v > max_off) {
          max_off = v;
          p = i;
          q = j;
        }
      }
    }
    if (max_off < 1e-13)
      break;
    double app = a[p * n + p], aqq = a[q * n + q];
    double apq = a[p * n + q];
    double phi = 0.5 * atan2(2.0 * apq, aqq - app);
    double c = cos(phi), s = sin(phi);
    for (int k = 0; k < n; k++) {
      double apk = a[p * n + k], aqk = a[q * n + k];
      a[p * n + k] = c * apk - s * aqk;
      a[q * n + k] = s * apk + c * aqk;
    }
    for (int k = 0; k < n; k++) {
      double akp = a[k * n + p], akq = a[k * n + q];
      a[k * n + p] = c * akp - s * akq;
      a[k * n + q] = s * akp + c * akq;
    }
    for (int k = 0; k < n; k++) {
      double vkp = eigvecs[k + p * n];
      double vkq = eigvecs[k + q * n];
      eigvecs[k + p * n] = c * vkp - s * vkq;
      eigvecs[k + q * n] = s * vkp + c * vkq;
    }
  }
  for (int i = 0; i < n; i++)
    eigvals[i] = a[i * n + i];
}

static int append_sdpt3_low_rank(const matvar_t *rank_var,
                                 const matvar_t *v_var, const matvar_t *d_var,
                                 int constraint_start, int cone, int dim,
                                 symmetric_low_rank_data_t *out,
                                 int *column_write, long long *factor_write) {
  int n_terms = (int)matvar_numel(rank_var);
  int total_rank = 0;
  int *ranks = (int *)safe_malloc((size_t)n_terms * sizeof(int));
  for (int t = 0; t < n_terms; t++) {
    double raw = get_scalar((matvar_t *)rank_var, t);
    if (!isfinite(raw) || raw < 1.0 || raw > INT_MAX ||
        fabs(raw - nearbyint(raw)) > 1e-9 ||
        raw > (double)(INT_MAX - total_rank)) {
      fprintf(stderr, "Error: invalid SDPT3 low-rank rank value %.17g.\n", raw);
      free(ranks);
      return 0;
    }
    ranks[t] = (int)raw;
    total_rank += ranks[t];
  }
  if (!v_var || !d_var || v_var->rank < 2 ||
      (v_var->class_type != MAT_C_SPARSE && v_var->class_type != MAT_C_DOUBLE &&
       v_var->class_type != MAT_C_SINGLE) ||
      (d_var->class_type != MAT_C_SPARSE && d_var->class_type != MAT_C_DOUBLE &&
       d_var->class_type != MAT_C_SINGLE) ||
      (int)v_var->dims[0] != dim || (int)v_var->dims[1] != total_rank) {
    fprintf(stderr,
            "Error: invalid SDPT3 low-rank V data for block %d; expected "
            "%d-by-%d factors.\n",
            cone + 1, dim, total_rank);
    free(ranks);
    return 0;
  }

  int diagonal_d =
      matvar_numel(d_var) == (size_t)total_rank &&
      (d_var->rank == 1 || d_var->dims[0] == 1 || d_var->dims[1] == 1);
  int dense_d = d_var->rank >= 2 && (int)d_var->dims[0] == total_rank &&
                (int)d_var->dims[1] == total_rank;
  int triplet_d = d_var->rank >= 2 && (int)d_var->dims[1] == 4 && !dense_d;
  if (!diagonal_d && !dense_d && !triplet_d) {
    fprintf(stderr,
            "Error: unsupported SDPT3 low-rank D data in block %d. Expected "
            "a diagonal vector, a square block matrix, or four-column "
            "triplets.\n",
            cone + 1);
    free(ranks);
    return 0;
  }

  if (triplet_d) {
    int rows = (int)d_var->dims[0];
    for (int e = 0; e < rows; e++) {
      double term_raw = matvar_matrix_value(d_var, e, 0);
      double row_raw = matvar_matrix_value(d_var, e, 1);
      double col_raw = matvar_matrix_value(d_var, e, 2);
      double value = matvar_matrix_value(d_var, e, 3);
      if (!isfinite(term_raw) || !isfinite(row_raw) || !isfinite(col_raw) ||
          !isfinite(value) || term_raw < 1.0 || term_raw > n_terms ||
          fabs(term_raw - nearbyint(term_raw)) > 1e-9 ||
          fabs(row_raw - nearbyint(row_raw)) > 1e-9 ||
          fabs(col_raw - nearbyint(col_raw)) > 1e-9) {
        fprintf(stderr, "Error: invalid SDPT3 low-rank D triplet at row %d.\n",
                e + 1);
        free(ranks);
        return 0;
      }
      int term = (int)term_raw - 1;
      if (row_raw < 1.0 || row_raw > ranks[term] || col_raw < 1.0 ||
          col_raw > ranks[term]) {
        fprintf(stderr,
                "Error: SDPT3 low-rank D triplet row %d is outside term %d "
                "of rank %d.\n",
                e + 1, term + 1, ranks[term]);
        free(ranks);
        return 0;
      }
    }
  }

  int rank_offset = 0;
  for (int term = 0; term < n_terms; term++) {
    int r = ranks[term];
    double *core = (double *)safe_calloc((size_t)r * r, sizeof(double));
    if (diagonal_d) {
      for (int j = 0; j < r; j++) {
        int index = rank_offset + j;
        int row = index % (int)d_var->dims[0];
        int col = index / (int)d_var->dims[0];
        core[j * r + j] = matvar_matrix_value(d_var, row, col);
        if (!isfinite(core[j * r + j])) {
          fprintf(stderr, "Error: non-finite SDPT3 low-rank D value.\n");
          free(core);
          free(ranks);
          return 0;
        }
      }
    } else if (dense_d) {
      for (int col = 0; col < r; col++) {
        for (int row = 0; row < r; row++) {
          core[row * r + col] =
              matvar_matrix_value(d_var, rank_offset + row, rank_offset + col);
          if (!isfinite(core[row * r + col])) {
            fprintf(stderr, "Error: non-finite SDPT3 low-rank D value.\n");
            free(core);
            free(ranks);
            return 0;
          }
        }
      }
    } else {
      int rows = (int)d_var->dims[0];
      for (int e = 0; e < rows; e++) {
        int term_id = (int)matvar_matrix_value(d_var, e, 0) - 1;
        if (term_id != term)
          continue;
        int row = (int)matvar_matrix_value(d_var, e, 1) - 1;
        int col = (int)matvar_matrix_value(d_var, e, 2) - 1;
        double value = matvar_matrix_value(d_var, e, 3);
        core[row * r + col] = value;
      }
    }
    for (int row = 0; row < r; row++) {
      for (int col = row + 1; col < r; col++) {
        double a = core[row * r + col], b = core[col * r + row];
        double value = (a == 0.0) ? b : ((b == 0.0) ? a : 0.5 * (a + b));
        core[row * r + col] = value;
        core[col * r + row] = value;
      }
    }

    double *eigvals = (double *)safe_malloc((size_t)r * sizeof(double));
    double *eigvecs = (double *)safe_malloc((size_t)r * r * sizeof(double));
    jacobi_eig_sym_cpu(core, r, eigvals, eigvecs);
    for (int eig = 0; eig < r; eig++) {
      int out_col = (*column_write)++;
      out->constraint_ind[out_col] = constraint_start + term;
      out->cone_ind[out_col] = cone;
      out->weights[out_col] = eigvals[eig];
      for (int row = 0; row < dim; row++) {
        double value = 0.0;
        for (int j = 0; j < r; j++) {
          double factor = matvar_matrix_value(v_var, row, rank_offset + j);
          if (!isfinite(factor)) {
            fprintf(stderr, "Error: non-finite SDPT3 low-rank V value.\n");
            free(eigvecs);
            free(eigvals);
            free(core);
            free(ranks);
            return 0;
          }
          value += factor * eigvecs[j + eig * r];
        }
        out->factor_values[(*factor_write)++] = value;
      }
      out->factor_ptr[out_col + 1] = *factor_write;
    }
    rank_offset += r;
    free(eigvecs);
    free(eigvals);
    free(core);
  }
  free(ranks);
  return 1;
}

static void free_matvars(matvar_t *const *vars, size_t count) {
  for (size_t i = 0; i < count; i++) {
    if (vars[i])
      Mat_VarFree(vars[i]);
  }
}

static void map_sedumi_var(int var_idx, int K_f, int K_l, int K_q_dim,
                           int n_cones, const int *blk_dims, var_type_t *type,
                           int *local_idx, int *blk_idx, int *row, int *col) {
  int offset = 0;

  if (var_idx < offset + K_f) {
    *type = VAR_FREE;
    *local_idx = var_idx - offset;
    return;
  }
  offset += K_f;

  if (var_idx < offset + K_l) {
    *type = VAR_LP;
    *local_idx = var_idx - offset;
    return;
  }
  offset += K_l;

  if (var_idx < offset + K_q_dim) {
    *type = VAR_SOCP;
    *local_idx = var_idx - offset;
    return;
  }
  offset += K_q_dim;

  for (int b = 0; b < n_cones; b++) {
    int dim = blk_dims[b];
    int size = dim * dim;
    if (var_idx < offset + size) {
      *type = VAR_SDP;
      *blk_idx = b;
      int idx = var_idx - offset;
      *row = idx % dim;
      *col = idx / dim;
      return;
    }
    offset += size;
  }
  *type = VAR_UNKNOWN;
}

static inline void process_c_elem(int var_idx, double val, int K_f, int K_l,
                                  int K_q_dim, int n_cones, const int *blk_dims,
                                  basic_sdp_t *sdp, int *o_count) {
  var_type_t vtype;
  int local_idx, blk, r, c;
  map_sedumi_var(var_idx, K_f, K_l, K_q_dim, n_cones, blk_dims, &vtype,
                 &local_idx, &blk, &r, &c);

  if (vtype == VAR_LP) {
    sdp->lp_objective[local_idx] = -val;
  } else if (vtype == VAR_FREE) {
    sdp->free_objective[local_idx] = -val;
  } else if (vtype == VAR_SDP) {
    if (r >= c) {
      sdp->psd_cone_objective->cone_ind[*o_count] = blk;
      sdp->psd_cone_objective->row_ind[*o_count] = r;
      sdp->psd_cone_objective->col_ind[*o_count] = c;
      sdp->psd_cone_objective->val[*o_count] = val;
      (*o_count)++;
    }
  }
}

static basic_sdp_t *read_sedumi_mat(mat_t *matfp, matvar_t *root_struct) {
  matvar_t *At_var = NULL, *A_var = NULL, *b_var = NULL, *c_var = NULL,
           *K_var = NULL;

  if (root_struct) {
    At_var = Mat_VarGetStructFieldByName(root_struct, "At", 0);
    A_var = Mat_VarGetStructFieldByName(root_struct, "A", 0);
    b_var = Mat_VarGetStructFieldByName(root_struct, "b", 0);
    c_var = Mat_VarGetStructFieldByName(root_struct, "c", 0);
    K_var = Mat_VarGetStructFieldByName(root_struct, "K", 0);
  } else {
    At_var = Mat_VarRead(matfp, "At");
    if (!At_var)
      A_var = Mat_VarRead(matfp, "A");
    b_var = Mat_VarRead(matfp, "b");
    c_var = Mat_VarRead(matfp, "c");
    K_var = Mat_VarRead(matfp, "K");
  }

  if ((!At_var && !A_var) || !b_var || !c_var || !K_var) {
    fprintf(stderr,
            "Fatal Error: Invalid SeDuMi format. Missing At/A, b, c, or K.\n");
    return NULL;
  }
  matvar_t *K_f_var = Mat_VarGetStructFieldByName(K_var, "f", 0);
  matvar_t *K_l_var = Mat_VarGetStructFieldByName(K_var, "l", 0);
  matvar_t *K_q_var = Mat_VarGetStructFieldByName(K_var, "q", 0);
  matvar_t *K_r_var = Mat_VarGetStructFieldByName(K_var, "r", 0);
  matvar_t *K_s_var = Mat_VarGetStructFieldByName(K_var, "s", 0);
  matvar_t *K_xcomplex_var = Mat_VarGetStructFieldByName(K_var, "xcomplex", 0);
  matvar_t *K_scomplex_var = Mat_VarGetStructFieldByName(K_var, "scomplex", 0);
  matvar_t *K_ycomplex_var = Mat_VarGetStructFieldByName(K_var, "ycomplex", 0);

  int K_f = (int)get_scalar(K_f_var, 0);
  int K_l = (int)get_scalar(K_l_var, 0);

  int K_q_dim = 0;
  if (matvar_contains_nonzero(K_q_var)) {
    fprintf(stderr, "Error: Unsupported SeDuMi cone K.q (second-order cone). "
                    "Reformulate it as PSD data before using CARDAL.\n");
    if (!root_struct) {
      matvar_t *vars[] = {At_var, A_var, b_var, c_var, K_var};
      free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
    }
    return NULL;
  }

  if (matvar_contains_nonzero(K_r_var)) {
    fprintf(stderr,
            "Error: Unsupported SeDuMi cone K.r (rotated second-order cone). "
            "Reformulate it as PSD data before using CARDAL.\n");
    if (!root_struct) {
      matvar_t *vars[] = {At_var, A_var, b_var, c_var, K_var};
      free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
    }
    return NULL;
  }

  if (matvar_numel(K_xcomplex_var) > 0 || matvar_numel(K_scomplex_var) > 0 ||
      matvar_numel(K_ycomplex_var) > 0 || matvar_contains_complex(At_var) ||
      matvar_contains_complex(A_var) || matvar_contains_complex(b_var) ||
      matvar_contains_complex(c_var)) {
    fprintf(stderr,
            "Error: Unsupported complex SeDuMi data "
            "(K.xcomplex/K.scomplex/K.ycomplex or complex coefficients). "
            "Convert it to a real SDP before using CARDAL.\n");
    if (!root_struct) {
      matvar_t *vars[] = {At_var, A_var, b_var, c_var, K_var};
      free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
    }
    return NULL;
  }
  int n_cones = 0;
  int *blk_dims = NULL;
  if (K_s_var && K_s_var->data) {
    n_cones = (int)(K_s_var->dims[0] * K_s_var->dims[1]);
    blk_dims = (int *)safe_malloc(n_cones * sizeof(int));
    for (int i = 0; i < n_cones; i++)
      blk_dims[i] = (int)get_scalar(K_s_var, i);
  }

  int m =
      (int)(b_var->dims[0] > b_var->dims[1] ? b_var->dims[0] : b_var->dims[1]);

  basic_sdp_t *sdp = (basic_sdp_t *)calloc(1, sizeof(basic_sdp_t));
  sdp->m = m;
  sdp->n_cones = n_cones;
  sdp->lp_dim = K_l;
  sdp->free_dim = K_f;
  sdp->blk_dims = blk_dims;

  int max_At_nnz = 0;
  if (At_var && At_var->class_type == MAT_C_SPARSE)
    max_At_nnz = (int)((mat_sparse_t *)At_var->data)->ndata;
  else if (A_var && A_var->class_type == MAT_C_SPARSE)
    max_At_nnz = (int)((mat_sparse_t *)A_var->data)->ndata;

  int max_c_nnz = (c_var->class_type == MAT_C_SPARSE)
                      ? (int)((mat_sparse_t *)c_var->data)->ndata
                      : (int)(c_var->dims[0] * c_var->dims[1]);

  sdp->psd_cone_constraints =
      (psd_cone_constraint_t *)calloc(1, sizeof(psd_cone_constraint_t));
  sdp->psd_cone_constraints->constr_ind =
      (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->psd_cone_constraints->cone_ind =
      (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->psd_cone_constraints->row_ind =
      (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->psd_cone_constraints->col_ind =
      (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->psd_cone_constraints->val =
      (double *)safe_malloc(max_At_nnz * sizeof(double));

  sdp->lp_constraints = (lp_constraint_t *)calloc(1, sizeof(lp_constraint_t));
  sdp->lp_constraints->row_ind = (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->lp_constraints->col_ind = (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->lp_constraints->val = (double *)safe_malloc(max_At_nnz * sizeof(double));
  sdp->free_constraints =
      (free_constraint_t *)calloc(1, sizeof(free_constraint_t));
  sdp->free_constraints->row_ind = (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->free_constraints->col_ind = (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->free_constraints->val =
      (double *)safe_malloc(max_At_nnz * sizeof(double));

  sdp->psd_cone_objective =
      (psd_cone_objective_t *)calloc(1, sizeof(psd_cone_objective_t));
  sdp->psd_cone_objective->cone_ind =
      (int *)safe_malloc(max_c_nnz * sizeof(int));
  sdp->psd_cone_objective->row_ind =
      (int *)safe_malloc(max_c_nnz * sizeof(int));
  sdp->psd_cone_objective->col_ind =
      (int *)safe_malloc(max_c_nnz * sizeof(int));
  sdp->psd_cone_objective->val =
      (double *)safe_malloc(max_c_nnz * sizeof(double));

  sdp->lp_objective = (double *)calloc((K_l > 0 ? K_l : 1), sizeof(double));
  sdp->free_objective = (double *)calloc((K_f > 0 ? K_f : 1), sizeof(double));
  sdp->right_hand_side = (double *)calloc(m, sizeof(double));

  if (b_var->class_type == MAT_C_SPARSE) {
    mat_sparse_t *b_sp = (mat_sparse_t *)b_var->data;
    int *b_ir = (int *)b_sp->ir;
    for (int i = 0; i < (int)b_sp->ndata; i++)
      sdp->right_hand_side[b_ir[i]] =
          read_typed_as_double(b_sp->data, b_var->data_type, i);
  } else {
    for (int i = 0; i < m; i++)
      sdp->right_hand_side[i] =
          read_typed_as_double(b_var->data, b_var->data_type, i);
  }

  int o_count = 0;

  if (c_var->class_type == MAT_C_SPARSE) {
    mat_sparse_t *c_sp = (mat_sparse_t *)c_var->data;
    for (int i = 0; i < (int)c_sp->ndata; i++) {
      process_c_elem(((int *)c_sp->ir)[i],
                     read_typed_as_double(c_sp->data, c_var->data_type, i), K_f,
                     K_l, K_q_dim, n_cones, blk_dims, sdp, &o_count);
    }
  } else {
    int N = (int)(c_var->dims[0] > c_var->dims[1] ? c_var->dims[0]
                                                  : c_var->dims[1]);
    for (int i = 0; i < N; i++) {
      double v = read_typed_as_double(c_var->data, c_var->data_type, i);
      if (v != 0.0) {
        process_c_elem(i, v, K_f, K_l, K_q_dim, n_cones, blk_dims, sdp,
                       &o_count);
      }
    }
  }

  int c_count = 0, lp_c_count = 0, free_c_count = 0;

  if (At_var && At_var->class_type == MAT_C_SPARSE) {
    mat_sparse_t *At_sp = (mat_sparse_t *)At_var->data;
    int *At_ir = (int *)At_sp->ir;
    int *At_jc = (int *)At_sp->jc;
    enum matio_types At_t = At_var->data_type;

    for (int j = 0; j < m; j++) {
      for (int k = At_jc[j]; k < At_jc[j + 1]; k++) {
        var_type_t vtype;
        int local_idx, blk, r, c;
        map_sedumi_var(At_ir[k], K_f, K_l, K_q_dim, n_cones, blk_dims, &vtype,
                       &local_idx, &blk, &r, &c);

        double v = read_typed_as_double(At_sp->data, At_t, k);
        if (vtype == VAR_LP) {
          sdp->lp_constraints->row_ind[lp_c_count] = j;
          sdp->lp_constraints->col_ind[lp_c_count] = local_idx;
          sdp->lp_constraints->val[lp_c_count] = v;
          lp_c_count++;
        } else if (vtype == VAR_FREE) {
          sdp->free_constraints->row_ind[free_c_count] = j;
          sdp->free_constraints->col_ind[free_c_count] = local_idx;
          sdp->free_constraints->val[free_c_count] = v;
          free_c_count++;
        } else if (vtype == VAR_SDP && r >= c) {
          sdp->psd_cone_constraints->constr_ind[c_count] = j;
          sdp->psd_cone_constraints->cone_ind[c_count] = blk;
          sdp->psd_cone_constraints->row_ind[c_count] = r;
          sdp->psd_cone_constraints->col_ind[c_count] = c;
          sdp->psd_cone_constraints->val[c_count] = v;
          c_count++;
        }
      }
    }
  } else if (A_var && A_var->class_type == MAT_C_SPARSE) {
    mat_sparse_t *A_sp = (mat_sparse_t *)A_var->data;
    int *A_ir = (int *)A_sp->ir; // Constraint Index
    int *A_jc = (int *)A_sp->jc; // Variable Column Pointer
    enum matio_types A_t = A_var->data_type;

    int N = A_var->dims[1];
    for (int var_idx = 0; var_idx < N; var_idx++) {
      var_type_t vtype = VAR_UNKNOWN;
      int local_idx = 0, blk = 0, r = 0, c = 0;
      map_sedumi_var(var_idx, K_f, K_l, K_q_dim, n_cones, blk_dims, &vtype,
                     &local_idx, &blk, &r, &c);

      for (int k = A_jc[var_idx]; k < A_jc[var_idx + 1]; k++) {
        double v = read_typed_as_double(A_sp->data, A_t, k);
        if (vtype == VAR_LP) {
          sdp->lp_constraints->row_ind[lp_c_count] = A_ir[k];
          sdp->lp_constraints->col_ind[lp_c_count] = local_idx;
          sdp->lp_constraints->val[lp_c_count] = v;
          lp_c_count++;
        } else if (vtype == VAR_FREE) {
          sdp->free_constraints->row_ind[free_c_count] = A_ir[k];
          sdp->free_constraints->col_ind[free_c_count] = local_idx;
          sdp->free_constraints->val[free_c_count] = v;
          free_c_count++;
        } else if (vtype == VAR_SDP && r >= c) {
          sdp->psd_cone_constraints->constr_ind[c_count] = A_ir[k];
          sdp->psd_cone_constraints->cone_ind[c_count] = blk;
          sdp->psd_cone_constraints->row_ind[c_count] = r;
          sdp->psd_cone_constraints->col_ind[c_count] = c;
          sdp->psd_cone_constraints->val[c_count] = v;
          c_count++;
        }
      }
    }
  }

  sdp->nnz_psd_constr = c_count;
  sdp->nnz_psd_obj = o_count;
  sdp->nnz_lp_constr = lp_c_count;
  sdp->nnz_lp_obj = K_l; // Objective is dense sized array
  sdp->nnz_free_constr = free_c_count;
  sdp->nnz_free_obj = K_f;

  LOG_DBG("  -> SeDuMi parsed: m=%d, PSD cones=%d, LP dim=%d, free dim=%d\n",
          sdp->m, sdp->n_cones, sdp->lp_dim, sdp->free_dim);
  LOG_DBG("     PSD NNZ: A=%d, C=%d | LP NNZ: A=%d | Free NNZ: A=%d\n",
          sdp->nnz_psd_constr, sdp->nnz_psd_obj, sdp->nnz_lp_constr,
          sdp->nnz_free_constr);

  if (!root_struct) {
    if (At_var)
      Mat_VarFree(At_var);
    if (A_var)
      Mat_VarFree(A_var);
    Mat_VarFree(b_var);
    Mat_VarFree(c_var);
    Mat_VarFree(K_var);
  }

  return sdp;
}

static basic_sdp_t *read_sdpt3_mat(mat_t *matfp, matvar_t *root_struct) {
  matvar_t *blk_var = NULL, *At_var = NULL, *C_var = NULL, *b_var = NULL,
           *options_var = NULL;

  if (root_struct) {
    blk_var = Mat_VarGetStructFieldByName(root_struct, "blk", 0);
    At_var = Mat_VarGetStructFieldByName(root_struct, "At", 0);
    C_var = Mat_VarGetStructFieldByName(root_struct, "C", 0);
    b_var = Mat_VarGetStructFieldByName(root_struct, "b", 0);
    options_var = Mat_VarGetStructFieldByName(root_struct, "OPTIONS", 0);
  } else {
    blk_var = Mat_VarRead(matfp, "blk");
    At_var = Mat_VarRead(matfp, "At");
    C_var = Mat_VarRead(matfp, "C");
    b_var = Mat_VarRead(matfp, "b");
    options_var = Mat_VarRead(matfp, "OPTIONS");
  }

  if (!blk_var || !At_var || !C_var || !b_var ||
      blk_var->class_type != MAT_C_CELL || At_var->class_type != MAT_C_CELL ||
      C_var->class_type != MAT_C_CELL || blk_var->rank < 2 ||
      blk_var->dims[1] < 2) {
    fprintf(stderr, "Fatal Error: Invalid SDPT3 format.\n");
    if (!root_struct) {
      matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
      free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
    }
    return NULL;
  }

  int K_blocks = blk_var->dims[0];
  if (At_var->rank < 2 || C_var->rank < 2 ||
      At_var->dims[0] < (size_t)K_blocks || C_var->dims[0] < (size_t)K_blocks) {
    fprintf(stderr,
            "Fatal Error: Invalid SDPT3 format. blk, At, and C block counts "
            "do not match.\n");
    if (!root_struct) {
      matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
      free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
    }
    return NULL;
  }

  if (matvar_contains_complex(At_var) || matvar_contains_complex(C_var) ||
      matvar_contains_complex(b_var)) {
    fprintf(stderr,
            "Error: Unsupported complex SDPT3 data. Convert it to a real SDP "
            "before using CARDAL.\n");
    if (!root_struct) {
      matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
      free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
    }
    return NULL;
  }

  matvar_t *parbarrier_var = NULL;
  if (options_var && options_var->class_type == MAT_C_STRUCT)
    parbarrier_var = Mat_VarGetStructFieldByName(options_var, "parbarrier", 0);
  if (matvar_contains_nonzero(parbarrier_var)) {
    fprintf(stderr,
            "Error: Unsupported SDPT3 OPTIONS.parbarrier objective. CARDAL "
            "requires a linear objective.\n");
    if (!root_struct) {
      matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
      free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
    }
    return NULL;
  }

  matvar_t **blk_cells = (matvar_t **)blk_var->data;
  matvar_t **At_cells = (matvar_t **)At_var->data;
  matvar_t **C_cells = (matvar_t **)C_var->data;

  int m =
      (int)(b_var->dims[0] > b_var->dims[1] ? b_var->dims[0] : b_var->dims[1]);
  int total_n_cones = 0;
  int total_lp_dim = 0;
  int total_free_dim = 0;
  int total_low_rank_columns = 0;
  long long total_low_rank_factors = 0;

  for (int p = 0; p < K_blocks; p++) {
    matvar_t *type_cell = blk_cells[p];
    matvar_t *size_cell = blk_cells[p + K_blocks];
    if (!type_cell || !type_cell->data || !size_cell || !size_cell->data) {
      fprintf(stderr,
              "Fatal Error: Invalid SDPT3 block description at block %d.\n",
              p + 1);
      if (!root_struct) {
        matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
        free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
      }
      return NULL;
    }
    char *blk_type = (char *)type_cell->data;
    int num_sub_blocks = size_cell->nbytes / size_cell->data_size;
    matvar_t *rank_cell =
        blk_var->dims[1] > 2 ? blk_cells[p + 2 * K_blocks] : NULL;
    int low_rank_terms = (int)matvar_numel(rank_cell);

    if (blk_type[0] == 'q' || blk_type[0] == 'r') {
      fprintf(stderr,
              "Error: Unsupported SDPT3 '%c' block at block %d. Reformulate "
              "second-order cones as PSD data before using CARDAL.\n",
              blk_type[0], p + 1);
      if (!root_struct) {
        matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
        free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
      }
      return NULL;
    }
    if (blk_type[0] != 's' && blk_type[0] != 'l' && blk_type[0] != 'u') {
      fprintf(stderr,
              "Error: Unsupported SDPT3 block type '%c' at block %d. CARDAL "
              "supports only 's', 'l', and 'u' blocks.\n",
              blk_type[0], p + 1);
      if (!root_struct) {
        matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
        free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
      }
      return NULL;
    }

    if (low_rank_terms > 0) {
      if (blk_type[0] != 's' || num_sub_blocks != 1 || At_var->dims[1] < 3 ||
          matvar_numel(At_cells[p + K_blocks]) == 0 ||
          matvar_numel(At_cells[p + 2 * K_blocks]) == 0) {
        fprintf(stderr,
                "Error: invalid SDPT3 low-rank block %d. Low-rank data "
                "requires one PSD sub-block and At{p,2:3}.\n",
                p + 1);
        if (!root_struct) {
          matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
          free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
        }
        return NULL;
      }
      int rank_sum = 0;
      for (int t = 0; t < low_rank_terms; t++) {
        double raw = get_scalar(rank_cell, t);
        int rank = (int)raw;
        if (rank <= 0 || fabs(raw - rank) > 1e-9) {
          fprintf(stderr, "Error: invalid SDPT3 low-rank rank in block %d.\n",
                  p + 1);
          if (!root_struct) {
            matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
            free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
          }
          return NULL;
        }
        rank_sum += rank;
      }
      int normal_constraints =
          At_cells[p] && At_cells[p]->rank >= 2 ? (int)At_cells[p]->dims[1] : 0;
      if (normal_constraints + low_rank_terms != m) {
        fprintf(stderr,
                "Error: SDPT3 low-rank block %d describes %d ordinary and "
                "%d low-rank constraints, but m=%d.\n",
                p + 1, normal_constraints, low_rank_terms, m);
        if (!root_struct) {
          matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
          free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
        }
        return NULL;
      }
      int dim = (int)get_scalar(size_cell, 0);
      total_low_rank_columns += rank_sum;
      total_low_rank_factors += (long long)dim * rank_sum;
    } else if (At_var->dims[1] > 1) {
      matvar_t *v_cell = At_cells[p + K_blocks];
      matvar_t *d_cell =
          At_var->dims[1] > 2 ? At_cells[p + 2 * K_blocks] : NULL;
      if (matvar_numel(v_cell) > 0 || matvar_numel(d_cell) > 0) {
        fprintf(stderr,
                "Error: SDPT3 At{p,2:3} is present without blk{p,3} "
                "rank metadata at block %d.\n",
                p + 1);
        if (!root_struct) {
          matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
          free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
        }
        return NULL;
      }
    }

    if (blk_type[0] == 's')
      total_n_cones += num_sub_blocks;
    else if (blk_type[0] == 'l') {
      for (int i = 0; i < num_sub_blocks; i++)
        total_lp_dim += (int)get_scalar(size_cell, i);
    } else if (blk_type[0] == 'u') {
      for (int i = 0; i < num_sub_blocks; i++)
        total_free_dim += (int)get_scalar(size_cell, i);
    }
  }

  basic_sdp_t *sdp = (basic_sdp_t *)calloc(1, sizeof(basic_sdp_t));
  sdp->m = m;
  sdp->n_cones = total_n_cones;
  sdp->lp_dim = total_lp_dim;
  sdp->free_dim = total_free_dim;
  sdp->blk_dims = (int *)safe_malloc(total_n_cones * sizeof(int));
  sdp->right_hand_side = (double *)calloc(m, sizeof(double));

  if (b_var->class_type == MAT_C_SPARSE) {
    mat_sparse_t *b_sp = (mat_sparse_t *)b_var->data;
    for (int i = 0; i < (int)b_sp->ndata; i++)
      sdp->right_hand_side[((int *)b_sp->ir)[i]] =
          read_typed_as_double(b_sp->data, b_var->data_type, i);
  } else {
    for (int i = 0; i < m; i++)
      sdp->right_hand_side[i] =
          read_typed_as_double(b_var->data, b_var->data_type, i);
  }

  sdp->lp_objective =
      (double *)calloc((total_lp_dim > 0 ? total_lp_dim : 1), sizeof(double));
  sdp->free_objective = (double *)calloc(
      (total_free_dim > 0 ? total_free_dim : 1), sizeof(double));
  if (total_low_rank_columns > 0) {
    sdp->low_rank_data = (symmetric_low_rank_data_t *)safe_calloc(
        1, sizeof(symmetric_low_rank_data_t));
    sdp->low_rank_data->num_columns = total_low_rank_columns;
    sdp->low_rank_data->constraint_ind =
        (int *)safe_malloc((size_t)total_low_rank_columns * sizeof(int));
    sdp->low_rank_data->cone_ind =
        (int *)safe_malloc((size_t)total_low_rank_columns * sizeof(int));
    sdp->low_rank_data->factor_ptr = (long long *)safe_malloc(
        (size_t)(total_low_rank_columns + 1) * sizeof(long long));
    sdp->low_rank_data->factor_values =
        (double *)safe_malloc((size_t)total_low_rank_factors * sizeof(double));
    sdp->low_rank_data->weights =
        (double *)safe_malloc((size_t)total_low_rank_columns * sizeof(double));
    sdp->low_rank_data->factor_ptr[0] = 0;
  }

  int max_At_nnz = 0, max_C_nnz = 0;
  for (int p = 0; p < K_blocks; p++) {
    if (At_cells[p]->class_type == MAT_C_SPARSE)
      max_At_nnz += ((mat_sparse_t *)At_cells[p]->data)->ndata;
    else if (At_cells[p]->class_type == MAT_C_DOUBLE)
      max_At_nnz += At_cells[p]->dims[0] * At_cells[p]->dims[1];

    if (C_cells[p]->class_type == MAT_C_SPARSE)
      max_C_nnz += ((mat_sparse_t *)C_cells[p]->data)->ndata;
    else if (C_cells[p]->class_type == MAT_C_DOUBLE)
      max_C_nnz += C_cells[p]->dims[0] * C_cells[p]->dims[1];
  }
  if (max_At_nnz == 0)
    max_At_nnz = 10000;
  if (max_C_nnz == 0)
    max_C_nnz = 10000;

  sdp->psd_cone_constraints =
      (psd_cone_constraint_t *)calloc(1, sizeof(psd_cone_constraint_t));
  sdp->psd_cone_constraints->constr_ind =
      (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->psd_cone_constraints->cone_ind =
      (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->psd_cone_constraints->row_ind =
      (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->psd_cone_constraints->col_ind =
      (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->psd_cone_constraints->val =
      (double *)safe_malloc(max_At_nnz * sizeof(double));

  sdp->lp_constraints = (lp_constraint_t *)calloc(1, sizeof(lp_constraint_t));
  sdp->lp_constraints->row_ind = (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->lp_constraints->col_ind = (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->lp_constraints->val = (double *)safe_malloc(max_At_nnz * sizeof(double));
  sdp->free_constraints =
      (free_constraint_t *)calloc(1, sizeof(free_constraint_t));
  sdp->free_constraints->row_ind = (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->free_constraints->col_ind = (int *)safe_malloc(max_At_nnz * sizeof(int));
  sdp->free_constraints->val =
      (double *)safe_malloc(max_At_nnz * sizeof(double));

  sdp->psd_cone_objective =
      (psd_cone_objective_t *)calloc(1, sizeof(psd_cone_objective_t));
  sdp->psd_cone_objective->cone_ind =
      (int *)safe_malloc(max_C_nnz * sizeof(int));
  sdp->psd_cone_objective->row_ind =
      (int *)safe_malloc(max_C_nnz * sizeof(int));
  sdp->psd_cone_objective->col_ind =
      (int *)safe_malloc(max_C_nnz * sizeof(int));
  sdp->psd_cone_objective->val =
      (double *)safe_malloc(max_C_nnz * sizeof(double));

  int current_cone_idx = 0, current_lp_offset = 0, current_free_offset = 0;
  int c_count = 0, o_count = 0, lp_c_count = 0, free_c_count = 0;
  int low_rank_column = 0;
  long long low_rank_factor = 0;

  for (int p = 0; p < K_blocks; p++) {
    matvar_t *type_cell = blk_cells[p], *size_cell = blk_cells[p + K_blocks];
    char *blk_type = (char *)type_cell->data;
    int num_sub_blocks = size_cell->nbytes / size_cell->data_size;
    matvar_t *Cp = C_cells[p], *Atp = At_cells[p];

    if (blk_type[0] == 'l' || blk_type[0] == 'u') {
      int is_free = blk_type[0] == 'u';
      int linear_size = 0;
      for (int i = 0; i < num_sub_blocks; i++)
        linear_size += (int)get_scalar(size_cell, i);
      int linear_offset = is_free ? current_free_offset : current_lp_offset;
      double *linear_objective =
          is_free ? sdp->free_objective : sdp->lp_objective;
      free_constraint_t *linear_constraints =
          is_free ? sdp->free_constraints : sdp->lp_constraints;
      int *linear_count = is_free ? &free_c_count : &lp_c_count;

      if (Cp->class_type == MAT_C_SPARSE) {
        mat_sparse_t *sp = (mat_sparse_t *)Cp->data;
        for (int i = 0; i < (int)sp->ndata; i++)
          linear_objective[linear_offset + ((int *)sp->ir)[i]] =
              -read_typed_as_double(sp->data, Cp->data_type, i);
      } else if (Cp->class_type == MAT_C_DOUBLE ||
                 Cp->class_type == MAT_C_SINGLE) {
        for (int i = 0; i < linear_size; i++)
          linear_objective[linear_offset + i] =
              -read_typed_as_double(Cp->data, Cp->data_type, i);
      }

      if (Atp->class_type == MAT_C_SPARSE) {
        mat_sparse_t *sp = (mat_sparse_t *)Atp->data;
        int *ir = (int *)sp->ir, *jc = (int *)sp->jc;
        enum matio_types t = Atp->data_type;
        int normal_cols = (int)Atp->dims[1];
        for (int j = 0; j < normal_cols; j++) {
          for (int k = jc[j]; k < jc[j + 1]; k++) {
            linear_constraints->row_ind[*linear_count] = j;
            linear_constraints->col_ind[*linear_count] = linear_offset + ir[k];
            linear_constraints->val[*linear_count] =
                read_typed_as_double(sp->data, t, k);
            (*linear_count)++;
          }
        }
      } else if (Atp->class_type == MAT_C_DOUBLE ||
                 Atp->class_type == MAT_C_SINGLE) {
        enum matio_types t = Atp->data_type;
        int rows = Atp->dims[0], cols = Atp->dims[1];
        for (int j = 0; j < cols; j++) {
          for (int row_idx = 0; row_idx < rows; row_idx++) {
            double v =
                read_typed_as_double(Atp->data, t, (size_t)j * rows + row_idx);
            if (v != 0.0) {
              linear_constraints->row_ind[*linear_count] = j;
              linear_constraints->col_ind[*linear_count] =
                  linear_offset + row_idx;
              linear_constraints->val[*linear_count] = v;
              (*linear_count)++;
            }
          }
        }
      }
      if (is_free)
        current_free_offset += linear_size;
      else
        current_lp_offset += linear_size;

    } else if (blk_type[0] == 's') {
      int *sub_dims = (int *)safe_malloc(num_sub_blocks * sizeof(int));
      int *sub_offsets = (int *)safe_malloc(num_sub_blocks * sizeof(int));
      int *sub_vector_offsets =
          (int *)safe_malloc(num_sub_blocks * sizeof(int));

      int offset = 0, vec_offset = 0;
      for (int i = 0; i < num_sub_blocks; i++) {
        sub_dims[i] = (int)get_scalar(size_cell, i);
        sub_offsets[i] = offset;
        sub_vector_offsets[i] = vec_offset;
        offset += sub_dims[i];
        vec_offset += (sub_dims[i] * (sub_dims[i] + 1)) / 2;
        sdp->blk_dims[current_cone_idx + i] = sub_dims[i];
      }

      if (Cp->class_type == MAT_C_SPARSE) {
        mat_sparse_t *sp = (mat_sparse_t *)Cp->data;
        int *ir = (int *)sp->ir, *jc = (int *)sp->jc;
        enum matio_types t = Cp->data_type;
        if (Cp->dims[0] == Cp->dims[1]) {
          for (int col = 0; col < offset; col++) {
            for (int k = jc[col]; k < jc[col + 1]; k++) {
              int row = ir[k];
              if (row >= col) {
                int sub_idx = 0;
                while (sub_idx < num_sub_blocks - 1 &&
                       row >= sub_offsets[sub_idx + 1])
                  sub_idx++;
                sdp->psd_cone_objective->cone_ind[o_count] =
                    current_cone_idx + sub_idx;
                sdp->psd_cone_objective->row_ind[o_count] =
                    row - sub_offsets[sub_idx];
                sdp->psd_cone_objective->col_ind[o_count] =
                    col - sub_offsets[sub_idx];
                sdp->psd_cone_objective->val[o_count] =
                    read_typed_as_double(sp->data, t, k);
                o_count++;
              }
            }
          }
        }
      } else if (Cp->class_type == MAT_C_DOUBLE ||
                 Cp->class_type == MAT_C_SINGLE) {
        if (Cp->dims[0] == Cp->dims[1]) {
          enum matio_types t = Cp->data_type;
          int dim = Cp->dims[0];
          for (int col = 0; col < dim; col++) {
            for (int row = col; row < dim; row++) {
              double v =
                  read_typed_as_double(Cp->data, t, (size_t)col * dim + row);
              if (v != 0.0) {
                int sub_idx = 0;
                while (sub_idx < num_sub_blocks - 1 &&
                       row >= sub_offsets[sub_idx + 1])
                  sub_idx++;
                sdp->psd_cone_objective->cone_ind[o_count] =
                    current_cone_idx + sub_idx;
                sdp->psd_cone_objective->row_ind[o_count] =
                    row - sub_offsets[sub_idx];
                sdp->psd_cone_objective->col_ind[o_count] =
                    col - sub_offsets[sub_idx];
                sdp->psd_cone_objective->val[o_count] = v;
                o_count++;
              }
            }
          }
        }
      }

      if (Atp->class_type == MAT_C_SPARSE) {
        mat_sparse_t *sp = (mat_sparse_t *)Atp->data;
        int *ir = (int *)sp->ir, *jc = (int *)sp->jc;
        enum matio_types t = Atp->data_type;
        int normal_cols = (int)Atp->dims[1];
        for (int j = 0; j < normal_cols; j++) {
          for (int k = jc[j]; k < jc[j + 1]; k++) {
            int row_idx = ir[k];
            int sub_idx = 0;
            while (sub_idx < num_sub_blocks - 1 &&
                   row_idx >= sub_vector_offsets[sub_idx + 1])
              sub_idx++;
            int local_vec_idx = row_idx - sub_vector_offsets[sub_idx];

            int c = 0, temp = local_vec_idx;
            while (temp >= c + 1) {
              temp -= (c + 1);
              c++;
            }
            int r = temp;

            int final_r = c;
            int final_c = r;

            double scale = (final_r == final_c) ? 1.0 : 0.7071067811865475244;

            sdp->psd_cone_constraints->constr_ind[c_count] = j;
            sdp->psd_cone_constraints->cone_ind[c_count] =
                current_cone_idx + sub_idx;
            sdp->psd_cone_constraints->row_ind[c_count] = final_r;
            sdp->psd_cone_constraints->col_ind[c_count] = final_c;
            sdp->psd_cone_constraints->val[c_count] =
                read_typed_as_double(sp->data, t, k) * scale;
            c_count++;
          }
        }
      } else if (Atp->class_type == MAT_C_DOUBLE ||
                 Atp->class_type == MAT_C_SINGLE) {
        enum matio_types t = Atp->data_type;
        int rows = Atp->dims[0], cols = Atp->dims[1];
        for (int j = 0; j < cols; j++) {
          for (int row_idx = 0; row_idx < rows; row_idx++) {
            double v =
                read_typed_as_double(Atp->data, t, (size_t)j * rows + row_idx);
            if (v != 0.0) {
              int sub_idx = 0;
              while (sub_idx < num_sub_blocks - 1 &&
                     row_idx >= sub_vector_offsets[sub_idx + 1])
                sub_idx++;
              int local_vec_idx = row_idx - sub_vector_offsets[sub_idx];

              int c = 0, temp = local_vec_idx;
              while (temp >= c + 1) {
                temp -= (c + 1);
                c++;
              }
              int r = temp;

              int final_r = c;
              int final_c = r;

              double scale = (final_r == final_c) ? 1.0 : 0.7071067811865475244;

              sdp->psd_cone_constraints->constr_ind[c_count] = j;
              sdp->psd_cone_constraints->cone_ind[c_count] =
                  current_cone_idx + sub_idx;
              sdp->psd_cone_constraints->row_ind[c_count] = final_r;
              sdp->psd_cone_constraints->col_ind[c_count] = final_c;
              sdp->psd_cone_constraints->val[c_count] = v * scale;
              c_count++;
            }
          }
        }
      }

      matvar_t *rank_cell =
          blk_var->dims[1] > 2 ? blk_cells[p + 2 * K_blocks] : NULL;
      if (matvar_numel(rank_cell) > 0) {
        int normal_constraints = Atp && Atp->rank >= 2 ? (int)Atp->dims[1] : 0;
        if (!append_sdpt3_low_rank(
                rank_cell, At_cells[p + K_blocks], At_cells[p + 2 * K_blocks],
                normal_constraints, current_cone_idx, sub_dims[0],
                sdp->low_rank_data, &low_rank_column, &low_rank_factor)) {
          free(sub_dims);
          free(sub_offsets);
          free(sub_vector_offsets);
          free_basic_sdp(sdp);
          if (!root_struct) {
            matvar_t *vars[] = {blk_var, At_var, C_var, b_var, options_var};
            free_matvars(vars, sizeof(vars) / sizeof(vars[0]));
          }
          return NULL;
        }
      }

      current_cone_idx += num_sub_blocks;
      free(sub_dims);
      free(sub_offsets);
      free(sub_vector_offsets);
    }
  }

  sdp->nnz_psd_constr = c_count;
  sdp->nnz_psd_obj = o_count;
  sdp->nnz_lp_constr = lp_c_count;
  sdp->nnz_lp_obj = total_lp_dim;
  sdp->nnz_free_constr = free_c_count;
  sdp->nnz_free_obj = total_free_dim;
  if (sdp->low_rank_data)
    sdp->low_rank_data->num_columns = low_rank_column;

  LOG_DBG("  -> SDPT3 parsed: m=%d, PSD cones=%d, LP dim=%d, free dim=%d\n",
          sdp->m, sdp->n_cones, sdp->lp_dim, sdp->free_dim);
  LOG_DBG("     PSD NNZ: A=%d, C=%d | LP NNZ: A=%d | Free NNZ: A=%d\n",
          sdp->nnz_psd_constr, sdp->nnz_psd_obj, sdp->nnz_lp_constr,
          sdp->nnz_free_constr);
  LOG_DBG("     Signed low-rank columns: %d\n", low_rank_column);

  if (!root_struct) {
    if (blk_var)
      Mat_VarFree(blk_var);
    if (At_var)
      Mat_VarFree(At_var);
    if (C_var)
      Mat_VarFree(C_var);
    if (b_var)
      Mat_VarFree(b_var);
    if (options_var)
      Mat_VarFree(options_var);
  }
  return sdp;
}

basic_sdp_t *read_mat_smart(const char *filename) {
  mat_t *matfp = Mat_Open(filename, MAT_ACC_RDONLY);
  if (!matfp) {
    fprintf(stderr, "Error: Cannot open MAT file %s\n", filename);
    return NULL;
  }

  matvar_t *blk_probe = Mat_VarReadInfo(matfp, "blk");
  matvar_t *K_probe = Mat_VarReadInfo(matfp, "K");

  if (blk_probe != NULL) {
    LOG_DBG("Detected 'blk' at root, engaging Flat SDPT3 parser...\n");
    Mat_VarFree(blk_probe);
    if (K_probe)
      Mat_VarFree(K_probe);
    basic_sdp_t *sdp = read_sdpt3_mat(matfp, NULL);
    Mat_Close(matfp);
    return sdp;
  } else if (K_probe != NULL) {
    LOG_DBG("Detected 'K' at root, engaging Flat SeDuMi parser...\n");
    Mat_VarFree(K_probe);
    basic_sdp_t *sdp = read_sedumi_mat(matfp, NULL);
    Mat_Close(matfp);
    return sdp;
  }

  Mat_Rewind(matfp);
  matvar_t *root_struct = NULL;

  while ((root_struct = Mat_VarReadNext(matfp)) != NULL) {
    if (root_struct->class_type == MAT_C_STRUCT) {

      if (Mat_VarGetStructFieldByName(root_struct, "blk", 0)) {
        LOG_DBG("Detected nested SDPT3 inside struct '%s'...\n",
                root_struct->name);
        basic_sdp_t *sdp = read_sdpt3_mat(matfp, root_struct);
        Mat_VarFree(root_struct);
        Mat_Close(matfp);
        return sdp;

      } else if (Mat_VarGetStructFieldByName(root_struct, "K", 0)) {
        LOG_DBG("Detected nested SeDuMi inside struct '%s'...\n",
                root_struct->name);
        basic_sdp_t *sdp = read_sedumi_mat(matfp, root_struct);
        Mat_VarFree(root_struct);
        Mat_Close(matfp);
        return sdp;
      }
    }
    Mat_VarFree(root_struct);
  }

  fprintf(stderr,
          "Error: MAT file is neither valid SDPT3 nor SeDuMi format.\n");
  Mat_Close(matfp);
  return NULL;
}

basic_sdp_t *handle_mat_from_memory(const char *original_filename, char *data,
                                    size_t size) {
  basic_sdp_t *sdp = NULL;

  size_t len = strlen(original_filename);
  bool is_compressed =
      (len > 3 && strcmp(original_filename + len - 3, ".gz") == 0);

  if (!is_compressed) {
    return read_mat_smart(original_filename);
  } else {
    char tmp_path[] = "/tmp/sdp_solver_XXXXXX.mat";
    int fd = mkstemps(tmp_path, 4);
    if (fd == -1)
      return NULL;

    if (write(fd, data, size) != (ssize_t)size) {
      fprintf(stderr, "Error: Failed to write temporary MAT file.\n");
    }
    close(fd);

    sdp = read_mat_smart(tmp_path);

    unlink(tmp_path);
    return sdp;
  }
}

#else

basic_sdp_t *read_mat_smart(const char *filename) {
  (void)filename;
  fprintf(stderr, "Error: .mat parsing is not supported in this build.\n");
  fprintf(stderr, "Please install libmatio-dev and recompile with CMake flag "
                  "-DENABLE_MATIO=ON.\n");
  return NULL;
}
basic_sdp_t *handle_mat_from_memory(const char *original_filename, char *data,
                                    size_t size) {
  (void)original_filename;
  (void)data;
  (void)size;
  fprintf(stderr, "Error: .mat parsing is not supported in this build.\n");
  fprintf(stderr, "Please install libmatio-dev and recompile with CMake flag "
                  "-DENABLE_MATIO=ON.\n");
  return NULL;
}
#endif
