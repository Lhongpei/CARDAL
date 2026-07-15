/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "sdp_types.h"
#include "utils.h"
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
  case MAT_T_DOUBLE: return ((const double *)base)[index];
  case MAT_T_SINGLE: return ((const float *)base)[index];
  case MAT_T_INT8:   return ((const signed char *)base)[index];
  case MAT_T_UINT8:  return ((const unsigned char *)base)[index];
  case MAT_T_INT16:  return ((const short *)base)[index];
  case MAT_T_UINT16: return ((const unsigned short *)base)[index];
  case MAT_T_INT32:  return ((const int *)base)[index];
  case MAT_T_UINT32: return ((const unsigned int *)base)[index];
  case MAT_T_INT64:  return (double)((const long long *)base)[index];
  case MAT_T_UINT64: return (double)((const unsigned long long *)base)[index];
  default:           return 0.0;
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
  matvar_t *K_s_var = Mat_VarGetStructFieldByName(K_var, "s", 0);

  int K_f = (int)get_scalar(K_f_var, 0);
  int K_l = (int)get_scalar(K_l_var, 0);

  int K_q_dim = 0;
  if (K_q_var && K_q_var->data) {
    int num_socp = (int)(K_q_var->dims[0] * K_q_var->dims[1]);
    for (int i = 0; i < num_socp; i++)
      K_q_dim += (int)get_scalar(K_q_var, i);
    if (K_q_dim > 0)
      LOG_DBG("[Warning] Ignoring %d SOCP constraints (K.q) - not supported by "
             "basic_sdp_t!\n",
             num_socp);
  }
  if (K_f > 0)
    LOG_DBG("[Warning] Ignoring %d Free variables (K.f) - not natively "
           "supported!\n",
           K_f);

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
                     read_typed_as_double(c_sp->data, c_var->data_type, i),
                     K_f, K_l, K_q_dim, n_cones, blk_dims, sdp, &o_count);
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

  int c_count = 0, lp_c_count = 0;

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

  LOG_DBG("  -> SeDuMi parsed: m=%d, PSD cones=%d, LP dim=%d\n", sdp->m,
         sdp->n_cones, sdp->lp_dim);
  LOG_DBG("     PSD NNZ: A=%d, C=%d | LP NNZ: A=%d\n", sdp->nnz_psd_constr,
         sdp->nnz_psd_obj, sdp->nnz_lp_constr);

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
  matvar_t *blk_var = NULL, *At_var = NULL, *C_var = NULL, *b_var = NULL;

  if (root_struct) {
    blk_var = Mat_VarGetStructFieldByName(root_struct, "blk", 0);
    At_var = Mat_VarGetStructFieldByName(root_struct, "At", 0);
    C_var = Mat_VarGetStructFieldByName(root_struct, "C", 0);
    b_var = Mat_VarGetStructFieldByName(root_struct, "b", 0);
  } else {
    blk_var = Mat_VarRead(matfp, "blk");
    At_var = Mat_VarRead(matfp, "At");
    C_var = Mat_VarRead(matfp, "C");
    b_var = Mat_VarRead(matfp, "b");
  }

  if (!blk_var || !At_var || !C_var || !b_var ||
      blk_var->class_type != MAT_C_CELL) {
    fprintf(stderr, "Fatal Error: Invalid SDPT3 format.\n");
    return NULL;
  }

  int K_blocks = blk_var->dims[0];
  matvar_t **blk_cells = (matvar_t **)blk_var->data;
  matvar_t **At_cells = (matvar_t **)At_var->data;
  matvar_t **C_cells = (matvar_t **)C_var->data;

  int m =
      (int)(b_var->dims[0] > b_var->dims[1] ? b_var->dims[0] : b_var->dims[1]);
  int total_n_cones = 0;
  int total_lp_dim = 0;

  for (int p = 0; p < K_blocks; p++) {
    matvar_t *type_cell = blk_cells[p];
    matvar_t *size_cell = blk_cells[p + K_blocks];
    char *blk_type = (char *)type_cell->data;
    int num_sub_blocks = size_cell->nbytes / size_cell->data_size;

    if (blk_type[0] == 's')
      total_n_cones += num_sub_blocks;
    else if (blk_type[0] == 'l') {
      for (int i = 0; i < num_sub_blocks; i++)
        total_lp_dim += (int)get_scalar(size_cell, i);
    }
  }

  basic_sdp_t *sdp = (basic_sdp_t *)calloc(1, sizeof(basic_sdp_t));
  sdp->m = m;
  sdp->n_cones = total_n_cones;
  sdp->lp_dim = total_lp_dim;
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

  int current_cone_idx = 0, current_lp_offset = 0;
  int c_count = 0, o_count = 0, lp_c_count = 0;

  for (int p = 0; p < K_blocks; p++) {
    matvar_t *type_cell = blk_cells[p], *size_cell = blk_cells[p + K_blocks];
    char *blk_type = (char *)type_cell->data;
    int num_sub_blocks = size_cell->nbytes / size_cell->data_size;
    matvar_t *Cp = C_cells[p], *Atp = At_cells[p];

    if (blk_type[0] == 'l') {
      int lp_size = 0;
      for (int i = 0; i < num_sub_blocks; i++)
        lp_size += (int)get_scalar(size_cell, i);

      if (Cp->class_type == MAT_C_SPARSE) {
        mat_sparse_t *sp = (mat_sparse_t *)Cp->data;
        for (int i = 0; i < (int)sp->ndata; i++)
          sdp->lp_objective[current_lp_offset + ((int *)sp->ir)[i]] =
              -read_typed_as_double(sp->data, Cp->data_type, i);
      } else if (Cp->class_type == MAT_C_DOUBLE ||
                 Cp->class_type == MAT_C_SINGLE) {
        for (int i = 0; i < lp_size; i++)
          sdp->lp_objective[current_lp_offset + i] =
              -read_typed_as_double(Cp->data, Cp->data_type, i);
      }

      if (Atp->class_type == MAT_C_SPARSE) {
        mat_sparse_t *sp = (mat_sparse_t *)Atp->data;
        int *ir = (int *)sp->ir, *jc = (int *)sp->jc;
        enum matio_types t = Atp->data_type;
        for (int j = 0; j < m; j++) {
          for (int k = jc[j]; k < jc[j + 1]; k++) {
            sdp->lp_constraints->row_ind[lp_c_count] = j;
            sdp->lp_constraints->col_ind[lp_c_count] =
                current_lp_offset + ir[k];
            sdp->lp_constraints->val[lp_c_count] =
                read_typed_as_double(sp->data, t, k);
            lp_c_count++;
          }
        }
      } else if (Atp->class_type == MAT_C_DOUBLE ||
                 Atp->class_type == MAT_C_SINGLE) {
        enum matio_types t = Atp->data_type;
        int rows = Atp->dims[0], cols = Atp->dims[1];
        for (int j = 0; j < cols; j++) {
          for (int row_idx = 0; row_idx < rows; row_idx++) {
            double v = read_typed_as_double(Atp->data, t,
                                            (size_t)j * rows + row_idx);
            if (v != 0.0) {
              sdp->lp_constraints->row_ind[lp_c_count] = j;
              sdp->lp_constraints->col_ind[lp_c_count] =
                  current_lp_offset + row_idx;
              sdp->lp_constraints->val[lp_c_count] = v;
              lp_c_count++;
            }
          }
        }
      }
      current_lp_offset += lp_size;

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
              double v = read_typed_as_double(Cp->data, t,
                                              (size_t)col * dim + row);
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
        for (int j = 0; j < m; j++) {
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
            double v = read_typed_as_double(Atp->data, t,
                                            (size_t)j * rows + row_idx);
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

  LOG_DBG("  -> SDPT3 parsed: m=%d, PSD cones=%d, LP dim=%d\n", sdp->m,
         sdp->n_cones, sdp->lp_dim);
  LOG_DBG("     PSD NNZ: A=%d, C=%d | LP NNZ: A=%d\n", sdp->nnz_psd_constr,
         sdp->nnz_psd_obj, sdp->nnz_lp_constr);

  if (!root_struct) {
    if (blk_var)
      Mat_VarFree(blk_var);
    if (At_var)
      Mat_VarFree(At_var);
    if (C_var)
      Mat_VarFree(C_var);
    if (b_var)
      Mat_VarFree(b_var);
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