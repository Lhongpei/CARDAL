/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "preconditioner.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SCALING_EPSILON 1e-12

static inline void decode_g_idx(long long g_idx, int n_blks,
                                const long long *blk_ptr, const int *blk_dims,
                                int *out_k, int *out_r, int *out_c) {
  int low = 0, high = n_blks - 1, k = 0;
  while (low <= high) {
    int mid = (low + high) / 2;
    if (g_idx < blk_ptr[mid])
      high = mid - 1;
    else if (g_idx >= blk_ptr[mid + 1])
      low = mid + 1;
    else {
      k = mid;
      break;
    }
  }
  long long local_idx = g_idx - blk_ptr[k];
  int n_k = blk_dims[k];
  *out_k = k;
  *out_c = (int)(local_idx / n_k);
  *out_r = (int)(local_idx % n_k);
}

static int *build_r_blk_ptr(int n_blks, const int *blk_dims,
                            int *out_total_r_dim) {
  int *r_blk_ptr = (int *)safe_malloc((n_blks + 1) * sizeof(int));
  int total = 0;
  r_blk_ptr[0] = 0;
  for (int i = 0; i < n_blks; i++) {
    total += blk_dims[i];
    r_blk_ptr[i + 1] = total;
  }
  *out_total_r_dim = total;
  return r_blk_ptr;
}

static sparse_csr_matrix_t *deepcopy_csr(const sparse_csr_matrix_t *mat) {
  if (!mat)
    return NULL;
  sparse_csr_matrix_t *copy =
      (sparse_csr_matrix_t *)safe_malloc(sizeof(sparse_csr_matrix_t));
  copy->num_rows = mat->num_rows;
  copy->num_cols = mat->num_cols;
  copy->num_nonzeros = mat->num_nonzeros;
  copy->row_ptr = (int *)safe_malloc((mat->num_rows + 1) * sizeof(int));
  memcpy(copy->row_ptr, mat->row_ptr, (mat->num_rows + 1) * sizeof(int));
  copy->col_ind = (int *)safe_malloc(mat->num_nonzeros * sizeof(int));
  memcpy(copy->col_ind, mat->col_ind, mat->num_nonzeros * sizeof(int));
  copy->val = (double *)safe_malloc(mat->num_nonzeros * sizeof(double));
  memcpy(copy->val, mat->val, mat->num_nonzeros * sizeof(double));
  return copy;
}

static void free_csr(sparse_csr_matrix_t *mat) {
  if (!mat)
    return;
  free(mat->row_ptr);
  free(mat->col_ind);
  free(mat->val);
  free(mat);
}

static compressed_sdp_problem_t *
deepcopy_sdp_problem(const compressed_sdp_problem_t *prob) {
  compressed_sdp_problem_t *out = (compressed_sdp_problem_t *)safe_calloc(
      1, sizeof(compressed_sdp_problem_t));

  // Scalars first.
  out->num_constraints = prob->num_constraints;
  out->n_blks = prob->n_blks;
  out->total_n_orig = prob->total_n_orig;
  out->n_active_vars = prob->n_active_vars;
  out->lp_dim = prob->lp_dim;
  out->lp_start_idx = prob->lp_start_idx;

  out->blk_dims = (int *)safe_malloc(prob->n_blks * sizeof(int));
  memcpy(out->blk_dims, prob->blk_dims, prob->n_blks * sizeof(int));
  out->blk_ptr = (long long *)safe_malloc((prob->n_blks + 1) * sizeof(long long));
  memcpy(out->blk_ptr, prob->blk_ptr,
         (prob->n_blks + 1) * sizeof(long long));

  if (prob->n_active_vars > 0) {
    out->col_mapping =
        (long long *)safe_malloc(prob->n_active_vars * sizeof(long long));
    memcpy(out->col_mapping, prob->col_mapping,
           prob->n_active_vars * sizeof(long long));
  }

  out->constraint_matrix = deepcopy_csr(prob->constraint_matrix);
  out->constraint_matrix_t = deepcopy_csr(prob->constraint_matrix_t);

  if (prob->objective_vector_sparse) {
    out->objective_vector_sparse =
        (sparse_vector_t *)safe_malloc(sizeof(sparse_vector_t));
    int len = prob->objective_vector_sparse->len;
    out->objective_vector_sparse->len = len;
    out->objective_vector_sparse->pos =
        (long long *)safe_malloc(len * sizeof(long long));
    out->objective_vector_sparse->val =
        (double *)safe_malloc(len * sizeof(double));
    memcpy(out->objective_vector_sparse->pos,
           prob->objective_vector_sparse->pos, len * sizeof(long long));
    memcpy(out->objective_vector_sparse->val,
           prob->objective_vector_sparse->val, len * sizeof(double));
  }

  {
    int n = prob->lp_dim > 0 ? prob->lp_dim : 1;
    out->lp_objective_vector = (double *)safe_calloc(n, sizeof(double));
    if (prob->lp_objective_vector)
      memcpy(out->lp_objective_vector, prob->lp_objective_vector,
             n * sizeof(double));
  }

  out->right_hand_side =
      (double *)safe_malloc(prob->num_constraints * sizeof(double));
  memcpy(out->right_hand_side, prob->right_hand_side,
         prob->num_constraints * sizeof(double));

  return out;
}

static void free_sdp_problem_clone(compressed_sdp_problem_t *prob) {
  if (!prob)
    return;
  free(prob->col_mapping);
  free(prob->blk_dims);
  free(prob->blk_ptr);
  free(prob->right_hand_side);
  free(prob->lp_objective_vector);
  free_csr(prob->constraint_matrix);
  free_csr(prob->constraint_matrix_t);
  if (prob->objective_vector_sparse) {
    free(prob->objective_vector_sparse->pos);
    free(prob->objective_vector_sparse->val);
    free(prob->objective_vector_sparse);
  }
  free(prob);
}

static void ruiz_rescaling_sdp(compressed_sdp_problem_t *prob, int num_iters,
                               rescale_info_t *info, int psd_scale_mode) {
  int m = prob->num_constraints;
  int lp_dim = prob->lp_dim;
  int lp_start_idx = prob->lp_start_idx;
  sparse_csr_matrix_t *A = prob->constraint_matrix;

  int total_r_dim = 0;
  int *r_blk_ptr = build_r_blk_ptr(prob->n_blks, prob->blk_dims, &total_r_dim);

  double *row_max = (double *)safe_malloc(m * sizeof(double));
  double *R_max = total_r_dim > 0
                      ? (double *)safe_malloc(total_r_dim * sizeof(double))
                      : NULL;
  double *lp_max =
      lp_dim > 0 ? (double *)safe_malloc(lp_dim * sizeof(double)) : NULL;

  for (int iter = 0; iter < num_iters; ++iter) {
    for (int i = 0; i < m; i++)
      row_max[i] = 0.0;
    for (int i = 0; i < total_r_dim; i++)
      R_max[i] = 0.0;
    for (int i = 0; i < lp_dim; i++)
      lp_max[i] = 0.0;

    for (int i = 0; i < m; i++) {
      for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
        int compact_col = A->col_ind[p];
        double val = fabs(A->val[p]);

        if (val > row_max[i])
          row_max[i] = val;

        if (compact_col < lp_start_idx) {
          long long g_idx = prob->col_mapping[compact_col];
          int k, r, c;
          decode_g_idx(g_idx, prob->n_blks, prob->blk_ptr, prob->blk_dims, &k,
                       &r, &c);
          int slot = r_blk_ptr[k] + r;
          if (val > R_max[slot])
            R_max[slot] = val;
        } else {
          int lp_idx = compact_col - lp_start_idx;
          if (val > lp_max[lp_idx])
            lp_max[lp_idx] = val;
        }
      }
    }

    for (int i = 0; i < m; i++)
      row_max[i] =
          (row_max[i] < SCALING_EPSILON) ? 1.0 : sqrt(row_max[i]);
    if (psd_scale_mode == PSD_SCALE_MODE_PER_CONE) {
      // Per-cone uniform: collapse R_max within each cone to a single value.
      for (int k = 0; k < prob->n_blks; k++) {
        int start = r_blk_ptr[k];
        int end = start + prob->blk_dims[k];
        double mx = 0.0;
        for (int i = start; i < end; i++) if (R_max[i] > mx) mx = R_max[i];
        double s = (mx < SCALING_EPSILON) ? 1.0 : sqrt(mx);
        for (int i = start; i < end; i++) R_max[i] = s;
      }
    } else {
      for (int i = 0; i < total_r_dim; i++)
        R_max[i] = (R_max[i] < SCALING_EPSILON) ? 1.0 : sqrt(R_max[i]);
    }
    for (int i = 0; i < lp_dim; i++)
      lp_max[i] = (lp_max[i] < SCALING_EPSILON) ? 1.0 : sqrt(lp_max[i]);

    for (int i = 0; i < m; i++)
      prob->right_hand_side[i] /= row_max[i];

    for (int i = 0; i < m; i++) {
      for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
        int compact_col = A->col_ind[p];
        if (compact_col < lp_start_idx) {
          long long g_idx = prob->col_mapping[compact_col];
          int k, r, c;
          decode_g_idx(g_idx, prob->n_blks, prob->blk_ptr, prob->blk_dims, &k,
                       &r, &c);
          A->val[p] /=
              (row_max[i] * R_max[r_blk_ptr[k] + r] * R_max[r_blk_ptr[k] + c]);
        } else {
          int lp_idx = compact_col - lp_start_idx;
          A->val[p] /= (row_max[i] * lp_max[lp_idx]);
        }
      }
    }

    if (prob->objective_vector_sparse) {
      for (int p = 0; p < prob->objective_vector_sparse->len; p++) {
        long long g_idx = prob->objective_vector_sparse->pos[p];
        int k, r, c;
        decode_g_idx(g_idx, prob->n_blks, prob->blk_ptr, prob->blk_dims, &k,
                     &r, &c);
        prob->objective_vector_sparse->val[p] /=
            (R_max[r_blk_ptr[k] + r] * R_max[r_blk_ptr[k] + c]);
      }
    }
    for (int i = 0; i < lp_dim; i++)
      prob->lp_objective_vector[i] /= lp_max[i];

    for (int i = 0; i < m; i++)
      info->constraint_rescaling[i] *= row_max[i];
    for (int i = 0; i < total_r_dim; i++)
      info->psd_cone_rescaling[i] *= R_max[i];
    for (int i = 0; i < lp_dim; i++)
      info->lp_variable_rescaling[i] *= lp_max[i];
  }

  free(row_max);
  free(R_max);
  free(lp_max);
  free(r_blk_ptr);
}

static void pock_chambolle_rescaling_sdp(compressed_sdp_problem_t *prob,
                                         double alpha, rescale_info_t *info,
                                         int psd_scale_mode) {
  int m = prob->num_constraints;
  int lp_dim = prob->lp_dim;
  int lp_start_idx = prob->lp_start_idx;
  sparse_csr_matrix_t *A = prob->constraint_matrix;

  int total_r_dim = 0;
  int *r_blk_ptr = build_r_blk_ptr(prob->n_blks, prob->blk_dims, &total_r_dim);

  double *row_sum = (double *)safe_calloc((size_t)m, sizeof(double));
  double *R_sum = total_r_dim > 0
                      ? (double *)safe_calloc((size_t)total_r_dim,
                                              sizeof(double))
                      : NULL;
  double *lp_sum = lp_dim > 0
                       ? (double *)safe_calloc((size_t)lp_dim, sizeof(double))
                       : NULL;

  double *row_max_buf = total_r_dim > 0
                            ? (double *)safe_calloc((size_t)total_r_dim,
                                                    sizeof(double))
                            : NULL;
  int *touched_idx = total_r_dim > 0
                         ? (int *)safe_malloc((size_t)total_r_dim *
                                              sizeof(int))
                         : NULL;

  for (int i = 0; i < m; i++) {
    int touched_count = 0;
    for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
      int compact_col = A->col_ind[p];
      double val = fabs(A->val[p]);
      row_sum[i] += pow(val, alpha);

      if (compact_col < lp_start_idx) {
        if (val == 0.0)
          continue;
        long long g_idx = prob->col_mapping[compact_col];
        int k, r, c;
        decode_g_idx(g_idx, prob->n_blks, prob->blk_ptr, prob->blk_dims, &k,
                     &r, &c);
        int r_idx = r_blk_ptr[k] + r;
        if (row_max_buf[r_idx] == 0.0)
          touched_idx[touched_count++] = r_idx;
        if (val > row_max_buf[r_idx])
          row_max_buf[r_idx] = val;
      } else {
        int lp_idx = compact_col - lp_start_idx;
        lp_sum[lp_idx] += pow(val, 2.0 - alpha);
      }
    }
    for (int t = 0; t < touched_count; t++) {
      int r_idx = touched_idx[t];
      R_sum[r_idx] += pow(row_max_buf[r_idx], 2.0 - alpha);
      row_max_buf[r_idx] = 0.0;
    }
  }
  free(row_max_buf);
  free(touched_idx);

  for (int i = 0; i < m; i++)
    row_sum[i] =
        (row_sum[i] < SCALING_EPSILON) ? 1.0 : sqrt(row_sum[i]);
  if (psd_scale_mode == PSD_SCALE_MODE_PER_CONE) {
    // Per-cone uniform PC: collapse R_sum within each cone to a single value
    // (max across the cone's rows -> shared scalar). Preserves cone PSD.
    for (int k = 0; k < prob->n_blks; k++) {
      int start = r_blk_ptr[k];
      int end = start + prob->blk_dims[k];
      double mx = 0.0;
      for (int i = start; i < end; i++) if (R_sum[i] > mx) mx = R_sum[i];
      double s = (mx < SCALING_EPSILON) ? 1.0 : pow(mx, 0.25);
      for (int i = start; i < end; i++) R_sum[i] = s;
    }
  } else {
    for (int i = 0; i < total_r_dim; i++)
      R_sum[i] = (R_sum[i] < SCALING_EPSILON) ? 1.0 : pow(R_sum[i], 0.25);
  }
  for (int i = 0; i < lp_dim; i++)
    lp_sum[i] = (lp_sum[i] < SCALING_EPSILON) ? 1.0 : sqrt(lp_sum[i]);

  // Cap CUMULATIVE (Ruiz * PC) diagonal scaling so max <= pc_max_diag.
  // Uniform scalar division preserves relative directions.
  const double pc_max_diag = 1000.0;
  if (pc_max_diag > 0.0) {
    double row_total_max = 0.0;
    for (int i = 0; i < m; i++) {
      double v = info->constraint_rescaling[i] * row_sum[i];
      if (v > row_total_max) row_total_max = v;
    }
    if (row_total_max > pc_max_diag) {
      double s = pc_max_diag / row_total_max;
      for (int i = 0; i < m; i++) row_sum[i] *= s;
    }
    double R_total_max = 0.0;
    for (int i = 0; i < total_r_dim; i++) {
      double v = info->psd_cone_rescaling[i] * R_sum[i];
      if (v > R_total_max) R_total_max = v;
    }
    if (R_total_max > pc_max_diag) {
      double s = pc_max_diag / R_total_max;
      for (int i = 0; i < total_r_dim; i++) R_sum[i] *= s;
    }
    double lp_total_max = 0.0;
    for (int i = 0; i < lp_dim; i++) {
      double v = info->lp_variable_rescaling[i] * lp_sum[i];
      if (v > lp_total_max) lp_total_max = v;
    }
    if (lp_total_max > pc_max_diag) {
      double s = pc_max_diag / lp_total_max;
      for (int i = 0; i < lp_dim; i++) lp_sum[i] *= s;
    }
  }

  for (int i = 0; i < m; i++)
    prob->right_hand_side[i] /= row_sum[i];

  for (int i = 0; i < m; i++) {
    for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
      int compact_col = A->col_ind[p];
      if (compact_col < lp_start_idx) {
        long long g_idx = prob->col_mapping[compact_col];
        int k, r, c;
        decode_g_idx(g_idx, prob->n_blks, prob->blk_ptr, prob->blk_dims, &k,
                     &r, &c);
        A->val[p] /=
            (row_sum[i] * R_sum[r_blk_ptr[k] + r] * R_sum[r_blk_ptr[k] + c]);
      } else {
        int lp_idx = compact_col - lp_start_idx;
        A->val[p] /= (row_sum[i] * lp_sum[lp_idx]);
      }
    }
  }

  if (prob->objective_vector_sparse) {
    for (int p = 0; p < prob->objective_vector_sparse->len; p++) {
      long long g_idx = prob->objective_vector_sparse->pos[p];
      int k, r, c;
      decode_g_idx(g_idx, prob->n_blks, prob->blk_ptr, prob->blk_dims, &k, &r,
                   &c);
      prob->objective_vector_sparse->val[p] /=
          (R_sum[r_blk_ptr[k] + r] * R_sum[r_blk_ptr[k] + c]);
    }
  }
  for (int i = 0; i < lp_dim; i++)
    prob->lp_objective_vector[i] /= lp_sum[i];

  for (int i = 0; i < m; i++)
    info->constraint_rescaling[i] *= row_sum[i];
  for (int i = 0; i < total_r_dim; i++)
    info->psd_cone_rescaling[i] *= R_sum[i];
  for (int i = 0; i < lp_dim; i++)
    info->lp_variable_rescaling[i] *= lp_sum[i];

  free(row_sum);
  free(R_sum);
  free(lp_sum);
  free(r_blk_ptr);
}

static void bound_obj_rescaling_sdp(compressed_sdp_problem_t *prob,
                                     rescale_info_t *info) {
  double b_norm_sq = 0.0;
  for (int i = 0; i < prob->num_constraints; ++i) {
    b_norm_sq += prob->right_hand_side[i] * prob->right_hand_side[i];
  }

  double c_norm_sq = 0.0;
  if (prob->objective_vector_sparse) {
    for (int p = 0; p < prob->objective_vector_sparse->len; p++) {
      c_norm_sq += prob->objective_vector_sparse->val[p] *
                   prob->objective_vector_sparse->val[p];
    }
  }
  for (int i = 0; i < prob->lp_dim; ++i) {
    c_norm_sq +=
        prob->lp_objective_vector[i] * prob->lp_objective_vector[i];
  }

  info->right_hand_side_rescaling = 1.0 / (sqrt(b_norm_sq) + 1.0);
  info->objective_vector_rescaling = 1.0 / (sqrt(c_norm_sq) + 1.0);

  for (int i = 0; i < prob->num_constraints; ++i)
    prob->right_hand_side[i] *= info->right_hand_side_rescaling;

  if (prob->objective_vector_sparse) {
    for (int p = 0; p < prob->objective_vector_sparse->len; p++) {
      prob->objective_vector_sparse->val[p] *= info->objective_vector_rescaling;
    }
  }
  for (int i = 0; i < prob->lp_dim; ++i)
    prob->lp_objective_vector[i] *= info->objective_vector_rescaling;
}

rescale_info_t *rescale_problem(const cardal_parameters_t *params,
                                const compressed_sdp_problem_t *original_problem) {
  if (original_problem == NULL || params == NULL)
    return NULL;

  int scaling_enabled = params->l_inf_ruiz_iterations > 0 ||
                        params->has_pock_chambolle_alpha ||
                        params->bound_objective_rescaling;
  if (!scaling_enabled)
    return NULL;

  clock_t t0 = clock();

  rescale_info_t *info =
      (rescale_info_t *)safe_calloc(1, sizeof(rescale_info_t));
  info->scaled_problem = deepcopy_sdp_problem(original_problem);

  int m = original_problem->num_constraints;
  int lp_dim = original_problem->lp_dim;

  {
    double nb = 0.0;
    for (int i = 0; i < m; i++)
      nb += fabs(original_problem->right_hand_side[i]);
    info->unscaled_right_hand_side_norm = nb;

    double nc = 0.0;
    if (original_problem->objective_vector_sparse) {
      int n_C = original_problem->objective_vector_sparse->len;
      for (int p = 0; p < n_C; p++)
        nc += fabs(original_problem->objective_vector_sparse->val[p]);
    }
    if (original_problem->lp_dim > 0 &&
        original_problem->lp_objective_vector) {
      for (int i = 0; i < original_problem->lp_dim; i++)
        nc += fabs(original_problem->lp_objective_vector[i]);
    }
    info->unscaled_objective_vector_norm = nc;
  }

  int total_r_dim = 0;
  int *tmp_r_blk = build_r_blk_ptr(original_problem->n_blks,
                                   original_problem->blk_dims, &total_r_dim);
  free(tmp_r_blk);

  info->constraint_rescaling = (double *)safe_malloc((size_t)m * sizeof(double));
  info->psd_cone_rescaling =
      total_r_dim > 0
          ? (double *)safe_malloc((size_t)total_r_dim * sizeof(double))
          : NULL;
  info->lp_variable_rescaling =
      lp_dim > 0 ? (double *)safe_malloc((size_t)lp_dim * sizeof(double))
                  : NULL;

  for (int i = 0; i < m; ++i)
    info->constraint_rescaling[i] = 1.0;
  for (int i = 0; i < total_r_dim; ++i)
    info->psd_cone_rescaling[i] = 1.0;
  for (int i = 0; i < lp_dim; ++i)
    info->lp_variable_rescaling[i] = 1.0;

  info->objective_vector_rescaling = 1.0;
  info->right_hand_side_rescaling = 1.0;

  if (params->l_inf_ruiz_iterations > 0)
    ruiz_rescaling_sdp(info->scaled_problem, params->l_inf_ruiz_iterations,
                       info, params->psd_scale_mode);

  double *ruiz_only_constraint = NULL;
  double *ruiz_only_psd = NULL;
  if (params->verbose >= 3) {
    // Snapshot Ruiz-only scaling for debug diagnostics.
    ruiz_only_constraint =
        (double *)safe_malloc((size_t)m * sizeof(double));
    ruiz_only_psd =
        total_r_dim > 0
            ? (double *)safe_malloc((size_t)total_r_dim * sizeof(double))
            : NULL;
    for (int i = 0; i < m; i++)
      ruiz_only_constraint[i] = info->constraint_rescaling[i];
    for (int i = 0; i < total_r_dim; i++)
      ruiz_only_psd[i] = info->psd_cone_rescaling[i];
  }

  // Optionally compute "PC-before-Ruiz" stats by running PC on a deep copy of
  // the original problem (does not affect the actual scaling pipeline).
  if (params->has_pock_chambolle_alpha && params->verbose >= 3) {
    rescale_info_t *probe = (rescale_info_t *)safe_calloc(1, sizeof(rescale_info_t));
    probe->scaled_problem = deepcopy_sdp_problem(original_problem);
    probe->constraint_rescaling = (double *)safe_calloc((size_t)m, sizeof(double));
    probe->psd_cone_rescaling = total_r_dim > 0
        ? (double *)safe_calloc((size_t)total_r_dim, sizeof(double)) : NULL;
    probe->lp_variable_rescaling = lp_dim > 0
        ? (double *)safe_calloc((size_t)lp_dim, sizeof(double)) : NULL;
    for (int i = 0; i < m; i++) probe->constraint_rescaling[i] = 1.0;
    for (int i = 0; i < total_r_dim; i++) probe->psd_cone_rescaling[i] = 1.0;
    for (int i = 0; i < lp_dim; i++) probe->lp_variable_rescaling[i] = 1.0;
    pock_chambolle_rescaling_sdp(probe->scaled_problem,
                                 params->pock_chambolle_alpha, probe,
                                 params->psd_scale_mode);
    double rmin = 1e300, rmax = 0.0;
    for (int i = 0; i < m; i++) {
      double v = probe->constraint_rescaling[i];
      if (v < rmin) rmin = v;
      if (v > rmax) rmax = v;
    }
    double pmin = 1e300, pmax = 0.0;
    for (int i = 0; i < total_r_dim; i++) {
      double v = probe->psd_cone_rescaling[i];
      if (v < pmin) pmin = v;
      if (v > pmax) pmax = v;
    }
    printf("  [PC-on-ORIG] constraint_rescaling: min=%.3e max=%.3e\n", rmin, rmax);
    printf("  [PC-on-ORIG] psd_cone_rescaling  : min=%.3e max=%.3e\n", pmin, pmax);
    free_rescale_info(probe);
  }

  // Print Ruiz-only stats.
  if (params->verbose >= 3 && params->l_inf_ruiz_iterations > 0) {
    double rmin = 1e300, rmax = 0.0;
    for (int i = 0; i < m; i++) {
      double v = ruiz_only_constraint[i];
      if (v < rmin) rmin = v;
      if (v > rmax) rmax = v;
    }
    double pmin = 1e300, pmax = 0.0;
    for (int i = 0; i < total_r_dim; i++) {
      double v = ruiz_only_psd[i];
      if (v < pmin) pmin = v;
      if (v > pmax) pmax = v;
    }
    printf("  [Ruiz-only] constraint_rescaling: min=%.3e max=%.3e\n", rmin, rmax);
    printf("  [Ruiz-only] psd_cone_rescaling  : min=%.3e max=%.3e\n", pmin, pmax);
  }

  if (params->has_pock_chambolle_alpha && params->pock_chambolle_alpha > 0.0) {
    pock_chambolle_rescaling_sdp(info->scaled_problem,
                                 params->pock_chambolle_alpha, info,
                                 params->psd_scale_mode);
    // Diagnostic: report the diagonal scaling factor statistics after PC.
    if (params->verbose >= 3) {
      double rmin = 1e300, rmax = 0.0, rsum = 0.0;
      for (int i = 0; i < m; i++) {
        double v = info->constraint_rescaling[i];
        if (v < rmin) rmin = v;
        if (v > rmax) rmax = v;
        rsum += v;
      }
      double pmin = 1e300, pmax = 0.0, psum = 0.0;
      int pn = total_r_dim;
      for (int i = 0; i < pn; i++) {
        double v = info->psd_cone_rescaling[i];
        if (v < pmin) pmin = v;
        if (v > pmax) pmax = v;
        psum += v;
      }
      // info->constraint_rescaling[i] is now cumulative (Ruiz * PC).
      printf("  [cumulative] constraint_rescaling: min=%.3e max=%.3e\n",
             rmin, rmax);
      printf("  [cumulative] psd_cone_rescaling  : min=%.3e max=%.3e\n",
             pmin, pmax);

      // PC-after-Ruiz alone = cumulative / Ruiz_only.
      double pcr_min = 1e300, pcr_max = 0.0;
      for (int i = 0; i < m; i++) {
        double rz = ruiz_only_constraint[i];
        double pc = (rz > 0) ? info->constraint_rescaling[i] / rz : 0.0;
        if (pc < pcr_min) pcr_min = pc;
        if (pc > pcr_max) pcr_max = pc;
      }
      double pcp_min = 1e300, pcp_max = 0.0;
      for (int i = 0; i < pn; i++) {
        double rz = ruiz_only_psd[i];
        double pc = (rz > 0) ? info->psd_cone_rescaling[i] / rz : 0.0;
        if (pc < pcp_min) pcp_min = pc;
        if (pc > pcp_max) pcp_max = pc;
      }
      printf("  [PC-after-Ruiz] constraint: min=%.3e max=%.3e\n", pcr_min, pcr_max);
      printf("  [PC-after-Ruiz] psd_cone  : min=%.3e max=%.3e\n", pcp_min, pcp_max);
    }
  }
  free(ruiz_only_constraint);
  free(ruiz_only_psd);
  if (params->bound_objective_rescaling)
    bound_obj_rescaling_sdp(info->scaled_problem, info);

  info->rescaling_time_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
  return info;
}

void free_rescale_info(rescale_info_t *info) {
  if (!info)
    return;
  free_sdp_problem_clone(info->scaled_problem);
  free(info->constraint_rescaling);
  free(info->psd_cone_rescaling);
  free(info->lp_variable_rescaling);
  free(info);
}
