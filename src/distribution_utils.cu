/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "distribution_utils.h"
#include "internal_types.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <limits.h>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
ncclComm_t init_nccl(MPI_Comm mpi_comm) {
  ncclUniqueId id;
  ncclComm_t nccl_comm;
  int rank, nranks;

  MPI_Comm_rank(mpi_comm, &rank);
  MPI_Comm_size(mpi_comm, &nranks);

  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }

  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);
  NCCL_CHECK(ncclCommInitRank(&nccl_comm, nranks, id, rank));

  return nccl_comm;
}

grid_context_t initialize_parallel_context(int P_row, int P_rank, int P_cone) {
  grid_context_t grid;
  int initialized;

  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(NULL, NULL);
  }

  if (P_cone < 1)
    P_cone = 1;
  if (P_rank < 1)
    P_rank = 1;
  if (P_row < 1)
    P_row = 1;

  grid.comm_global = MPI_COMM_WORLD;
  MPI_Comm_rank(grid.comm_global, &grid.rank_global);

  grid.dims[0] = P_row;
  grid.dims[1] = P_rank;
  grid.dims[2] = P_cone;

  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  int local_device_id = grid.rank_global % num_devices;
  CUDA_CHECK(cudaSetDevice(local_device_id));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, local_device_id));

  LOG_DBG("[MPI Rank %d] Bound to GPU %d: %s (Total GPUs on node: %d, VRAM: "
         "%.1f GB)\n",
         grid.rank_global, local_device_id, prop.name, num_devices,
         (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  fflush(stdout);

  int my_cone = grid.rank_global % P_cone;
  int rr = grid.rank_global / P_cone;
  int my_row = rr / P_rank;
  int my_rank = rr % P_rank;

  grid.coords[0] = my_row;
  grid.coords[1] = my_rank;
  grid.coords[2] = my_cone;

  int row_color = my_rank * P_cone + my_cone;
  int rank_color = my_row * P_cone + my_cone;
  int cone_color = my_row * P_rank + my_rank;
  MPI_Comm_split(grid.comm_global, row_color, grid.rank_global, &grid.comm_row);
  MPI_Comm_split(grid.comm_global, rank_color, grid.rank_global, &grid.comm_rank);
  MPI_Comm_split(grid.comm_global, cone_color, grid.rank_global, &grid.comm_cone);
  grid.nccl_row = init_nccl(grid.comm_row);
  grid.nccl_rank = init_nccl(grid.comm_rank);
  grid.nccl_cone = init_nccl(grid.comm_cone);

  return grid;
}

static const long long *g_cmp_cost_ptr = NULL;
static int cmp_idx_by_cost_desc(const void *a, const void *b) {
  int ia = *(const int *)a, ib = *(const int *)b;
  long long ca = g_cmp_cost_ptr[ia], cb = g_cmp_cost_ptr[ib];
  return (ca < cb) - (ca > cb);
}

static const unsigned long long *g_locality_keys = NULL;
static int cmp_by_locality_key(const void *a, const void *b) {
  int ia = *(const int *)a, ib = *(const int *)b;
  unsigned long long ka = g_locality_keys[ia], kb = g_locality_keys[ib];
  return (ka > kb) - (ka < kb);
}

static void compute_nnz_balanced_row_range(
    const compressed_sdp_problem_t *global_prob, int P_row, int my_row,
    int *out_start, int *out_end, int *out_local_m) {
  int m_total = global_prob->num_constraints;
  long long total_nnz_global =
      global_prob->constraint_matrix
          ? (long long)global_prob->constraint_matrix->num_nonzeros
          : 0;
  if (P_row <= 1 || total_nnz_global == 0) {
    *out_start = 0;
    *out_end = m_total;
    *out_local_m = m_total;
    return;
  }
  const int *gptr = global_prob->constraint_matrix->row_ptr;
  long long target_lo = total_nnz_global * (long long)my_row / P_row;
  long long target_hi = total_nnz_global * (long long)(my_row + 1) / P_row;
  long long base = gptr[0];
  int lo = 0, hi = m_total;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (gptr[mid] - base < target_lo) lo = mid + 1;
    else hi = mid;
  }
  int start_row = lo;
  lo = start_row;
  hi = m_total;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (gptr[mid] - base < target_hi) lo = mid + 1;
    else hi = mid;
  }
  int end_row = (my_row == P_row - 1) ? m_total : lo;
  if (end_row < start_row) end_row = start_row;
  *out_start = start_row;
  *out_end = end_row;
  *out_local_m = end_row - start_row;
}

static void compute_cone_assignment(
    const compressed_sdp_problem_t *prob, int P_cone,
    int *owner_per_blk, int *lp_owner_out) {
  int n_blks = prob->n_blks;
  if (lp_owner_out) *lp_owner_out = 0;
  if (P_cone <= 1) {
    for (int b = 0; b < n_blks; b++) owner_per_blk[b] = 0;
    return;
  }

  // ---- m_b per block: count distinct rows of A touching each block ----
  int n_act = prob->n_active_vars;
  int *var_to_blk = (int *)safe_malloc((n_act > 0 ? n_act : 1) * sizeof(int));
  int b_cursor = 0;
  for (int i = 0; i < n_act; i++) {
    long long g = prob->col_mapping[i];
    if (g >= prob->total_n_orig) { var_to_blk[i] = -1; continue; }
    while (b_cursor < n_blks && g >= prob->blk_ptr[b_cursor + 1])
      b_cursor++;
    var_to_blk[i] = (b_cursor < n_blks) ? b_cursor : -1;
  }
  int *m_per_blk = (int *)safe_calloc(n_blks > 0 ? n_blks : 1, sizeof(int));
  int *stamp    = (int *)safe_calloc(n_blks > 0 ? n_blks : 1, sizeof(int));
  int cur_stamp = 0;
  const sparse_csr_matrix_t *A = prob->constraint_matrix;
  for (int r = 0; r < A->num_rows; r++) {
    cur_stamp++;
    for (int p = A->row_ptr[r]; p < A->row_ptr[r + 1]; p++) {
      int c = A->col_ind[p];
      if (c < 0 || c >= n_act) continue;
      int b = var_to_blk[c];
      if (b < 0) continue;
      if (stamp[b] != cur_stamp) { stamp[b] = cur_stamp; m_per_blk[b]++; }
    }
  }
  free(stamp);
  free(var_to_blk);

  long long *cost_b =
      (long long *)safe_malloc((n_blks > 0 ? n_blks : 1) * sizeof(long long));
  for (int b = 0; b < n_blks; b++) {
    int n = prob->blk_dims[b];
    int r_pataki = compute_theoretical_max_rank(m_per_blk[b]);
    int r = (n < r_pataki) ? n : r_pataki;
    cost_b[b] = (long long)n * n * r;
  }
  free(m_per_blk);

  int small_t = CARDAL_SMALL_CONE_DIM_THRESHOLD;
  int min_batch = CARDAL_MIN_BATCH_SIZE;

  int item_cap = n_blks + P_cone + 1;
  int *item_blk_starts = (int *)safe_malloc((item_cap + 1) * sizeof(int));
  int *item_blks =
      (int *)safe_malloc((n_blks > 0 ? n_blks : 1) * sizeof(int));
  long long *item_cost =
      (long long *)safe_malloc((item_cap > 0 ? item_cap : 1) * sizeof(long long));
  int *item_is_lp = (int *)safe_calloc(item_cap > 0 ? item_cap : 1, sizeof(int));
  int *item_is_batch =
      (int *)safe_calloc(item_cap > 0 ? item_cap : 1, sizeof(int));
  int n_items = 0;
  int blk_w = 0;
  item_blk_starts[0] = 0;

  int b = 0;
  while (b < n_blks) {
    int n_b = prob->blk_dims[b];
    if (n_b > small_t) {
      item_blks[blk_w++] = b;
      item_cost[n_items] = cost_b[b];
      n_items++;
      item_blk_starts[n_items] = blk_w;
      b++;
    } else {
      int g_start = b;
      while (b < n_blks && prob->blk_dims[b] == n_b)
        b++;
      int g_end = b;
      int N = g_end - g_start;
      int K_cap = N / min_batch;
      if (K_cap < 1) K_cap = 1;
      int K = (P_cone < K_cap) ? P_cone : K_cap;
      if (K < 1) K = 1;
      if (K > N) K = N;
      int chunk_base = N / K;
      int chunk_rem  = N % K;
      int cur = g_start;
      for (int k = 0; k < K; k++) {
        int chunk = chunk_base + (k < chunk_rem ? 1 : 0);
        long long chunk_cost = 0;
        for (int j = 0; j < chunk; j++) {
          item_blks[blk_w++] = cur + j;
          chunk_cost += cost_b[cur + j];
        }
        item_cost[n_items] = chunk_cost;
        item_is_batch[n_items] = 1;
        n_items++;
        item_blk_starts[n_items] = blk_w;
        cur += chunk;
      }
    }
  }

  if (prob->lp_dim > 0) {
    item_cost[n_items] = (long long)prob->lp_dim;
    item_is_lp[n_items] = 1;
    n_items++;
    item_blk_starts[n_items] = blk_w;
  }

  free(cost_b);

  int *order = (int *)safe_malloc((n_items > 0 ? n_items : 1) * sizeof(int));
  for (int i = 0; i < n_items; i++) order[i] = i;
  g_cmp_cost_ptr = item_cost;
  qsort(order, n_items, sizeof(int), cmp_idx_by_cost_desc);
  g_cmp_cost_ptr = NULL;

  long long *rank_load = (long long *)safe_calloc(P_cone, sizeof(long long));
  long long *batch_load = (long long *)safe_calloc(P_cone, sizeof(long long));
  int *item_owner = (int *)safe_malloc((n_items > 0 ? n_items : 1) * sizeof(int));
  for (int i = 0; i < n_items; i++) {
    int it = order[i];
    if (item_is_batch[it]) continue;
    int min_r = 0;
    for (int r = 1; r < P_cone; r++)
      if (rank_load[r] < rank_load[min_r]) min_r = r;
    item_owner[it] = min_r;
    rank_load[min_r] += item_cost[it];
  }

  for (int i = 0; i < n_items; i++) {
    int it = order[i];
    if (!item_is_batch[it]) continue;
    int min_r = 0;
    for (int r = 1; r < P_cone; r++) {
      if (batch_load[r] < batch_load[min_r] ||
          (batch_load[r] == batch_load[min_r] &&
           rank_load[r] < rank_load[min_r])) {
        min_r = r;
      }
    }
    item_owner[it] = min_r;
    batch_load[min_r] += item_cost[it];
    rank_load[min_r] += item_cost[it];
  }
  free(order);

  // ---- materialize owner_per_blk[] and lp_owner ----
  for (int bb = 0; bb < n_blks; bb++) owner_per_blk[bb] = -1;
  for (int i = 0; i < n_items; i++) {
    int owner = item_owner[i];
    if (item_is_lp[i]) {
      if (lp_owner_out) *lp_owner_out = owner;
    } else {
      int s = item_blk_starts[i], e = item_blk_starts[i + 1];
      for (int p = s; p < e; p++) owner_per_blk[item_blks[p]] = owner;
    }
  }

  if (LOG_V(3)) {
    int rank_global = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_global);
    if (rank_global == 0) {
      fprintf(stderr, "[LPT] cone assignment over P_cone=%d, %d items:\n",
              P_cone, n_items);
      for (int r = 0; r < P_cone; r++)
        fprintf(stderr, "  cone-rank %d: load=%lld\n", r, rank_load[r]);
      fflush(stderr);
    }
  }

  free(rank_load);
  free(batch_load);
  free(item_owner);
  free(item_is_batch);
  free(item_is_lp);
  free(item_cost);
  free(item_blks);
  free(item_blk_starts);
}

compressed_sdp_problem_t *
partition_problem(const compressed_sdp_problem_t *global_prob,
                  const grid_context_t *grid) {
  int P_row = grid->dims[0];
  int my_row = grid->coords[0];
  int P_cone = grid->dims[2];
  int my_cone = grid->coords[2];

  int m_total = global_prob->num_constraints;
  int start_row, end_row, local_m;
  compute_nnz_balanced_row_range(global_prob, P_row, my_row, &start_row,
                                 &end_row, &local_m);
  if (grid->coords[1] == 0 && grid->coords[2] == 0) {
    LOG_DBG("[Grid Row %d] Assigned constraints %d to %d (local_m = %d)\n",
            my_row, start_row, end_row - 1, local_m);
  }

  int n_blks_global = global_prob->n_blks;
  int *owner_per_blk =
      (int *)safe_malloc((n_blks_global > 0 ? n_blks_global : 1) * sizeof(int));
  int lp_owner = 0;
  compute_cone_assignment(global_prob, P_cone, owner_per_blk, &lp_owner);

  int lp_dim_global = global_prob->lp_dim;
  int lp_dim_local = (lp_dim_global > 0 && lp_owner == my_cone) ? lp_dim_global : 0;

  int n_blks_local = 0;
  for (int b = 0; b < n_blks_global; b++)
    if (owner_per_blk[b] == my_cone) n_blks_local++;

  int n_act_global = global_prob->n_active_vars;
  int lp_start_active = global_prob->lp_start_idx;
  int *keep_idx = (int *)safe_malloc(n_act_global * sizeof(int));
  int n_act_local = 0;

  char *col_used = (char *)safe_calloc(
      n_act_global > 0 ? (size_t)n_act_global : 1, 1);
  {
    sparse_csr_matrix_t *g_csr_scan = global_prob->constraint_matrix;
    for (int r = start_row; r < end_row; r++) {
      for (int p = g_csr_scan->row_ptr[r];
           p < g_csr_scan->row_ptr[r + 1]; p++) {
        int c = g_csr_scan->col_ind[p];
        if (c >= 0 && c < n_act_global) col_used[c] = 1;
      }
    }

    if (global_prob->objective_vector_sparse != NULL) {
      sparse_vector_t *gobj = global_prob->objective_vector_sparse;
      const long long *cm = global_prob->col_mapping;
      for (int k = 0; k < gobj->len; k++) {
        long long pos = gobj->pos[k];
        int lo = 0, hi = n_act_global - 1, found = -1;
        while (lo <= hi) {
          int mid = lo + (hi - lo) / 2;
          if (cm[mid] == pos) { found = mid; break; }
          if (cm[mid] < pos) lo = mid + 1;
          else hi = mid - 1;
        }
        if (found >= 0) col_used[found] = 1;
      }
    }
  }

  {
    int b_cursor = 0;
    for (int i = 0; i < n_act_global; i++) {
      long long g = global_prob->col_mapping[i];
      int owned;
      if (g >= global_prob->total_n_orig) {
        owned = (lp_dim_local > 0 && i >= lp_start_active &&
                 i < lp_start_active + lp_dim_local);
      } else {
        while (b_cursor < n_blks_global &&
               g >= global_prob->blk_ptr[b_cursor + 1])
          b_cursor++;
        owned = (b_cursor < n_blks_global &&
                 owner_per_blk[b_cursor] == my_cone);
      }

      int is_lp = (g >= global_prob->total_n_orig);
      int kept = owned && (is_lp || col_used[i]);
      keep_idx[i] = kept ? n_act_local++ : -1;
    }
  }
  free(col_used);

  long long *local_col_mapping = NULL;
  if (n_act_local > 0) {
    local_col_mapping =
        (long long *)safe_malloc(n_act_local * sizeof(long long));
    for (int i = 0, j = 0; i < n_act_global; i++) {
      if (keep_idx[i] >= 0)
        local_col_mapping[j++] = global_prob->col_mapping[i];
    }
  }

  compressed_sdp_problem_t *local_prob =
      (compressed_sdp_problem_t *)calloc(1, sizeof(compressed_sdp_problem_t));

  local_prob->num_constraints = local_m;
  local_prob->right_hand_side =
      (double *)safe_malloc(local_m * sizeof(double));
  memcpy(local_prob->right_hand_side,
         global_prob->right_hand_side + start_row,
         local_m * sizeof(double));

  local_prob->n_blks = n_blks_local;
  local_prob->lp_dim = lp_dim_local;
  local_prob->n_active_vars = n_act_local;
  local_prob->total_n_orig = global_prob->total_n_orig;
  local_prob->col_mapping = local_col_mapping;

  if (n_blks_local > 0) {
    local_prob->blk_dims = (int *)safe_malloc(n_blks_local * sizeof(int));
    local_prob->blk_ptr =
        (long long *)safe_malloc((n_blks_local + 1) * sizeof(long long));
    int w = 0;
    for (int b = 0; b < n_blks_global; b++) {
      if (owner_per_blk[b] != my_cone) continue;
      local_prob->blk_dims[w] = global_prob->blk_dims[b];
      local_prob->blk_ptr[w] = global_prob->blk_ptr[b];
      local_prob->blk_ptr[w + 1] = global_prob->blk_ptr[b + 1];
      w++;
    }
  } else {
    local_prob->blk_dims = NULL;
    local_prob->blk_ptr = (long long *)safe_malloc(sizeof(long long));
    local_prob->blk_ptr[0] = 0;
  }
  local_prob->lp_start_idx = (lp_dim_local > 0)
                                 ? (n_act_local - lp_dim_local)
                                 : n_act_local;

  sparse_csr_matrix_t *g_csr = global_prob->constraint_matrix;
  sparse_csr_matrix_t *l_csr =
      (sparse_csr_matrix_t *)safe_malloc(sizeof(sparse_csr_matrix_t));
  l_csr->num_rows = local_m;
  l_csr->num_cols = n_act_local;
  int *new_row_ptr = (int *)safe_malloc((local_m + 1) * sizeof(int));
  new_row_ptr[0] = 0;
  for (int r = 0; r < local_m; r++) {
    int gr = start_row + r;
    int cnt = 0;
    for (int p = g_csr->row_ptr[gr]; p < g_csr->row_ptr[gr + 1]; p++) {
      if (keep_idx[g_csr->col_ind[p]] >= 0) cnt++;
    }
    new_row_ptr[r + 1] = new_row_ptr[r] + cnt;
  }
  int local_nnz = new_row_ptr[local_m];
  int *new_col_ind =
      (int *)safe_malloc((local_nnz > 0 ? local_nnz : 1) * sizeof(int));
  double *new_val = (double *)safe_malloc(
      (local_nnz > 0 ? local_nnz : 1) * sizeof(double));
  for (int r = 0, w = 0; r < local_m; r++) {
    int gr = start_row + r;
    for (int p = g_csr->row_ptr[gr]; p < g_csr->row_ptr[gr + 1]; p++) {
      int gc = g_csr->col_ind[p];
      int lc = keep_idx[gc];
      if (lc >= 0) {
        new_col_ind[w] = lc;
        new_val[w] = g_csr->val[p];
        w++;
      }
    }
  }
  l_csr->row_ptr = new_row_ptr;
  l_csr->col_ind = new_col_ind;
  l_csr->val = new_val;
  l_csr->num_nonzeros = local_nnz;
  local_prob->constraint_matrix = l_csr;

  if (global_prob->objective_vector_sparse != NULL) {
    sparse_vector_t *g_obj = global_prob->objective_vector_sparse;
    int keep_obj = 0;
    if (n_blks_local > 0) {
      long long *owned_lo = (long long *)safe_malloc(
          n_blks_local * sizeof(long long));
      long long *owned_hi = (long long *)safe_malloc(
          n_blks_local * sizeof(long long));
      {
        int w = 0;
        for (int b = 0; b < n_blks_global; b++) {
          if (owner_per_blk[b] != my_cone) continue;
          owned_lo[w] = global_prob->blk_ptr[b];
          owned_hi[w] = global_prob->blk_ptr[b + 1];
          w++;
        }
      }
      int blk_cur = 0;
      for (int i = 0; i < g_obj->len; i++) {
        long long p = g_obj->pos[i];
        while (blk_cur < n_blks_local && p >= owned_hi[blk_cur])
          blk_cur++;
        if (blk_cur < n_blks_local && p >= owned_lo[blk_cur])
          keep_obj++;
      }
      sparse_vector_t *l_obj =
          (sparse_vector_t *)safe_malloc(sizeof(sparse_vector_t));
      l_obj->len = keep_obj;
      l_obj->pos = (long long *)safe_malloc(
          (keep_obj > 0 ? keep_obj : 1) * sizeof(long long));
      l_obj->val = (double *)safe_malloc(
          (keep_obj > 0 ? keep_obj : 1) * sizeof(double));
      blk_cur = 0;
      for (int i = 0, w = 0; i < g_obj->len; i++) {
        long long p = g_obj->pos[i];
        while (blk_cur < n_blks_local && p >= owned_hi[blk_cur])
          blk_cur++;
        if (blk_cur < n_blks_local && p >= owned_lo[blk_cur]) {
          l_obj->pos[w] = p;
          l_obj->val[w] = g_obj->val[i];
          w++;
        }
      }
      local_prob->objective_vector_sparse = l_obj;
      free(owned_lo);
      free(owned_hi);
    } else {
      sparse_vector_t *l_obj =
          (sparse_vector_t *)safe_malloc(sizeof(sparse_vector_t));
      l_obj->len = 0;
      l_obj->pos = (long long *)safe_malloc(sizeof(long long));
      l_obj->val = (double *)safe_malloc(sizeof(double));
      local_prob->objective_vector_sparse = l_obj;
    }
  } else {
    local_prob->objective_vector_sparse = NULL;
  }

  if (lp_dim_local > 0 && global_prob->lp_objective_vector != NULL) {
    local_prob->lp_objective_vector =
        (double *)safe_malloc(lp_dim_local * sizeof(double));
    memcpy(local_prob->lp_objective_vector,
           global_prob->lp_objective_vector,
           lp_dim_local * sizeof(double));
  } else {
    local_prob->lp_objective_vector = NULL;
  }

  free(keep_idx);
  free(owner_per_blk);
  return local_prob;
}

rescale_info_t *
partition_rescale_info(const rescale_info_t *global_info,
                       const compressed_sdp_problem_t *global_prob,
                       const grid_context_t *grid) {
  if (global_info == NULL || global_prob == NULL || grid == NULL)
    return NULL;

  int P_row = grid->dims[0];
  int my_row = grid->coords[0];
  int P_cone = grid->dims[2];
  int my_cone = grid->coords[2];

  int start_row, end_row, local_m;
  compute_nnz_balanced_row_range(global_prob, P_row, my_row, &start_row,
                                 &end_row, &local_m);
  (void)end_row;

  int n_blks_global = global_prob->n_blks;
  int *owner_per_blk = (int *)safe_malloc(
      (n_blks_global > 0 ? n_blks_global : 1) * sizeof(int));
  int lp_owner = 0;
  compute_cone_assignment(global_prob, P_cone, owner_per_blk, &lp_owner);
  int lp_dim_local =
      (global_prob->lp_dim > 0 && lp_owner == my_cone) ? global_prob->lp_dim
                                                       : 0;

  rescale_info_t *li =
      (rescale_info_t *)safe_calloc(1, sizeof(rescale_info_t));
  li->scaled_problem = NULL;
  li->objective_vector_rescaling = global_info->objective_vector_rescaling;
  li->right_hand_side_rescaling = global_info->right_hand_side_rescaling;
  li->unscaled_right_hand_side_norm =
      global_info->unscaled_right_hand_side_norm;
  li->unscaled_objective_vector_norm =
      global_info->unscaled_objective_vector_norm;
  li->rescaling_time_sec = global_info->rescaling_time_sec;

  if (local_m > 0 && global_info->constraint_rescaling != NULL) {
    li->constraint_rescaling =
        (double *)safe_malloc((size_t)local_m * sizeof(double));
    memcpy(li->constraint_rescaling,
           global_info->constraint_rescaling + start_row,
           (size_t)local_m * sizeof(double));
  } else {
    li->constraint_rescaling = NULL;
  }

  int local_psd_dim = 0;
  for (int b = 0; b < n_blks_global; b++)
    if (owner_per_blk[b] == my_cone)
      local_psd_dim += global_prob->blk_dims[b];
  if (local_psd_dim > 0 && global_info->psd_cone_rescaling != NULL) {
    li->psd_cone_rescaling =
        (double *)safe_malloc((size_t)local_psd_dim * sizeof(double));
    int global_off = 0;
    int local_off = 0;
    for (int b = 0; b < n_blks_global; b++) {
      int n_b = global_prob->blk_dims[b];
      if (owner_per_blk[b] == my_cone) {
        memcpy(li->psd_cone_rescaling + local_off,
               global_info->psd_cone_rescaling + global_off,
               (size_t)n_b * sizeof(double));
        local_off += n_b;
      }
      global_off += n_b;
    }
  } else {
    li->psd_cone_rescaling = NULL;
  }

  if (lp_dim_local > 0 && global_info->lp_variable_rescaling != NULL) {
    li->lp_variable_rescaling =
        (double *)safe_malloc((size_t)lp_dim_local * sizeof(double));
    memcpy(li->lp_variable_rescaling, global_info->lp_variable_rescaling,
           (size_t)lp_dim_local * sizeof(double));
  } else {
    li->lp_variable_rescaling = NULL;
  }

  free(owner_per_blk);
  return li;
}

void select_valid_grid_size(const cardal_parameters_t *params,
                            const compressed_sdp_problem_t *original_problem,
                            cardal_parameters_t *sub_params) {
    (void)original_problem;
    int world_size, rank_global;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_global);

    if (params->grid_size.decided) {
        int provided_row = params->grid_size.row_dims;
        int provided_rank = params->grid_size.rank_dims;
        int provided_cone = params->grid_size.cone_dims > 0
                              ? params->grid_size.cone_dims
                              : 1;
        int product = provided_row * provided_rank * provided_cone;

        if (product != world_size) {
            if (rank_global == 0) {
                fprintf(stderr, "\n[Error] MPI World Size Mismatch!\n");
                fprintf(stderr, "------------------------------------------------\n");
                fprintf(stderr,
                        "User specified grid:  %d x %d x %d = %d processes\n",
                        provided_row, provided_rank, provided_cone, product);
                fprintf(stderr, "Actual MPI world size: %d processes\n", world_size);
                fprintf(stderr, "Please adjust mpirun -n or --grid_size.\n");
                fprintf(stderr, "------------------------------------------------\n");
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        sub_params->grid_size.row_dims = provided_row;
        sub_params->grid_size.rank_dims = provided_rank;
        sub_params->grid_size.cone_dims = provided_cone;
    } else {
        sub_params->grid_size.row_dims = world_size;
        sub_params->grid_size.rank_dims = 1;
        sub_params->grid_size.cone_dims = 1;
        sub_params->grid_size.decided = true;
    }
}

#define CHUNK_SIZE (1024 * 1024 * 1024)

void big_bcast_bytes(void **buffer_ptr, size_t *size_ptr, int root,
                     MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  int is_root = (rank == root);

  unsigned long long total_len = is_root ? *size_ptr : 0;
  MPI_Bcast(&total_len, 1, MPI_UNSIGNED_LONG_LONG, root, comm);

  if (!is_root) {
    *size_ptr = (size_t)total_len;
    *buffer_ptr = safe_malloc(total_len);
    if (*buffer_ptr == NULL && total_len > 0) {
      fprintf(stderr,
              "[FATAL] Rank %d failed to allocate %llu bytes for Broadcast!\n",
              rank, total_len);
      fflush(stderr);
      MPI_Abort(comm, EXIT_FAILURE);
    }
  }

  if (is_root && *buffer_ptr == NULL && total_len > 0) {
    fprintf(stderr, "[FATAL] Rank %d Buffer is NULL before Broadcast!\n", rank);
    fflush(stderr);
    MPI_Abort(comm, EXIT_FAILURE);
  }

  char *buf = (char *)(*buffer_ptr);
  size_t offset = 0;

  while (offset < total_len) {
    size_t remaining = total_len - offset;
    int current_chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : (int)remaining;

    MPI_Bcast(buf + offset, current_chunk, MPI_BYTE, root, comm);

    offset += current_chunk;
  }
}

void serialize_compressed_sdp(const compressed_sdp_problem_t *prob,
                              void **out_buf, size_t *out_size) {
  size_t size = 0;

  size += sizeof(int) * 3;       // num_constraints, n_blks, n_active_vars
  size += sizeof(long long) * 1; // total_n_orig
  size += prob->n_active_vars * sizeof(long long); // col_mapping
  size += prob->n_blks * sizeof(int);              // blk_dims
  size += (prob->n_blks + 1) * sizeof(long long);  // blk_ptr
  size += prob->num_constraints * sizeof(double);  // right_hand_side

  size += sizeof(int); // lp_dim
  size += sizeof(int); // lp_start_idx
  size += sizeof(int);
  if (prob->lp_dim > 0 && prob->lp_objective_vector != NULL) {
    size += prob->lp_dim * sizeof(double);
  }

  int has_csr = (prob->constraint_matrix != NULL);
  size += sizeof(int);
  if (has_csr) {
    size += sizeof(int) * 3; // num_rows, num_cols, num_nonzeros
    size += (prob->constraint_matrix->num_rows + 1) * sizeof(int);
    size += prob->constraint_matrix->num_nonzeros * sizeof(int);
    size += prob->constraint_matrix->num_nonzeros * sizeof(double);
  }

  int has_obj_sp = (prob->objective_vector_sparse != NULL);
  size += sizeof(int);
  if (has_obj_sp) {
    size += sizeof(int); // len
    size += prob->objective_vector_sparse->len * sizeof(long long);
    size += prob->objective_vector_sparse->len * sizeof(double);
  }

  *out_buf = safe_malloc(size);
  *out_size = size;
  char *ptr = (char *)(*out_buf);

#define WRITE_BUF(src, bytes)                                                  \
  do {                                                                         \
    memcpy(ptr, src, bytes);                                                   \
    ptr += bytes;                                                              \
  } while (0)

  WRITE_BUF(&prob->num_constraints, sizeof(int));
  WRITE_BUF(&prob->n_blks, sizeof(int));
  WRITE_BUF(&prob->n_active_vars, sizeof(int));
  WRITE_BUF(&prob->total_n_orig, sizeof(long long));

  WRITE_BUF(prob->col_mapping, prob->n_active_vars * sizeof(long long));
  WRITE_BUF(prob->blk_dims, prob->n_blks * sizeof(int));
  WRITE_BUF(prob->blk_ptr, (prob->n_blks + 1) * sizeof(long long));
  WRITE_BUF(prob->right_hand_side, prob->num_constraints * sizeof(double));

  WRITE_BUF(&prob->lp_dim, sizeof(int));
  WRITE_BUF(&prob->lp_start_idx, sizeof(int));

  int has_lp_obj = (prob->lp_dim > 0 && prob->lp_objective_vector != NULL);
  WRITE_BUF(&has_lp_obj, sizeof(int));
  if (has_lp_obj) {
    WRITE_BUF(prob->lp_objective_vector, prob->lp_dim * sizeof(double));
  }

  WRITE_BUF(&has_csr, sizeof(int));
  if (has_csr) {
    WRITE_BUF(&prob->constraint_matrix->num_rows, sizeof(int));
    WRITE_BUF(&prob->constraint_matrix->num_cols, sizeof(int));
    WRITE_BUF(&prob->constraint_matrix->num_nonzeros, sizeof(int));

    WRITE_BUF(prob->constraint_matrix->row_ptr,
              (prob->constraint_matrix->num_rows + 1) * sizeof(int));
    WRITE_BUF(prob->constraint_matrix->col_ind,
              prob->constraint_matrix->num_nonzeros * sizeof(int));
    WRITE_BUF(prob->constraint_matrix->val,
              prob->constraint_matrix->num_nonzeros * sizeof(double));
  }

  WRITE_BUF(&has_obj_sp, sizeof(int));
  if (has_obj_sp) {
    WRITE_BUF(&prob->objective_vector_sparse->len, sizeof(int));
    WRITE_BUF(prob->objective_vector_sparse->pos,
              prob->objective_vector_sparse->len * sizeof(long long));
    WRITE_BUF(prob->objective_vector_sparse->val,
              prob->objective_vector_sparse->len * sizeof(double));
  }
}

compressed_sdp_problem_t *deserialize_compressed_sdp(void *buf) {
  compressed_sdp_problem_t *prob = (compressed_sdp_problem_t *)safe_calloc(
      1, sizeof(compressed_sdp_problem_t));
  char *ptr = (char *)buf;

#define READ_BUF(dst, bytes)                                                   \
  do {                                                                         \
    memcpy(dst, ptr, bytes);                                                   \
    ptr += bytes;                                                              \
  } while (0)

  READ_BUF(&prob->num_constraints, sizeof(int));
  READ_BUF(&prob->n_blks, sizeof(int));
  READ_BUF(&prob->n_active_vars, sizeof(int));
  READ_BUF(&prob->total_n_orig, sizeof(long long));

  prob->col_mapping =
      (long long *)safe_malloc(prob->n_active_vars * sizeof(long long));
  READ_BUF(prob->col_mapping, prob->n_active_vars * sizeof(long long));

  prob->blk_dims = (int *)safe_malloc(prob->n_blks * sizeof(int));
  READ_BUF(prob->blk_dims, prob->n_blks * sizeof(int));

  prob->blk_ptr =
      (long long *)safe_malloc((prob->n_blks + 1) * sizeof(long long));
  READ_BUF(prob->blk_ptr, (prob->n_blks + 1) * sizeof(long long));

  prob->right_hand_side =
      (double *)safe_malloc(prob->num_constraints * sizeof(double));
  READ_BUF(prob->right_hand_side, prob->num_constraints * sizeof(double));

  READ_BUF(&prob->lp_dim, sizeof(int));
  READ_BUF(&prob->lp_start_idx, sizeof(int));
  int has_lp_obj;
  READ_BUF(&has_lp_obj, sizeof(int));
  if (has_lp_obj) {
    prob->lp_objective_vector =
        (double *)safe_malloc(prob->lp_dim * sizeof(double));
    READ_BUF(prob->lp_objective_vector, prob->lp_dim * sizeof(double));
  } else {
    prob->lp_objective_vector = NULL;
  }

  int has_csr;
  READ_BUF(&has_csr, sizeof(int));
  if (has_csr) {
    prob->constraint_matrix =
        (sparse_csr_matrix_t *)safe_calloc(1, sizeof(sparse_csr_matrix_t));

    READ_BUF(&prob->constraint_matrix->num_rows, sizeof(int));
    READ_BUF(&prob->constraint_matrix->num_cols, sizeof(int));
    READ_BUF(&prob->constraint_matrix->num_nonzeros, sizeof(int));

    int nrows = prob->constraint_matrix->num_rows;
    int nnz = prob->constraint_matrix->num_nonzeros;

    prob->constraint_matrix->row_ptr =
        (int *)safe_malloc((nrows + 1) * sizeof(int));
    READ_BUF(prob->constraint_matrix->row_ptr, (nrows + 1) * sizeof(int));

    prob->constraint_matrix->col_ind = (int *)safe_malloc(nnz * sizeof(int));
    READ_BUF(prob->constraint_matrix->col_ind, nnz * sizeof(int));

    prob->constraint_matrix->val = (double *)safe_malloc(nnz * sizeof(double));
    READ_BUF(prob->constraint_matrix->val, nnz * sizeof(double));
  }

  int has_obj_sp;
  READ_BUF(&has_obj_sp, sizeof(int));
  if (has_obj_sp) {
    prob->objective_vector_sparse =
        (sparse_vector_t *)safe_calloc(1, sizeof(sparse_vector_t));

    READ_BUF(&prob->objective_vector_sparse->len, sizeof(int));
    int len = prob->objective_vector_sparse->len;

    prob->objective_vector_sparse->pos =
        (long long *)safe_malloc(len * sizeof(long long));
    READ_BUF(prob->objective_vector_sparse->pos, len * sizeof(long long));

    prob->objective_vector_sparse->val =
        (double *)safe_malloc(len * sizeof(double));
    READ_BUF(prob->objective_vector_sparse->val, len * sizeof(double));
  }

  return prob;
}

void cleanup_parallel_context(grid_context_t *grid) {
  if (grid->nccl_rank)
    NCCL_CHECK(ncclCommDestroy(grid->nccl_rank));
  if (grid->nccl_row)
    NCCL_CHECK(ncclCommDestroy(grid->nccl_row));
  if (grid->nccl_cone)
    NCCL_CHECK(ncclCommDestroy(grid->nccl_cone));

  if (grid->comm_rank != MPI_COMM_NULL)
    MPI_Comm_free(&grid->comm_rank);
  if (grid->comm_row != MPI_COMM_NULL)
    MPI_Comm_free(&grid->comm_row);
  if (grid->comm_cone != MPI_COMM_NULL)
    MPI_Comm_free(&grid->comm_cone);
}

int *permute_global_problem_constraints(compressed_sdp_problem_t *prob,
                                        shuffle_type_t type, int block_size,
                                        int seed) {
  int m = prob->num_constraints;
  if (m <= 1 || type == SHUFFLE_NONE) {
    return NULL;
  }

  int *p = (int *)safe_malloc(m * sizeof(int));
  srand(seed);

  if (type == SHUFFLE_UNIFORM) {
    for (int i = 0; i < m; i++)
      p[i] = i;
    for (int i = m - 1; i > 0; i--) {
      int j = rand() % (i + 1);
      int temp = p[i];
      p[i] = p[j];
      p[j] = temp;
    }
  } else if (type == SHUFFLE_BLOCK) {
    if (block_size <= 0)
      block_size = 128;
    int num_blocks = (m + block_size - 1) / block_size;

    int *b_perm = (int *)safe_malloc(num_blocks * sizeof(int));
    for (int i = 0; i < num_blocks; i++)
      b_perm[i] = i;

    for (int i = num_blocks - 1; i > 0; i--) {
      int j = rand() % (i + 1);
      int temp = b_perm[i];
      b_perm[i] = b_perm[j];
      b_perm[j] = temp;
    }

    int idx = 0;
    for (int i = 0; i < num_blocks; i++) {
      int blk = b_perm[i];
      int start_row = blk * block_size;
      int end_row = start_row + block_size;
      if (end_row > m)
        end_row = m;

      for (int r = start_row; r < end_row; r++) {
        p[idx++] = r;
      }
    }
    free(b_perm);
  } else if (type == SHUFFLE_COL_LOCALITY) {
    const int *row_ptr = prob->constraint_matrix->row_ptr;
    const int *col_ind = prob->constraint_matrix->col_ind;
    unsigned long long *keys =
        (unsigned long long *)safe_malloc(m * sizeof(unsigned long long));
    for (int i = 0; i < m; i++) {
      int s = row_ptr[i], e = row_ptr[i + 1];
      if (s >= e) {
        keys[i] = (unsigned long long)-1;
      } else {
        unsigned long long c0 = (unsigned int)col_ind[s];
        unsigned long long c1 =
            (s + 1 < e) ? (unsigned int)col_ind[s + 1] : c0;
        keys[i] = (c0 << 32) | c1;
      }
    }
    for (int i = 0; i < m; i++) p[i] = i;
    g_locality_keys = keys;
    qsort(p, (size_t)m, sizeof(int), cmp_by_locality_key);
    g_locality_keys = NULL;
    free(keys);
  }

  int nnz = prob->constraint_matrix->num_nonzeros;
  double *new_rhs = (double *)safe_malloc(m * sizeof(double));
  int *new_row_ptr = (int *)safe_malloc((m + 1) * sizeof(int));
  int *new_col_ind = (int *)safe_malloc(nnz * sizeof(int));
  double *new_val = (double *)safe_malloc(nnz * sizeof(double));

  new_row_ptr[0] = 0;
  int current_nnz = 0;

  for (int i = 0; i < m; i++) {
    int orig_row = p[i];
    new_rhs[i] = prob->right_hand_side[orig_row];

    int row_start = prob->constraint_matrix->row_ptr[orig_row];
    int row_end = prob->constraint_matrix->row_ptr[orig_row + 1];
    int row_nnz = row_end - row_start;

    if (row_nnz > 0) {
      memcpy(&new_col_ind[current_nnz],
             &prob->constraint_matrix->col_ind[row_start],
             row_nnz * sizeof(int));
      memcpy(&new_val[current_nnz], &prob->constraint_matrix->val[row_start],
             row_nnz * sizeof(double));
    }

    current_nnz += row_nnz;
    new_row_ptr[i + 1] = current_nnz;
  }

  free(prob->right_hand_side);
  free(prob->constraint_matrix->row_ptr);
  free(prob->constraint_matrix->col_ind);
  free(prob->constraint_matrix->val);

  prob->right_hand_side = new_rhs;
  prob->constraint_matrix->row_ptr = new_row_ptr;
  prob->constraint_matrix->col_ind = new_col_ind;
  prob->constraint_matrix->val = new_val;

  return p;
}

void unpermute_dual_solution(int m, const double *shuffled_global_dual,
                             double *orig_global_dual, const int *p) {
  if (p == NULL) {
    memcpy(orig_global_dual, shuffled_global_dual, m * sizeof(double));
    return;
  }

  for (int i = 0; i < m; i++) {
    int orig_idx = p[i];
    orig_global_dual[orig_idx] = shuffled_global_dual[i];
  }
}

void gather_sdp_result(sdp_result_t *result,
                                         cardal_sdp_solver_state_t *state) {
  if (!result || !state || !state->grid_context)
    return;
  grid_context_t *gc = state->grid_context;
  int P_row = gc->dims[0], P_rank = gc->dims[1];
  int my_row = gc->coords[0], my_rank = gc->coords[1];
  int n_blks = state->n_blks;

  if (P_row == 1 && P_rank == 1)
    return;

  if (my_row != 0) {
    free(result->low_rank_primal_solution);
    result->low_rank_primal_solution = NULL;
    result->low_rank_solution_length = 0;
    free(result->rank_list);
    result->rank_list = NULL;
    result->n_cones = 0;
    return;
  }

  int *rank_global = (int *)safe_malloc((size_t)n_blks * sizeof(int));
  MPI_Allreduce(state->rank_list, rank_global, n_blks, MPI_INT, MPI_SUM,
                gc->comm_rank);

  int *rank_per_rank = NULL;
  if (my_rank == 0)
    rank_per_rank = (int *)safe_malloc((size_t)n_blks * P_rank * sizeof(int));
  MPI_Gather(state->rank_list, n_blks, MPI_INT, rank_per_rank, n_blks, MPI_INT,
             0, gc->comm_rank);

  const double *R_local_host = result->low_rank_primal_solution;
  long long local_len = result->low_rank_solution_length;

  long long *off_local = (long long *)safe_malloc(
      (size_t)(n_blks + 1) * sizeof(long long));
  off_local[0] = 0;
  for (int b = 0; b < n_blks; b++)
    off_local[b + 1] =
        off_local[b] + (long long)state->blk_dims[b] * state->rank_list[b];
  if (off_local[n_blks] != local_len) {

    fprintf(stderr,
            "gather_sdp_result: local cone slices "
            "(%lld) != length_low_rank_solution (%lld)\n",
            off_local[n_blks], local_len);
  }

  long long total_global = 0;
  long long *off_global = NULL;
  double *R_global = NULL;
  if (my_rank == 0) {
    off_global = (long long *)safe_malloc((size_t)(n_blks + 1) *
                                          sizeof(long long));
    off_global[0] = 0;
    for (int b = 0; b < n_blks; b++)
      off_global[b + 1] =
          off_global[b] + (long long)state->blk_dims[b] * rank_global[b];
    total_global = off_global[n_blks];
    R_global = (double *)safe_malloc((size_t)total_global * sizeof(double));
  }

  for (int b = 0; b < n_blks; b++) {
    int n_b = state->blk_dims[b];
    int send_count = n_b * state->rank_list[b];
    int *recvcounts = NULL;
    int *displs = NULL;
    if (my_rank == 0) {
      recvcounts = (int *)safe_malloc((size_t)P_rank * sizeof(int));
      displs = (int *)safe_malloc((size_t)P_rank * sizeof(int));
      int cur = 0;
      for (int c = 0; c < P_rank; c++) {
        int r_c = rank_per_rank[c * n_blks + b];
        recvcounts[c] = n_b * r_c;
        displs[c] = cur;
        cur += recvcounts[c];
      }
    }
    MPI_Gatherv(R_local_host + off_local[b], send_count, MPI_DOUBLE,
                my_rank == 0 ? (R_global + off_global[b]) : NULL,
                recvcounts, displs, MPI_DOUBLE, 0, gc->comm_rank);
    if (my_rank == 0) {
      free(recvcounts);
      free(displs);
    }
  }

  free(off_local);

  if (my_rank == 0) {
    free(result->low_rank_primal_solution);
    result->low_rank_primal_solution = R_global;
    result->low_rank_solution_length = total_global;
    free(result->rank_list);
    result->rank_list = rank_global;
    free(off_global);
    free(rank_per_rank);
  } else {
    free(result->low_rank_primal_solution);
    result->low_rank_primal_solution = NULL;
    result->low_rank_solution_length = 0;
    free(result->rank_list);
    result->rank_list = NULL;
    result->n_cones = 0;
    free(rank_global);
  }
  (void)P_row;
}

static void format_dim_count_str(const int *dims, const int *counts, int n,
                                 char *out, int out_cap) {
  if (n == 0) { snprintf(out, out_cap, "-"); return; }
  int idx[64];
  if (n > 64) n = 64;
  for (int i = 0; i < n; i++) idx[i] = i;
  for (int i = 1; i < n; i++) {
    int k = idx[i], j = i;
    while (j > 0 && dims[idx[j - 1]] < dims[k]) { idx[j] = idx[j - 1]; j--; }
    idx[j] = k;
  }
  int pos = 0;
  for (int i = 0; i < n && pos < out_cap - 1; i++) {
    int j = idx[i];
    int w = snprintf(out + pos, out_cap - pos, "%s%dx%d",
                     i > 0 ? ", " : "", counts[j], dims[j]);
    if (w < 0) break;
    pos += w;
  }
}

void print_per_rank_workload(const grid_context_t *grid,
                             const compressed_sdp_problem_t *local_prob,
                             int m_total_global,
                             int initial_rank_global) {
  enum {
    WL_STR_CAP = 96,
  };

  int world_size = grid->dims[0] * grid->dims[1] * grid->dims[2];
  int my_world = grid->rank_global;
  int P_row = grid->dims[0];
  int P_rank = grid->dims[1];
  int my_grid_row = grid->coords[0];
  int my_grid_rank = grid->coords[1];

  int bm_base = (P_rank > 0) ? initial_rank_global / P_rank : initial_rank_global;
  int bm_rem  = (P_rank > 0) ? initial_rank_global % P_rank : 0;
  int bm_count_local = bm_base + (my_grid_rank < bm_rem ? 1 : 0);

  int row_base = (P_row > 0) ? m_total_global / P_row : m_total_global;
  int row_rem  = (P_row > 0) ? m_total_global % P_row : 0;
  int local_m  = row_base + (my_grid_row < row_rem ? 1 : 0);

  int single_dims[64], single_counts[64], n_single = 0;
  int batch_dims[64],  batch_counts[64],  n_batch  = 0;
  (void)INT_MAX;
  for (int i = 0; i < local_prob->n_blks; i++) {
    int d = local_prob->blk_dims[i];
    int is_batch = (d <= CARDAL_SMALL_CONE_DIM_THRESHOLD);
    int *dims  = is_batch ? batch_dims  : single_dims;
    int *cnts  = is_batch ? batch_counts : single_counts;
    int *np    = is_batch ? &n_batch    : &n_single;
    int cap    = 64;
    int found = -1;
    for (int j = 0; j < *np; j++) if (dims[j] == d) { found = j; break; }
    if (found >= 0) cnts[found]++;
    else if (*np < cap) { dims[*np] = d; cnts[*np] = 1; (*np)++; }
  }

  char single_str[WL_STR_CAP];
  char batch_str[WL_STR_CAP];
  format_dim_count_str(single_dims, single_counts, n_single, single_str,
                       WL_STR_CAP);
  format_dim_count_str(batch_dims, batch_counts, n_batch, batch_str,
                       WL_STR_CAP);

  enum { N_INTS = 7 };
  size_t blob_bytes = N_INTS * sizeof(int) + 2 * WL_STR_CAP;
  char *my_blob = (char *)safe_calloc(1, blob_bytes);
  int *my_ints = (int *)my_blob;
  my_ints[0] = grid->coords[0];
  my_ints[1] = grid->coords[1];
  my_ints[2] = grid->coords[2];
  my_ints[3] = local_m;
  my_ints[4] = bm_count_local;
  my_ints[5] = local_prob->lp_dim;
  my_ints[6] = local_prob->n_blks;
  char *my_single = my_blob + N_INTS * sizeof(int);
  char *my_batch  = my_single + WL_STR_CAP;
  snprintf(my_single, WL_STR_CAP, "%s", single_str);
  snprintf(my_batch,  WL_STR_CAP, "%s", batch_str);

  char *all_blobs = NULL;
  if (my_world == 0) {
    all_blobs = (char *)safe_malloc(world_size * blob_bytes);
  }
  MPI_Gather(my_blob, (int)blob_bytes, MPI_BYTE, all_blobs,
             (int)blob_bytes, MPI_BYTE, 0, grid->comm_global);

  if (my_world == 0 && LOG_V(2)) {
    print_subtitle("Per-rank Workload");
    printf("  single cone = dim > %d (PERCONE);  batch cone = dim <= %d "
           "(Batched CUSTOM).\n",
           CARDAL_SMALL_CONE_DIM_THRESHOLD, CARDAL_SMALL_CONE_DIM_THRESHOLD);
    printf("  %-5s %-9s | %-5s | %-7s | %-3s | %-26s | %s\n",
           "world", "(r,k,c)", "rows", "BM-rank", "LP",
           "single cones (n x dim)", "batch cones (n x dim)");
    for (int w = 0; w < world_size; w++) {
      const char *blob = all_blobs + w * blob_bytes;
      const int *ip = (const int *)blob;
      const char *sstr = blob + N_INTS * sizeof(int);
      const char *bstr = sstr + WL_STR_CAP;
      double row_pct =
          (m_total_global > 0) ? 100.0 * ip[3] / m_total_global : 0.0;
      double bm_pct =
          (initial_rank_global > 0) ? 100.0 * ip[4] / initial_rank_global : 0.0;
      char coord[16], rowstr[8], bmstr[8], lpstr[8];
      snprintf(coord,  sizeof(coord),  "(%d,%d,%d)", ip[0], ip[1], ip[2]);
      snprintf(rowstr, sizeof(rowstr), "%.0f%%", row_pct);
      snprintf(bmstr,  sizeof(bmstr),  "%.0f%%", bm_pct);
      snprintf(lpstr,  sizeof(lpstr),  "%s", ip[5] > 0 ? "yes" : "-");
      printf("  %-5d %-9s | %-5s | %-7s | %-3s | %-26s | %s\n",
             w, coord, rowstr, bmstr, lpstr, sstr, bstr);
    }
  }

  free(my_blob);
  if (all_blobs) free(all_blobs);
}
}