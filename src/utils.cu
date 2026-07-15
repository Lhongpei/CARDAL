/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */


#include "internal_types.h"
#include "sdp_types.h"
#include "utils.h"
#include <math.h>
#include <random>
#include <errno.h>
#include <limits.h>
#include <string.h>

std::mt19937 gen(1);
std::normal_distribution<double> dist(0.0, 1.0);

const double HOST_ONE = 1.0;
const double HOST_ZERO = 0.0;

int g_log_verbose = 3;

#define CARDAL_RULE                                                             \
  "================================================================================"
#define CARDAL_LOG_RULE                                                         \
  "----------------------------------------------------------------------------" \
  "--------------------"
#define CARDAL_VERSION "0.0.1"

void print_cardal_banner(void) {
  if (!LOG_V(1))
    return;
  printf("%s\n", CARDAL_RULE);
  printf("                                 CARDAL v%s\n", CARDAL_VERSION);
  printf("          A Burer-Monteiro Augmented Lagrangian Method for "
         "Large-Scale\n");
  printf("                          SDPs on Multi-GPU Systems\n");
  printf("\n");
  printf("                     Hongpei Li (ishongpeili@gmail.com)\n");
  printf("%s\n", CARDAL_RULE);
}

void print_subtitle(const char *title) {
  if (!LOG_V(2))
    return;
  // Centered "-- Title --" header within the 80-col rule width.
  size_t inner_len = strlen(title) + 6; // "-- " + title + " --"
  int total_width = 80;
  int left_pad = ((int)((size_t)total_width - inner_len)) / 2;
  if (left_pad < 0)
    left_pad = 0;
  printf("\n%*s-- %s --\n", left_pad, "", title);
}

void print_kv_str(const char *key, const char *val) {
  if (!LOG_V(2))
    return;
  printf("  %-18s: %s\n", key, val);
}
void print_kv_int(const char *key, long long val) {
  if (!LOG_V(2))
    return;
  printf("  %-18s: %lld\n", key, val);
}
void print_kv_dbl(const char *key, const char *fmt, double val) {
  if (!LOG_V(2))
    return;
  char buf[64];
  snprintf(buf, sizeof(buf), fmt, val);
  printf("  %-18s: %s\n", key, buf);
}

const char *termination_reason_to_string(termination_reason_t r) {
  switch (r) {
  case TERMINATION_REASON_OPTIMAL:
    return "OPTIMAL";
  case TERMINATION_REASON_TIME_LIMIT:
    return "TIME_LIMIT";
  case TERMINATION_REASON_ITERATION_LIMIT:
    return "ITERATION_LIMIT";
  case TERMINATION_REASON_USER_INTERRUPT:
    return "USER_INTERRUPT";
  case TERMINATION_REASON_UNSPECIFIED:
  default:
    return "UNSPECIFIED";
  }
}

void free_sdp_result(sdp_result_t *result) {
  if (!result)
    return;
  free(result->low_rank_primal_solution);
  free(result->dual_solution);
  free(result->rank_list);
  free(result);
}

void print_runtime_environment_section(int verbose_floor, int is_distributed,
                                       int world_size, int row_dims,
                                       int rank_dims, int cone_dims) {
  if (!LOG_V(verbose_floor))
    return;
  print_subtitle("Runtime Environment");
  print_kv_str("Execution mode",
               is_distributed ? "Distributed (MPI)" : "Single GPU");
  if (is_distributed) {
    print_kv_int("MPI processes", world_size);
    char grid_buf[80];
    int k = (cone_dims > 0) ? cone_dims : 1;
    if (row_dims > 0 && rank_dims > 0) {
      if (k > 1)
        snprintf(grid_buf, sizeof(grid_buf),
                 "%d x %d x %d (row x rank x cone)", row_dims, rank_dims, k);
      else
        snprintf(grid_buf, sizeof(grid_buf), "%d x %d (row x rank)", row_dims,
                 rank_dims);
    } else {
      snprintf(grid_buf, sizeof(grid_buf), "auto (P x 1)");
    }
    print_kv_str("Grid topology", grid_buf);
  }
}

void print_parameters_section(const cardal_parameters_t *params,
                              int verbose_floor) {
  if (!LOG_V(verbose_floor) || params == NULL)
    return;
  print_subtitle("Parameters");
  print_kv_dbl("Tol (feas/opt)", "%.1e",
               params->termination_criteria.eps_feasible_relative);
  print_kv_int("Outer iter cap",
               params->termination_criteria.iteration_limit);
  print_kv_int("Inner iter cap", (long long)params->inner_iterations_limit);
  print_kv_dbl("Penalty factor", "%.3g", params->penalty_factor);
  print_kv_dbl("Max penalty rho", "%.2e", params->max_penalty_coef);
  if (params->termination_criteria.time_sec_limit > 0.0)
    print_kv_dbl("Time limit (sec)", "%.1f",
                 params->termination_criteria.time_sec_limit);
  else
    print_kv_str("Time limit (sec)", "(none)");
  print_kv_int("LBFGS history m", params->lbfgs_history_size);
  print_kv_str("Initial rank",
               params->initial_rank > 0 ? "Fixed" : "Auto (2*log m)");
  if (params->max_rank > 0)
    print_kv_int("Max rank", params->max_rank);
  else
    print_kv_str("Max rank", "Auto (ceil((sqrt(8m+1)-1)/2))");
  print_kv_str("Initial rho", params->initial_penalty_coef > 0
                                  ? "Fixed"
                                  : "Auto (2/sqrt N)");
}

void print_problem_statistics_section(const compressed_sdp_problem_t *prob,
                                      int verbose_floor) {
  if (!LOG_V(verbose_floor) || prob == NULL)
    return;
  print_subtitle("SDP Problem Statistics");
  print_kv_int("Constraints (m)", prob->num_constraints);
  print_kv_int("Active variables", prob->n_active_vars);
  print_kv_int("LP variables", prob->lp_dim);
  print_kv_int("PSD cones", prob->n_blks);
  if (prob->constraint_matrix != NULL)
    print_kv_int("Constraint NNZ", prob->constraint_matrix->num_nonzeros);

  if (prob->n_blks > 0) {
    int n_cones = prob->n_blks;
    int *uniq_sizes = (int *)safe_malloc(n_cones * sizeof(int));
    int *uniq_counts = (int *)safe_calloc(n_cones, sizeof(int));
    int num_distinct = 0;
    for (int i = 0; i < n_cones; i++) {
      int dim = prob->blk_dims[i];
      int found = 0;
      for (int j = 0; j < num_distinct; j++) {
        if (uniq_sizes[j] == dim) {
          uniq_counts[j]++;
          found = 1;
          break;
        }
      }
      if (!found) {
        uniq_sizes[num_distinct] = dim;
        uniq_counts[num_distinct] = 1;
        num_distinct++;
      }
    }
    printf("  %-18s:\n", "Cone size distrib.");
    for (int i = 0; i < num_distinct; i++) {
      const char *tag = (uniq_sizes[i] <= CARDAL_SMALL_CONE_DIM_THRESHOLD &&
                         uniq_counts[i] >= CARDAL_MIN_BATCH_SIZE)
                            ? " (Batched)"
                            : "";
      printf("    - %d cone(s) of size %d x %d%s\n", uniq_counts[i],
             uniq_sizes[i], uniq_sizes[i], tag);
    }
    free(uniq_sizes);
    free(uniq_counts);
  }
}

void print_optimization_footer(const sdp_result_t *result,
                               const char *summary_file_path,
                               int verbose_floor) {
  if (!LOG_V(verbose_floor) || result == NULL)
    return;
  printf("\n%s\n", CARDAL_RULE);
  printf("                             Optimization Summary\n");
  printf("%s\n", CARDAL_RULE);
  printf("  %-18s: %s\n", "Status",
         termination_reason_to_string(result->termination_reason));
  printf("  %-18s: %.2f\n", "Runtime (sec)", result->cumulative_time_sec);
  printf("  %-18s: %d\n", "Outer iters", result->total_count);
  printf("  %-18s: %d\n", "Inner iters", result->total_inner_count);
  printf("  %-18s: %.6e\n", "Primal obj", result->primal_objective_value);
  printf("  %-18s: %.6e\n", "Dual   obj", result->dual_objective_value);
  printf("  %-18s: %.2e\n", "Rel primal res", result->relative_primal_residual);
  printf("  %-18s: %.2e\n", "Rel dual   res", result->relative_dual_residual);
  printf("  %-18s: %.2e\n", "Rel obj gap", result->relative_objective_gap);
  printf("  %-18s: %d\n", "BM rank (total)", result->rank);
  if (summary_file_path != NULL)
    printf("  %-18s: %s\n", "Summary file", summary_file_path);
  printf("%s\n", CARDAL_RULE);
}

void *safe_malloc(size_t size) {
  void *ptr = malloc(size);
  if (ptr == NULL) {
    fprintf(stderr, "Fatal: malloc(%zu bytes / %.2f GB) failed: %s\n",
            size, size / 1e9, strerror(errno));
    exit(EXIT_FAILURE);
  }
  return ptr;
}

void *safe_calloc(size_t num, size_t size) {
  void *ptr = calloc(num, size);
  if (ptr == NULL) {
    perror("Fatal error: calloc failed");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

void *safe_realloc(void *ptr, size_t new_size) {
  if (new_size == 0) {
    free(ptr);
    return NULL;
  }
  void *tmp = realloc(ptr, new_size);
  if (!tmp) {
    perror("Fatal error: realloc failed");
    exit(EXIT_FAILURE);
  }
  return tmp;
}

void *safe_memcpy(void *dest, const void *src, size_t n) {
  if (n == 0) {
    return dest;
  }
  if (dest == NULL || src == NULL) {
    fprintf(
        stderr,
        "Fatal error: safe_memcpy received NULL pointer (dest=%p, src=%p)\n",
        dest, src);
    exit(EXIT_FAILURE);
  }
  void *tmp = memcpy(dest, src, n);
  if (!tmp) {
    perror("Fatal error: memcpy failed");
    exit(EXIT_FAILURE);
  }

  return tmp;
}

int compare_triplets(const void *a, const void *b) {
  coo_triplet_t *t1 = (coo_triplet_t *)a;
  coo_triplet_t *t2 = (coo_triplet_t *)b;
  if (t1->row != t2->row)
    return t1->row - t2->row;
  return t1->col - t2->col;
}

int cmp_ll_asc(const void *a, const void *b) {
  long long arg1 = *(const long long *)a;
  long long arg2 = *(const long long *)b;
  if (arg1 < arg2)
    return -1;
  if (arg1 > arg2)
    return 1;
  return 0;
}

typedef struct {
  int row;
  long long col;
  double val;
} coo_triplet_ll_t;

int compare_triplets_ll(const void *a, const void *b) {
  coo_triplet_ll_t *t1 = (coo_triplet_ll_t *)a;
  coo_triplet_ll_t *t2 = (coo_triplet_ll_t *)b;
  if (t1->row != t2->row)
    return t1->row - t2->row;
  if (t1->col < t2->col)
    return -1;
  if (t1->col > t2->col)
    return 1;
  return 0;
}

void free_compressed_sdp(compressed_sdp_problem_t *prob) {
  if (!prob)
    return;
  if (prob->col_mapping)
    free(prob->col_mapping);
  if (prob->blk_dims)
    free(prob->blk_dims);
  if (prob->blk_ptr)
    free(prob->blk_ptr);
  if (prob->right_hand_side)
    free(prob->right_hand_side);
  if (prob->lp_objective_vector)
    free(prob->lp_objective_vector);

  if (prob->constraint_matrix) {
    if (prob->constraint_matrix->row_ptr)
      free(prob->constraint_matrix->row_ptr);
    if (prob->constraint_matrix->col_ind)
      free(prob->constraint_matrix->col_ind);
    if (prob->constraint_matrix->val)
      free(prob->constraint_matrix->val);
    free(prob->constraint_matrix);
  }
  if (prob->objective_vector_sparse) {
    if (prob->objective_vector_sparse->pos)
      free(prob->objective_vector_sparse->pos);
    if (prob->objective_vector_sparse->val)
      free(prob->objective_vector_sparse->val);
    free(prob->objective_vector_sparse);
  }
  free(prob);
}

// Helper to safely add triplets with symmetry expansion
// Returns the new count of triplets
int add_symmetric_triplets(coo_triplet_t *triplets, int count, int row,
                           int blk_start, int n_k, int r, int c, double v) {
  // 1. Add (r, c)
  // Map to flat index using Column-Major: index = start + c * n + r
  int flat_idx_1 = blk_start + (c * n_k + r);
  triplets[count].row = row;
  triplets[count].col = flat_idx_1;
  triplets[count].val = v;
  count++;

  // 2. If off-diagonal, add symmetric (c, r)
  if (r != c) {
    int flat_idx_2 = blk_start + (r * n_k + c);
    triplets[count].row = row;
    triplets[count].col = flat_idx_2;
    triplets[count].val = v; // Same value for symmetry
    count++;
  }
  return count;
}

static inline int find_compact_idx(const long long *arr, int size,
                                   long long target) {
  int left = 0, right = size - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] == target)
      return mid;
    if (arr[mid] < target)
      left = mid + 1;
    else
      right = mid - 1;
  }
  return -1;
}

compressed_sdp_problem_t *convert_to_compressed(basic_sdp_t *input) {

  compressed_sdp_problem_t *prob =
      (compressed_sdp_problem_t *)safe_malloc(sizeof(compressed_sdp_problem_t));

  // 1. Basic Setup & Dimensions
  prob->num_constraints = input->m;
  prob->n_blks = input->n_cones;
  prob->lp_dim = input->lp_dim;

  prob->blk_dims = (int *)safe_malloc(input->n_cones * sizeof(int));
  prob->blk_ptr =
      (long long *)safe_malloc((input->n_cones + 1) * sizeof(long long));

  size_t total_flat_dim = 0;
  prob->blk_ptr[0] = 0;
  for (int i = 0; i < input->n_cones; i++) {
    prob->blk_dims[i] = input->blk_dims[i];
    total_flat_dim += (size_t)input->blk_dims[i] * input->blk_dims[i];
    prob->blk_ptr[i + 1] = (long long)total_flat_dim;
  }
  prob->total_n_orig = (long long)total_flat_dim;

  // PASS 1: active-set discovery for PSD.
  int max_possible_vars = 2 * input->nnz_psd_constr;
  long long *temp_col_mapping =
      (long long *)safe_malloc(max_possible_vars * sizeof(long long));
  int active_vars_count = 0;

#define CHECK_AND_MAP(blk, r, c)                                               \
  do {                                                                         \
    if (blk < prob->n_blks) {                                                  \
      long long start = prob->blk_ptr[blk];                                    \
      long long n_k = prob->blk_dims[blk];                                     \
      long long g_idx = start + (long long)(c) * n_k + (r);                    \
      temp_col_mapping[active_vars_count++] = g_idx;                           \
      if ((r) != (c)) {                                                        \
        long long g_sym = start + (long long)(r) * n_k + (c);                  \
        temp_col_mapping[active_vars_count++] = g_sym;                         \
      }                                                                        \
    }                                                                          \
  } while (0)

  // Scan PSD Constraints
  for (int k = 0; k < input->nnz_psd_constr; k++) {
    CHECK_AND_MAP(input->psd_cone_constraints->cone_ind[k],
                  input->psd_cone_constraints->row_ind[k],
                  input->psd_cone_constraints->col_ind[k]);
  }

  qsort(temp_col_mapping, active_vars_count, sizeof(long long), cmp_ll_asc);

  int unique_count = 0;
  if (active_vars_count > 0) {
    unique_count = 1;
    for (int i = 1; i < active_vars_count; i++) {
      if (temp_col_mapping[i] != temp_col_mapping[i - 1]) {
        temp_col_mapping[unique_count++] = temp_col_mapping[i];
      }
    }
  }

  prob->lp_start_idx = unique_count;
  prob->n_active_vars = unique_count + prob->lp_dim;

  prob->col_mapping =
      (long long *)safe_malloc(prob->n_active_vars * sizeof(long long));
  if (unique_count > 0) {
    memcpy(prob->col_mapping, temp_col_mapping,
           unique_count * sizeof(long long));
  }

  for (int i = 0; i < prob->lp_dim; i++) {
    prob->col_mapping[prob->lp_start_idx + i] = prob->total_n_orig + i;
  }
  free(temp_col_mapping);

  // PASS 2: build structures via direct lookup.

  // --- Build A (CSR) for both PSD and LP ---
  // Symmetric expansion can push the count over INT_MAX even when each
  // individual nnz field fits in int. Use size_t for the global estimate +
  // the running counter; the per-rank counts (after partition_problem) are
  // partitioned by row count P_row >= 1 and stay safely under INT_MAX.
  size_t nnz_A_est = (size_t)2 * (size_t)input->nnz_psd_constr +
                     (size_t)input->nnz_lp_constr;
  coo_triplet_t *temp_A =
      (coo_triplet_t *)safe_malloc(nnz_A_est * sizeof(coo_triplet_t));
  size_t actual_nnz_A = 0;

  for (int k = 0; k < input->nnz_psd_constr; k++) {
    int blk = input->psd_cone_constraints->cone_ind[k];
    if (blk >= prob->n_blks)
      continue;

    int r = input->psd_cone_constraints->row_ind[k];
    int c = input->psd_cone_constraints->col_ind[k];
    int constr_id = input->psd_cone_constraints->constr_ind[k];
    double val = input->psd_cone_constraints->val[k];

    long long start = prob->blk_ptr[blk];
    long long n_k = prob->blk_dims[blk];

    // 1. Original (r, c)
    long long global_idx = start + (long long)c * n_k + r;
    int compact_idx =
        find_compact_idx(prob->col_mapping, unique_count, global_idx);

    temp_A[actual_nnz_A].row = constr_id;
    temp_A[actual_nnz_A].col = compact_idx;
    temp_A[actual_nnz_A].val = val;
    actual_nnz_A++;

    // 2. Symmetric (c, r)
    if (r != c) {
      long long global_sym = start + (long long)r * n_k + c;
      int compact_sym =
          find_compact_idx(prob->col_mapping, unique_count, global_sym);

      temp_A[actual_nnz_A].row = constr_id;
      temp_A[actual_nnz_A].col = compact_sym;
      temp_A[actual_nnz_A].val = val;
      actual_nnz_A++;
    }
  }

  for (int k = 0; k < input->nnz_lp_constr; k++) {
    int constr_id = input->lp_constraints->row_ind[k];
    int lp_var_idx = input->lp_constraints->col_ind[k];
    double val = input->lp_constraints->val[k];

    int compact_idx = prob->lp_start_idx + lp_var_idx;

    temp_A[actual_nnz_A].row = constr_id;
    temp_A[actual_nnz_A].col = compact_idx;
    temp_A[actual_nnz_A].val = val;
    actual_nnz_A++;
  }

  qsort(temp_A, actual_nnz_A, sizeof(coo_triplet_t), compare_triplets);

  sparse_csr_matrix_t *A =
      (sparse_csr_matrix_t *)safe_malloc(sizeof(sparse_csr_matrix_t));
  A->num_rows = prob->num_constraints;
  A->num_cols = prob->n_active_vars;
  A->row_ptr = (int *)calloc(prob->num_constraints + 1, sizeof(int));
  A->col_ind = (int *)safe_malloc(actual_nnz_A * sizeof(int));
  A->val = (double *)safe_malloc(actual_nnz_A * sizeof(double));

  size_t current_nnz = 0;
  for (size_t i = 0; i < actual_nnz_A; i++) {
    int r = temp_A[i].row;
    int c = temp_A[i].col;
    double v = temp_A[i].val;

    if (i > 0 && temp_A[i - 1].row == r && temp_A[i - 1].col == c) {
      A->val[current_nnz - 1] += v;
    } else {
      A->col_ind[current_nnz] = c;
      A->val[current_nnz] = v;
      A->row_ptr[r + 1]++;
      current_nnz++;
    }
  }
  for (int i = 0; i < prob->num_constraints; i++)
    A->row_ptr[i + 1] += A->row_ptr[i];
  // num_nonzeros remains `int` in the struct; assert the final count fits.
  // (For N>=130 the global CSR no longer fits int and downstream widening of
  // num_nonzeros / row_ptr would be required; for now hard-fail clearly.)
  if (current_nnz > (size_t)INT_MAX) {
    fprintf(stderr,
            "Fatal: global CSR num_nonzeros=%zu exceeds INT_MAX. "
            "Widen sparse_csr_matrix_t.num_nonzeros / row_ptr to long long, "
            "or partition the problem more aggressively before CSR build.\n",
            current_nnz);
    exit(EXIT_FAILURE);
  }
  A->num_nonzeros = (int)current_nnz;
  prob->constraint_matrix = A;
  free(temp_A);

  // PASS 3: build objective vector C for PSD.
  int nnz_C_est = 2 * input->nnz_psd_obj;
  coo_triplet_ll_t *temp_C =
      (coo_triplet_ll_t *)safe_malloc(nnz_C_est * sizeof(coo_triplet_ll_t));
  int actual_nnz_C = 0;

  for (int k = 0; k < input->nnz_psd_obj; k++) {
    int blk = input->psd_cone_objective->cone_ind[k];
    int r = input->psd_cone_objective->row_ind[k];
    int c = input->psd_cone_objective->col_ind[k];
    double val = input->psd_cone_objective->val[k];
    long long start = prob->blk_ptr[blk];
    long long n_k = prob->blk_dims[blk];

    long long g_idx = start + (long long)c * n_k + r;
    temp_C[actual_nnz_C].col = g_idx;
    temp_C[actual_nnz_C].val = val;
    temp_C[actual_nnz_C].row = 0;
    actual_nnz_C++;

    if (r != c) {
      long long g_sym = start + (long long)r * n_k + c;
      temp_C[actual_nnz_C].col = g_sym;
      temp_C[actual_nnz_C].val = val;
      temp_C[actual_nnz_C].row = 0;
      actual_nnz_C++;
    }
  }
  qsort(temp_C, actual_nnz_C, sizeof(coo_triplet_ll_t), compare_triplets_ll);

  prob->objective_vector_sparse =
      (sparse_vector_t *)safe_malloc(sizeof(sparse_vector_t));
  prob->objective_vector_sparse->pos =
      (long long *)safe_malloc(actual_nnz_C * sizeof(long long));
  prob->objective_vector_sparse->val =
      (double *)safe_malloc(actual_nnz_C * sizeof(double));

  int c_count = 0;
  for (int i = 0; i < actual_nnz_C; i++) {
    long long idx = temp_C[i].col;
    double v = temp_C[i].val;
    if (i > 0 && temp_C[i - 1].col == idx) {
      prob->objective_vector_sparse->val[c_count - 1] += v;
    } else {
      prob->objective_vector_sparse->pos[c_count] = idx;
      prob->objective_vector_sparse->val[c_count] = v;
      c_count++;
    }
  }
  prob->objective_vector_sparse->len = c_count;
  free(temp_C);

  prob->lp_objective_vector =
      (double *)calloc((prob->lp_dim > 0 ? prob->lp_dim : 1), sizeof(double));
  if (input->lp_objective && prob->lp_dim > 0) {
    memcpy(prob->lp_objective_vector, input->lp_objective,
           prob->lp_dim * sizeof(double));
  }

  // --- Dense Allocations (RHS) ---
  prob->right_hand_side =
      (double *)calloc(prob->num_constraints, sizeof(double));
  if (input->right_hand_side)
    memcpy(prob->right_hand_side, input->right_hand_side,
           prob->num_constraints * sizeof(double));

  prob->constraint_matrix_t = NULL;
  return prob;
}

int compute_theoretical_max_rank(int m) {
  if (m <= 0)
    return 1;
  int r = (int)ceil((sqrt(8.0 * m + 1.0) - 1.0) / 2.0);
  return (r < 1) ? 1 : r;
}

void compute_per_block_max_rank(const compressed_sdp_problem_t *prob,
                                int user_cap, int *out_per_blk) {
  int n_blks = prob->n_blks;
  if (n_blks <= 0)
    return;

  int n_act = prob->n_active_vars;
  int *var_to_blk = (int *)safe_malloc((n_act > 0 ? n_act : 1) * sizeof(int));
  int b_cursor = 0;
  for (int i = 0; i < n_act; i++) {
    long long g = prob->col_mapping[i];
    if (g >= prob->total_n_orig) {
      var_to_blk[i] = -1;
      continue;
    }
    while (b_cursor < n_blks && g >= prob->blk_ptr[b_cursor + 1])
      b_cursor++;
    var_to_blk[i] = (b_cursor < n_blks) ? b_cursor : -1;
  }
  int *m_per_blk = (int *)safe_calloc(n_blks, sizeof(int));
  int *stamp = (int *)safe_calloc(n_blks, sizeof(int));
  int cur_stamp = 0;
  const sparse_csr_matrix_t *A = prob->constraint_matrix;
  for (int r = 0; r < A->num_rows; r++) {
    cur_stamp++;
    for (int p = A->row_ptr[r]; p < A->row_ptr[r + 1]; p++) {
      int c = A->col_ind[p];
      if (c < 0 || c >= n_act)
        continue;
      int b = var_to_blk[c];
      if (b < 0)
        continue;
      if (stamp[b] != cur_stamp) {
        stamp[b] = cur_stamp;
        m_per_blk[b]++;
      }
    }
  }

  for (int b = 0; b < n_blks; b++) {
    int r = compute_theoretical_max_rank(m_per_blk[b]);
    if (user_cap > 0 && r > user_cap)
      r = user_cap;
    out_per_blk[b] = r;
  }

  free(stamp);
  free(m_per_blk);
  free(var_to_blk);
}

int compute_initial_rank(int m) {
  if (m <= 1)
    return 1;
  int r = (int)ceil(2.0 * log((double)m));
  return (r < 1) ? 1 : r;
}

double
compute_initial_penalty_coef(const compressed_sdp_problem_t *sdp_problem) {
  double sum_dims = 0.0;
  for (int i = 0; i < sdp_problem->n_blks; i++) {
    sum_dims += (double)sdp_problem->blk_dims[i];
  }
  sum_dims += (double)sdp_problem->lp_dim;
  if (sum_dims <= 0.0) {
    return 1.0;
  }
  return 2.0 / sqrt(sum_dims);
}

void set_default_parameters(cardal_parameters_t *params) {
  params->termination_criteria.eps_feasible_relative = 1e-4;
  params->termination_criteria.eps_optimal_relative = 1e-4;
  params->termination_criteria.time_sec_limit = 3600.0;
  params->initial_rank = -1;
  params->max_rank = -1;
  params->termination_criteria.iteration_limit = 20000000;
  params->inner_iterations_limit = 30000;
  params->initial_penalty_coef = -1.0;
  params->penalty_factor = 1.2;
  params->max_penalty_coef = 5e5;
  params->lbfgs_history_size = 5;
  params->verbose = 2;
  params->grid_size.row_dims = 0;
  params->grid_size.rank_dims = 0;
  params->grid_size.cone_dims = 0;
  params->grid_size.decided = false;
  params->l_inf_ruiz_iterations = 10;
  params->has_pock_chambolle_alpha = true;
  params->pock_chambolle_alpha = 1.0;
  params->bound_objective_rescaling = true;
  params->psd_scale_mode = PSD_SCALE_MODE_PER_ELEMENT;
  params->shuffle_mode = SHUFFLE_COL_LOCALITY;
}

static double pythag(double a, double b) {
  double absa = fabs(a), absb = fabs(b);
  if (absa > absb)
    return absa * sqrt(1.0 + (absb / absa) * (absb / absa));
  else
    return (absb == 0.0 ? 0.0
                        : absb * sqrt(1.0 + (absa / absb) * (absa / absb)));
}

int tridiagonal_eigen_solver(int n, double *d, double *e, double *z) {
  int m, l, iter, i, k;
  double s, r, p, g, f, c;

  for (i = 1; i < n; i++)
    e[i - 1] = e[i];
  e[n - 1] = 0.0;

  for (l = 0; l < n; l++) {
    iter = 0;
    do {
      for (m = l; m < n - 1; m++) {
        double dd = fabs(d[m]) + fabs(d[m + 1]);
        if (fabs(e[m]) + dd == dd)
          break;
      }
      if (m != l) {
        if (iter++ == 30)
          return -1;

        g = (d[l + 1] - d[l]) / (2.0 * e[l]);
        r = pythag(g, 1.0);
        g = d[m] - d[l] + e[l] / (g + copysign(r, g));
        s = 1.0;
        c = 1.0;
        p = 0.0;

        for (i = m - 1; i >= l; i--) {
          f = s * e[i];
          double b = c * e[i];
          e[i + 1] = (r = pythag(f, g));
          if (r == 0.0) {
            d[i + 1] -= p;
            e[m] = 0.0;
            break;
          }
          s = f / r;
          c = g / r;
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2.0 * c * b;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - b;

          for (k = 0; k < n; k++) {
            f = z[k + (i + 1) * n];
            z[k + (i + 1) * n] = s * z[k + i * n] + c * f;
            z[k + i * n] = c * z[k + i * n] - s * f;
          }
        }
        if (r == 0.0 && i >= l)
          continue;
        d[l] -= p;
        e[l] = g;
        e[m] = 0.0;
      }
    } while (m != l);
  }
  return 0;
}

void print_header(void) {
  printf("%s\n", CARDAL_LOG_RULE);
  printf("          runtime          |     objective     |    relative "
         "residuals   |       detail       \n");
  printf(" iter  inner  step   time  |   pobj      dobj  |   pres    dres   "
         "  gap  |  rank   rho   iter/s\n");
  printf("%s\n", CARDAL_LOG_RULE);
}

void print_log_entry(const cardal_sdp_solver_state_t *state,
                     int inner_this_iter) {
  double iter_per_sec =
      (state->cumulative_time_sec > 0)
          ? (double)state->num_inner_iteration / state->cumulative_time_sec
          : 0.0;

  char dres_buf[16];
  if (state->dual_residual_evaluated)
    snprintf(dres_buf, sizeof(dres_buf), "%.1e", state->relative_dual_residual);
  else
    snprintf(dres_buf, sizeof(dres_buf), " ------");

  printf("%5d %6d %5d %.1e | %+8.1e %+8.1e | %.1e %7s %.1e | %4d %.1e %.1e\n",
         state->num_outer_iteration + 1, state->num_inner_iteration,
         inner_this_iter, state->cumulative_time_sec, state->primal_objective,
         state->dual_objective, state->relative_primal_residual, dres_buf,
         state->relative_objective_gap, state->total_rank, state->penalty_coef,
         iter_per_sec);
}

