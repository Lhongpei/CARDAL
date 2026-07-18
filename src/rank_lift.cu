/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "rank_lift.h"
#include "sdp_op.h"
#include "solver_state.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef USE_MPI
#include "distribution_utils.h"
#include <mpi.h>
#include <nccl.h>
#endif

namespace {

constexpr double kRankLiftTolerance = 1e-8;
constexpr int kRankLiftMaxIterations = 1000;
constexpr int kRankLiftDirectionCap = 4;

struct LocalBlock {
  int block_index;
  int direction_offset;
  int direction_count;
  int variable_offset;
  int variable_count;
  double *d_U;
};

static int packed_dim(int r) { return r * (r + 1) / 2; }

static void unpack_symmetric(const double *packed, int r, double *matrix) {
  std::fill(matrix, matrix + (size_t)r * r, 0.0);
  for (int i = 0; i < r; i++)
    matrix[(size_t)i * r + i] = packed[i];
  int p = r;
  for (int i = 0; i < r; i++) {
    for (int j = i + 1; j < r; j++) {
      matrix[(size_t)i * r + j] = packed[p];
      matrix[(size_t)j * r + i] = packed[p];
      p++;
    }
  }
}

static void pack_symmetric(const double *matrix, int r, double *packed) {
  for (int i = 0; i < r; i++)
    packed[i] = matrix[(size_t)i * r + i];
  int p = r;
  for (int i = 0; i < r; i++)
    for (int j = i + 1; j < r; j++)
      packed[p++] = matrix[(size_t)i * r + j];
}

// Cyclic Jacobi for the small dense matrices used by the rank-lift backend.
// Input is row-major and destroyed; eigenvectors are returned column-major.
static void jacobi_eigendecomposition(double *matrix, int n, double *eigvals,
                                      double *eigvecs) {
  std::fill(eigvecs, eigvecs + (size_t)n * n, 0.0);
  for (int i = 0; i < n; i++)
    eigvecs[i + (size_t)i * n] = 1.0;
  if (n == 1) {
    eigvals[0] = matrix[0];
    return;
  }

  for (int sweep = 0; sweep < 100; sweep++) {
    double off_sq = 0.0;
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++)
        off_sq += matrix[(size_t)i * n + j] *
                  matrix[(size_t)i * n + j];
    if (off_sq <= 1e-28)
      break;

    for (int p = 0; p < n; p++) {
      for (int q = p + 1; q < n; q++) {
        double apq = matrix[(size_t)p * n + q];
        if (fabs(apq) <= 1e-30)
          continue;
        double app = matrix[(size_t)p * n + p];
        double aqq = matrix[(size_t)q * n + q];
        double phi = 0.5 * atan2(2.0 * apq, aqq - app);
        double c = cos(phi);
        double s = sin(phi);

        for (int k = 0; k < n; k++) {
          double akp = matrix[(size_t)k * n + p];
          double akq = matrix[(size_t)k * n + q];
          matrix[(size_t)k * n + p] = c * akp - s * akq;
          matrix[(size_t)k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; k++) {
          double apk = matrix[(size_t)p * n + k];
          double aqk = matrix[(size_t)q * n + k];
          matrix[(size_t)p * n + k] = c * apk - s * aqk;
          matrix[(size_t)q * n + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; k++) {
          double vkp = eigvecs[k + (size_t)p * n];
          double vkq = eigvecs[k + (size_t)q * n];
          eigvecs[k + (size_t)p * n] = c * vkp - s * vkq;
          eigvecs[k + (size_t)q * n] = s * vkp + c * vkq;
        }
      }
    }
  }
  for (int i = 0; i < n; i++)
    eigvals[i] = matrix[(size_t)i * n + i];
}

static void project_psd_blocks(std::vector<double> &x,
                               const std::vector<int> &block_sizes,
                               double tolerance) {
  int offset = 0;
  for (int r : block_sizes) {
    int p = packed_dim(r);
    std::vector<double> matrix((size_t)r * r);
    std::vector<double> work((size_t)r * r);
    std::vector<double> eigvals(r);
    std::vector<double> eigvecs((size_t)r * r);
    unpack_symmetric(x.data() + offset, r, matrix.data());
    work = matrix;
    jacobi_eigendecomposition(work.data(), r, eigvals.data(),
                              eigvecs.data());
    std::fill(matrix.begin(), matrix.end(), 0.0);
    double max_eval = 0.0;
    for (double value : eigvals)
      max_eval = std::max(max_eval, value);
    double cutoff = tolerance * std::max(1.0, max_eval);
    for (int e = 0; e < r; e++) {
      if (eigvals[e] <= cutoff)
        continue;
      for (int i = 0; i < r; i++)
        for (int j = 0; j < r; j++)
          matrix[(size_t)i * r + j] +=
              eigvals[e] * eigvecs[i + (size_t)e * r] *
              eigvecs[j + (size_t)e * r];
    }
    pack_symmetric(matrix.data(), r, x.data() + offset);
    offset += p;
  }
}

static double gram_spectral_upper_bound(const std::vector<double> &gram,
                                        int n) {
  if (n <= 0)
    return 0.0;
  double bound = 0.0;
  for (int i = 0; i < n; i++) {
    double row_sum = 0.0;
    for (int j = 0; j < n; j++)
      row_sum += fabs(gram[(size_t)i * n + j]);
    bound = std::max(bound, row_sum);
  }
  return bound;
}

static bool solve_shared_scalar(const std::vector<double> &linear,
                                const std::vector<double> &gram, double rho,
                                double tolerance, bool psd_layout,
                                const std::vector<int> &block_sizes,
                                double *shared_t) {
  int n = (int)linear.size();
  std::vector<int> active;
  if (psd_layout) {
    int offset = 0;
    for (int r : block_sizes) {
      for (int j = 0; j < r; j++)
        active.push_back(offset + j);
      offset += packed_dim(r);
    }
  } else {
    active.resize(n);
    for (int i = 0; i < n; i++)
      active[i] = i;
  }

  double numerator = 0.0;
  for (int i : active)
    numerator -= linear[i];
  double denominator = 0.0;
  for (int i : active)
    for (int j : active)
      denominator += gram[(size_t)i * n + j];
  if (numerator <= 0.0 || denominator <= tolerance || rho <= 0.0)
    return false;
  *shared_t = sqrt(numerator / (rho * denominator));
  return std::isfinite(*shared_t);
}

static bool solve_projected_problem(const std::vector<double> &linear,
                                    const std::vector<double> &gram,
                                    double rho, int max_iterations,
                                    double tolerance, bool psd_blocks,
                                    const std::vector<int> &block_sizes,
                                    std::vector<double> &solution) {
  int n = (int)linear.size();
  solution.assign(n, 0.0);
  if (n == 0 || rho <= 0.0)
    return false;

  double lambda_max = gram_spectral_upper_bound(gram, n);
  if (!(lambda_max > 0.0) || !std::isfinite(lambda_max))
    return false;
  double step = 1.0 / (rho * lambda_max);
  std::vector<double> extrapolated(n, 0.0);
  std::vector<double> next(n, 0.0);
  std::vector<double> inverse_metric(n, 1.0);
  if (psd_blocks) {
    int offset = 0;
    for (int r : block_sizes) {
      for (int j = r; j < packed_dim(r); j++)
        inverse_metric[offset + j] = 0.5;
      offset += packed_dim(r);
    }
  }
  double momentum = 1.0;
  double linear_scale = 1.0;
  for (double value : linear)
    linear_scale = std::max(linear_scale, fabs(value));

  for (int iter = 0; iter < max_iterations; iter++) {
    for (int i = 0; i < n; i++) {
      double gradient = linear[i];
      for (int j = 0; j < n; j++)
        gradient += rho * gram[(size_t)i * n + j] * extrapolated[j];
      next[i] =
          extrapolated[i] - step * inverse_metric[i] * gradient;
      if (!std::isfinite(next[i]))
        return false;
    }
    if (psd_blocks)
      project_psd_blocks(next, block_sizes, tolerance);
    else
      for (double &value : next)
        value = std::max(0.0, value);

    double diff_inf = 0.0;
    double sol_inf = 0.0;
    for (int i = 0; i < n; i++) {
      diff_inf = std::max(diff_inf, fabs(next[i] - solution[i]));
      sol_inf = std::max(sol_inf, fabs(next[i]));
    }
    if (diff_inf <= tolerance * (1.0 + sol_inf + linear_scale)) {
      solution = next;
      return true;
    }

    double next_momentum = 0.5 * (1.0 + sqrt(1.0 + 4.0 * momentum * momentum));
    double factor = (momentum - 1.0) / next_momentum;
    for (int i = 0; i < n; i++)
      extrapolated[i] =
          next[i] + factor * (next[i] - solution[i]);
    solution.swap(next);
    momentum = next_momentum;
  }
  return true;
}

static void compute_gram(const std::vector<double> &constraint_columns,
                         int m, int n, std::vector<double> &gram) {
  gram.assign((size_t)n * n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double value = 0.0;
      const double *ci = constraint_columns.data() + (size_t)i * m;
      const double *cj = constraint_columns.data() + (size_t)j * m;
      for (int row = 0; row < m; row++)
        value += ci[row] * cj[row];
      gram[(size_t)i * n + j] = value;
      gram[(size_t)j * n + i] = value;
    }
  }
#ifdef USE_MPI
  // Constraint rows are partitioned only along this communicator.
  // Cone-axis peers already hold identical all-gathered columns.
  // Rank-axis peers other than coordinate zero never enter this routine.
#endif
}

static void allreduce_constraint_gram(cardal_sdp_solver_state_t *state,
                                      std::vector<double> &gram) {
#ifdef USE_MPI
  if (state->grid_context != NULL && state->grid_context->dims[0] > 1) {
    std::vector<double> reduced(gram.size(), 0.0);
    MPI_Allreduce(gram.data(), reduced.data(), (int)gram.size(), MPI_DOUBLE,
                  MPI_SUM, state->grid_context->comm_row);
    gram.swap(reduced);
  }
#else
  (void)state;
  (void)gram;
#endif
}

static void gather_cone_axis(cardal_sdp_solver_state_t *state,
                             const std::vector<double> &local_linear,
                             const std::vector<double> &local_columns,
                             const std::vector<int> &local_block_sizes,
                             std::vector<double> &global_linear,
                             std::vector<double> &global_columns,
                             std::vector<int> &global_block_sizes,
                             int *local_variable_displacement) {
#ifdef USE_MPI
  if (state->grid_context != NULL && state->grid_context->dims[2] > 1) {
    int m = state->num_constraints;
    int peers = state->grid_context->dims[2];
    int local_vars = (int)local_linear.size();
    std::vector<int> var_counts(peers), var_displs(peers);
    MPI_Allgather(&local_vars, 1, MPI_INT, var_counts.data(), 1, MPI_INT,
                  state->grid_context->comm_cone);
    int total_vars = 0;
    for (int i = 0; i < peers; i++) {
      var_displs[i] = total_vars;
      total_vars += var_counts[i];
    }
    *local_variable_displacement =
        var_displs[state->grid_context->coords[2]];
    global_linear.resize(total_vars);
    MPI_Allgatherv(local_linear.data(), local_vars, MPI_DOUBLE,
                   global_linear.data(), var_counts.data(), var_displs.data(),
                   MPI_DOUBLE, state->grid_context->comm_cone);

    std::vector<int> column_counts(peers), column_displs(peers);
    for (int i = 0; i < peers; i++) {
      column_counts[i] = var_counts[i] * m;
      column_displs[i] = var_displs[i] * m;
    }
    global_columns.resize((size_t)m * total_vars);
    MPI_Allgatherv(local_columns.data(), local_vars * m, MPI_DOUBLE,
                   global_columns.data(), column_counts.data(),
                   column_displs.data(), MPI_DOUBLE,
                   state->grid_context->comm_cone);

    int local_blocks = (int)local_block_sizes.size();
    std::vector<int> block_counts(peers), block_displs(peers);
    MPI_Allgather(&local_blocks, 1, MPI_INT, block_counts.data(), 1, MPI_INT,
                  state->grid_context->comm_cone);
    int total_blocks = 0;
    for (int i = 0; i < peers; i++) {
      block_displs[i] = total_blocks;
      total_blocks += block_counts[i];
    }
    global_block_sizes.resize(total_blocks);
    MPI_Allgatherv(local_block_sizes.data(), local_blocks, MPI_INT,
                   global_block_sizes.data(), block_counts.data(),
                   block_displs.data(), MPI_INT,
                   state->grid_context->comm_cone);
    return;
  }
#else
  (void)state;
#endif
  global_linear = local_linear;
  global_columns = local_columns;
  global_block_sizes = local_block_sizes;
  *local_variable_displacement = 0;
}

static void compute_projected_slack(cardal_sdp_solver_state_t *state,
                                    block_low_rank_state_t *blk,
                                    const double *d_V, int r,
                                    std::vector<double> &projected) {
  int dim = blk->dim;
  projected.assign((size_t)r * r, 0.0);
  double *d_out = NULL;
  double *d_scratch = NULL;
  CUDA_CHECK(cudaMalloc(&d_out, (size_t)dim * sizeof(double)));
  if (blk->psd_cone_rescaling != NULL)
    CUDA_CHECK(cudaMalloc(&d_scratch, (size_t)dim * sizeof(double)));
  cusparseDnVecDescr_t vec_in, vec_out;
  CUSPARSE_CHECK(cusparseCreateDnVec(&vec_in, dim, (void *)d_V, CUDA_R_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vec_out, dim, d_out, CUDA_R_64F));
  size_t buffer_size = 0;
  void *buffer = NULL;
  double one = 1.0, zero = 0.0;
  if (blk->matSpS != NULL) {
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
        blk->matSpS, vec_in, &zero, vec_out, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
  }
  if (buffer_size > 0)
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

  for (int j = 0; j < r; j++) {
    double *vj = const_cast<double *>(d_V + (size_t)j * dim);
    CUSPARSE_CHECK(cusparseDnVecSetValues(vec_in, vj));
    unscaled_dual_spmv(state, blk, vj, d_out, d_scratch, vec_in, vec_out,
                       buffer);
#ifdef USE_MPI
    if (state->grid_context != NULL && state->grid_context->dims[0] > 1) {
      NCCL_CHECK(ncclAllReduce(d_out, d_out, dim, ncclDouble, ncclSum,
                               state->grid_context->nccl_row, 0));
    }
#endif
    for (int i = 0; i < r; i++) {
      double value = 0.0;
      CUBLAS_CHECK(cublasDdot(state->blas_handle, dim,
                              d_V + (size_t)i * dim, 1, d_out, 1, &value));
      projected[(size_t)i * r + j] = value;
    }
  }

  if (buffer != NULL)
    CUDA_CHECK(cudaFree(buffer));
  if (d_scratch != NULL)
    CUDA_CHECK(cudaFree(d_scratch));
  CUDA_CHECK(cudaFree(d_out));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vec_in));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vec_out));
}

static int build_qp_or_closed_columns(
    cardal_sdp_solver_state_t *state, const std::vector<LocalBlock> &blocks,
    const std::vector<double> &local_solution, double shared_t,
    bool closed_form, rank_lift_correction_t *correction) {
  int total_added = 0;
  double tolerance = kRankLiftTolerance;
  int solution_offset = 0;
  for (const LocalBlock &meta : blocks) {
    block_low_rank_state_t *blk =
        state->block_low_rank_state[meta.block_index];
    int r = meta.direction_count;
    double max_z = 0.0;
    if (!closed_form)
      for (int j = 0; j < r; j++)
        max_z = std::max(max_z, local_solution[solution_offset + j]);
    double cutoff = tolerance * std::max(1.0, max_z);
    std::vector<int> retained;
    for (int j = 0; j < r; j++) {
      double z = closed_form ? shared_t * shared_t
                             : local_solution[solution_offset + j];
      if (z > cutoff)
        retained.push_back(j);
    }
    solution_offset += r;
    if (retained.empty())
      continue;

    int dim = blk->dim;
    int inc = (int)retained.size();
    double *d_cols = NULL;
    CUDA_CHECK(cudaMalloc(&d_cols, (size_t)dim * inc * sizeof(double)));
    for (int out = 0; out < inc; out++) {
      int j = retained[out];
      double z = closed_form ? shared_t * shared_t
                             : local_solution[solution_offset - r + j];
      double scale = sqrt(z);
      CUBLAS_CHECK(cublasDcopy(state->blas_handle, dim,
                               meta.d_U + (size_t)j * dim, 1,
                               d_cols + (size_t)out * dim, 1));
      CUBLAS_CHECK(cublasDscal(state->blas_handle, dim, &scale,
                               d_cols + (size_t)out * dim, 1));
    }
    correction->rank_incs[meta.block_index] = inc;
    correction->d_columns[meta.block_index] = d_cols;
    total_added += inc;
  }
  return total_added;
}

static int build_sdp_columns(cardal_sdp_solver_state_t *state,
                             const std::vector<LocalBlock> &blocks,
                             const std::vector<double> &local_solution,
                             rank_lift_correction_t *correction) {
  int total_added = 0;
  int solution_offset = 0;
  for (const LocalBlock &meta : blocks) {
    block_low_rank_state_t *blk =
        state->block_low_rank_state[meta.block_index];
    int r = meta.direction_count;
    int p = packed_dim(r);
    std::vector<double> Z((size_t)r * r);
    std::vector<double> work((size_t)r * r);
    std::vector<double> eigvals(r);
    std::vector<double> eigvecs((size_t)r * r);
    unpack_symmetric(local_solution.data() + solution_offset, r, Z.data());
    solution_offset += p;
    work = Z;
    jacobi_eigendecomposition(work.data(), r, eigvals.data(), eigvecs.data());
    double max_eval = 0.0;
    for (double value : eigvals)
      max_eval = std::max(max_eval, value);
    double cutoff =
        kRankLiftTolerance * std::max(1.0, max_eval);
    std::vector<int> retained;
    for (int e = 0; e < r; e++)
      if (eigvals[e] > cutoff)
        retained.push_back(e);
    if (retained.empty())
      continue;

    int inc = (int)retained.size();
    std::vector<double> mix((size_t)r * inc);
    for (int out = 0; out < inc; out++) {
      int e = retained[out];
      double scale = sqrt(eigvals[e]);
      for (int i = 0; i < r; i++)
        mix[i + (size_t)out * r] =
            eigvecs[i + (size_t)e * r] * scale;
    }
    double *d_mix = NULL;
    double *d_cols = NULL;
    CUDA_CHECK(cudaMalloc(&d_mix, (size_t)r * inc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, (size_t)blk->dim * inc * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_mix, mix.data(), (size_t)r * inc * sizeof(double),
                          cudaMemcpyHostToDevice));
    double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasDgemm(state->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             blk->dim, inc, r, &one, meta.d_U, blk->dim,
                             d_mix, r, &zero, d_cols, blk->dim));
    CUDA_CHECK(cudaFree(d_mix));
    correction->rank_incs[meta.block_index] = inc;
    correction->d_columns[meta.block_index] = d_cols;
    total_added += inc;
  }
  return total_added;
}

static int build_random_columns(cardal_sdp_solver_state_t *state,
                                const int *direction_counts,
                                rank_lift_correction_t *correction) {
  int total_added = 0;
  for (int b = 0; b < state->n_blks; b++) {
    int inc = direction_counts[b];
    if (inc <= 0)
      continue;
    int dim = state->block_low_rank_state[b]->dim;
    double *d_columns = NULL;
    CUDA_CHECK(cudaMalloc(&d_columns, (size_t)dim * inc * sizeof(double)));
    unsigned seed =
        (unsigned)(0x9E3779B9u +
                   (unsigned)state->num_outer_iteration * 2654435761u +
                   (unsigned)b * 40503u);
    randomize_device_array(d_columns, dim * inc, seed);
    double noise_scale = 1e-4;
    CUBLAS_CHECK(cublasDscal(state->blas_handle, dim * inc, &noise_scale,
                             d_columns, 1));
    correction->rank_incs[b] = inc;
    correction->d_columns[b] = d_columns;
    total_added += inc;
  }
  return total_added;
}

} // namespace

extern "C" void
initialize_rank_lift_correction(const cardal_sdp_solver_state_t *state,
                                rank_lift_correction_t *correction) {
  correction->rank_incs =
      (int *)safe_calloc((size_t)std::max(1, state->n_blks), sizeof(int));
  correction->d_columns = (double **)safe_calloc(
      (size_t)std::max(1, state->n_blks), sizeof(double *));
}

extern "C" void
free_rank_lift_correction(const cardal_sdp_solver_state_t *state,
                          rank_lift_correction_t *correction) {
  if (correction == NULL)
    return;
  if (correction->d_columns != NULL) {
    for (int b = 0; b < state->n_blks; b++)
      if (correction->d_columns[b] != NULL)
        CUDA_CHECK(cudaFree(correction->d_columns[b]));
  }
  free(correction->rank_incs);
  free(correction->d_columns);
  correction->rank_incs = NULL;
  correction->d_columns = NULL;
}

extern "C" int solve_joint_rank_lift(
    cardal_sdp_solver_state_t *state, const int *direction_counts,
    double *const *d_directions, double *const *eigenvalues,
    rank_lift_correction_t *correction) {
  cublasPointerMode_t saved_mode;
  CUBLAS_CHECK(cublasGetPointerMode(state->blas_handle, &saved_mode));
  CUBLAS_CHECK(
      cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

  if (state->augmentation_mode == AUGMENTATION_MODE_RANDOM) {
    int total_added =
        build_random_columns(state, direction_counts, correction);
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, saved_mode));
    return total_added;
  }

  std::vector<LocalBlock> blocks;
  int local_direction_count = 0;
  int local_variable_count = 0;
  bool use_sdp = (state->augmentation_mode == AUGMENTATION_MODE_SDP);
  for (int b = 0; b < state->n_blks; b++) {
    int r = direction_counts[b];
    if (r > kRankLiftDirectionCap)
      r = kRankLiftDirectionCap;
    if (r <= 0)
      continue;
    LocalBlock meta;
    meta.block_index = b;
    meta.direction_offset = local_direction_count;
    meta.direction_count = r;
    meta.variable_offset = local_variable_count;
    meta.variable_count = use_sdp ? packed_dim(r) : r;
    meta.d_U = NULL;
    local_direction_count += r;
    local_variable_count += meta.variable_count;
    blocks.push_back(meta);
  }

  std::vector<double> local_linear(local_variable_count, 0.0);
  std::vector<double> local_columns(
      (size_t)state->num_constraints * local_variable_count, 0.0);
  std::vector<int> local_block_sizes;

  double *d_column = NULL;
  if (state->num_constraints > 0)
    CUDA_CHECK(cudaMalloc(&d_column,
                          (size_t)state->num_constraints * sizeof(double)));

  for (LocalBlock &meta : blocks) {
    block_low_rank_state_t *blk =
        state->block_low_rank_state[meta.block_index];
    int dim = blk->dim;
    int r = meta.direction_count;
    CUDA_CHECK(cudaMalloc(&meta.d_U, (size_t)dim * r * sizeof(double)));
    if (blk->psd_cone_rescaling != NULL) {
      CUBLAS_CHECK(cublasDdgmm(
          state->blas_handle, CUBLAS_SIDE_LEFT, dim, r,
          d_directions[meta.block_index], dim, blk->psd_cone_rescaling, 1,
          meta.d_U, dim));
    } else {
      CUDA_CHECK(cudaMemcpy(meta.d_U, d_directions[meta.block_index],
                            (size_t)dim * r * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    }

    std::vector<double> H;
    if (use_sdp)
      compute_projected_slack(state, blk, d_directions[meta.block_index], r,
                              H);

    int variable = meta.variable_offset;
    for (int j = 0; j < r; j++) {
      local_linear[variable] =
          use_sdp ? H[(size_t)j * r + j]
                  : eigenvalues[meta.block_index][j];
      compute_rank_lift_A_ww(state, blk, meta.d_U + (size_t)j * dim,
                             d_column);
      CUDA_CHECK(cudaMemcpy(
          local_columns.data() + (size_t)variable * state->num_constraints,
          d_column, (size_t)state->num_constraints * sizeof(double),
          cudaMemcpyDeviceToHost));
      variable++;
    }
    if (use_sdp) {
      for (int i = 0; i < r; i++) {
        for (int j = i + 1; j < r; j++) {
          local_linear[variable] = 2.0 * H[(size_t)i * r + j];
          compute_rank_lift_A_uv(state, blk,
                                 meta.d_U + (size_t)i * dim,
                                 meta.d_U + (size_t)j * dim, d_column);
          CUDA_CHECK(cudaMemcpy(
              local_columns.data() +
                  (size_t)variable * state->num_constraints,
              d_column, (size_t)state->num_constraints * sizeof(double),
              cudaMemcpyDeviceToHost));
          variable++;
        }
      }
      local_block_sizes.push_back(r);
    }
  }
  if (d_column != NULL)
    CUDA_CHECK(cudaFree(d_column));

  std::vector<double> global_linear;
  std::vector<double> global_columns;
  std::vector<int> global_block_sizes;
  int local_variable_displacement = 0;
  gather_cone_axis(state, local_linear, local_columns, local_block_sizes,
                   global_linear, global_columns, global_block_sizes,
                   &local_variable_displacement);

  int global_variables = (int)global_linear.size();
  std::vector<double> gram;
  compute_gram(global_columns, state->num_constraints, global_variables, gram);
  allreduce_constraint_gram(state, gram);

  std::vector<double> global_solution;
  bool closed_form =
      (state->augmentation_mode == AUGMENTATION_MODE_CLOSED_FORM);
  bool solved = false;
  double shared_t = 0.0;
  if (closed_form) {
    solved = solve_shared_scalar(
        global_linear, gram, state->penalty_coef, kRankLiftTolerance,
        false, global_block_sizes, &shared_t);
  } else {
    solved = solve_projected_problem(
        global_linear, gram, state->penalty_coef, kRankLiftMaxIterations,
        kRankLiftTolerance, use_sdp,
        global_block_sizes, global_solution);
    if (!solved) {
      solved = solve_shared_scalar(
          global_linear, gram, state->penalty_coef, kRankLiftTolerance,
          use_sdp, global_block_sizes, &shared_t);
      closed_form = solved;
      if (state->verbose >= 3 && solved)
        printf("  [rank lift] joint solver failed; using shared scalar "
               "closed form\n");
    }
  }

  int total_added = 0;
  if (solved) {
    if (closed_form) {
      std::vector<double> unused;
      total_added = build_qp_or_closed_columns(
          state, blocks, unused, shared_t, true, correction);
    } else {
      std::vector<double> local_solution(
          global_solution.begin() + local_variable_displacement,
          global_solution.begin() + local_variable_displacement +
              local_variable_count);
      if (use_sdp)
        total_added =
            build_sdp_columns(state, blocks, local_solution, correction);
      else
        total_added = build_qp_or_closed_columns(
            state, blocks, local_solution, 0.0, false, correction);
    }
  }

  for (LocalBlock &meta : blocks)
    if (meta.d_U != NULL)
      CUDA_CHECK(cudaFree(meta.d_U));
  CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, saved_mode));
  return total_added;
}
