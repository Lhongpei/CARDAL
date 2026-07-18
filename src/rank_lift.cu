/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "rank_lift.h"
#include "sdp_op.h"
#include "solver_state.h"
#include "utils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#ifdef USE_MPI
#include "distribution_utils.h"
#include <mpi.h>
#include <nccl.h>
#endif

namespace {

constexpr double kRankLiftTolerance = 1e-8;
constexpr int kRankLiftPerConeDirectionCap = 4;
constexpr int kRankLiftGlobalDirectionCap = 32;
constexpr int kQpActiveSetMaxIterations = 256;
constexpr int kSdpWarmStartIterations = 12;
constexpr int kSdpNewtonMaxIterations = 50;

struct LocalBlock {
  int block_index;
  int direction_count;
  int variable_offset;
  int variable_count;
  double *d_U;
};

struct DirectionCandidate {
  double eigenvalue;
  int block_index;
  int direction_index;
};

struct RankLiftSolveInfo {
  int iterations;
  double kkt_residual;
  bool converged;
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
        off_sq += matrix[(size_t)i * n + j] * matrix[(size_t)i * n + j];
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
    jacobi_eigendecomposition(work.data(), r, eigvals.data(), eigvecs.data());
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
          matrix[(size_t)i * r + j] += eigvals[e] * eigvecs[i + (size_t)e * r] *
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

static void dense_matvec(const std::vector<double> &matrix,
                         const std::vector<double> &x,
                         std::vector<double> &result) {
  int n = (int)x.size();
  result.assign(n, 0.0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      result[i] += matrix[(size_t)i * n + j] * x[j];
}

static bool solve_dense_system(std::vector<double> matrix,
                               std::vector<double> rhs,
                               std::vector<double> &solution,
                               double regularization) {
  int n = (int)rhs.size();
  solution.assign(n, 0.0);
  if (n == 0)
    return true;

  double scale = 1.0;
  for (double value : matrix)
    scale = std::max(scale, fabs(value));
  for (int i = 0; i < n; i++)
    matrix[(size_t)i * n + i] += regularization;

  for (int col = 0; col < n; col++) {
    int pivot = col;
    double pivot_abs = fabs(matrix[(size_t)col * n + col]);
    for (int row = col + 1; row < n; row++) {
      double candidate = fabs(matrix[(size_t)row * n + col]);
      if (candidate > pivot_abs) {
        pivot = row;
        pivot_abs = candidate;
      }
    }
    if (pivot_abs <= 1e-14 * scale || !std::isfinite(pivot_abs))
      return false;
    if (pivot != col) {
      for (int j = col; j < n; j++)
        std::swap(matrix[(size_t)col * n + j], matrix[(size_t)pivot * n + j]);
      std::swap(rhs[col], rhs[pivot]);
    }
    double diagonal = matrix[(size_t)col * n + col];
    for (int row = col + 1; row < n; row++) {
      double factor = matrix[(size_t)row * n + col] / diagonal;
      matrix[(size_t)row * n + col] = 0.0;
      for (int j = col + 1; j < n; j++)
        matrix[(size_t)row * n + j] -= factor * matrix[(size_t)col * n + j];
      rhs[row] -= factor * rhs[col];
    }
  }

  for (int row = n - 1; row >= 0; row--) {
    double value = rhs[row];
    for (int j = row + 1; j < n; j++)
      value -= matrix[(size_t)row * n + j] * solution[j];
    double diagonal = matrix[(size_t)row * n + row];
    if (fabs(diagonal) <= 1e-14 * scale || !std::isfinite(diagonal))
      return false;
    solution[row] = value / diagonal;
    if (!std::isfinite(solution[row]))
      return false;
  }
  return true;
}

static bool solve_regularized_system(const std::vector<double> &matrix,
                                     const std::vector<double> &rhs,
                                     std::vector<double> &solution) {
  double scale = 1.0;
  int n = (int)rhs.size();
  for (int i = 0; i < n; i++)
    scale = std::max(scale, fabs(matrix[(size_t)i * n + i]));
  for (int attempt = 0; attempt < 9; attempt++) {
    double regularization =
        attempt == 0 ? 0.0 : scale * 1e-12 * pow(10.0, attempt - 1);
    if (solve_dense_system(matrix, rhs, solution, regularization))
      return true;
  }
  return false;
}

static double qp_kkt_residual(const std::vector<double> &linear,
                              const std::vector<double> &gram, double rho,
                              const std::vector<double> &solution,
                              std::vector<double> *gradient_out = NULL) {
  std::vector<double> gradient;
  dense_matvec(gram, solution, gradient);
  double residual = 0.0;
  double scale = 1.0;
  for (int i = 0; i < (int)solution.size(); i++) {
    if (!std::isfinite(solution[i]) || !std::isfinite(linear[i]) ||
        !std::isfinite(gradient[i]))
      return std::numeric_limits<double>::infinity();
    gradient[i] = linear[i] + rho * gradient[i];
    if (!std::isfinite(gradient[i]))
      return std::numeric_limits<double>::infinity();
    scale = std::max(scale, fabs(linear[i]));
    double component =
        solution[i] > 1e-12 ? fabs(gradient[i]) : std::max(0.0, -gradient[i]);
    residual = std::max(residual, component);
  }
  if (gradient_out != NULL)
    *gradient_out = gradient;
  return residual / scale;
}

static bool solve_active_set_qp(const std::vector<double> &linear,
                                const std::vector<double> &gram, double rho,
                                double tolerance, std::vector<double> &solution,
                                RankLiftSolveInfo *info) {
  int n = (int)linear.size();
  solution.assign(n, 0.0);
  std::vector<unsigned char> free_set(n, 0);
  info->iterations = 0;
  info->kkt_residual = std::numeric_limits<double>::infinity();
  info->converged = false;
  if (n == 0 || rho <= 0.0)
    return false;

  for (int iteration = 0; iteration < kQpActiveSetMaxIterations; iteration++) {
    info->iterations = iteration + 1;
    std::vector<double> gradient;
    info->kkt_residual =
        qp_kkt_residual(linear, gram, rho, solution, &gradient);
    if (!std::isfinite(info->kkt_residual))
      return false;
    if (info->kkt_residual <= tolerance) {
      info->converged = true;
      return true;
    }

    int entering = -1;
    double most_negative = 0.0;
    for (int i = 0; i < n; i++) {
      if (!free_set[i] && gradient[i] < most_negative) {
        most_negative = gradient[i];
        entering = i;
      }
    }
    if (entering < 0)
      break;
    free_set[entering] = 1;

    for (int inner = 0; inner <= n; inner++) {
      std::vector<int> indices;
      for (int i = 0; i < n; i++)
        if (free_set[i])
          indices.push_back(i);
      int p = (int)indices.size();
      std::vector<double> reduced_matrix((size_t)p * p);
      std::vector<double> reduced_rhs(p);
      for (int i = 0; i < p; i++) {
        reduced_rhs[i] = -linear[indices[i]];
        for (int j = 0; j < p; j++)
          reduced_matrix[(size_t)i * p + j] =
              rho * gram[(size_t)indices[i] * n + indices[j]];
      }
      std::vector<double> reduced_solution;
      if (!solve_regularized_system(reduced_matrix, reduced_rhs,
                                    reduced_solution))
        return false;

      std::vector<double> candidate(n, 0.0);
      bool strictly_positive = true;
      for (int i = 0; i < p; i++) {
        candidate[indices[i]] = reduced_solution[i];
        if (reduced_solution[i] <= 0.0)
          strictly_positive = false;
      }
      if (strictly_positive) {
        solution.swap(candidate);
        break;
      }

      double step = 1.0;
      for (int index : indices) {
        if (candidate[index] <= 0.0) {
          double denominator = solution[index] - candidate[index];
          if (denominator > 0.0)
            step = std::min(step, solution[index] / denominator);
          else
            step = 0.0;
        }
      }
      for (int i = 0; i < n; i++)
        solution[i] += step * (candidate[i] - solution[i]);
      for (double value : solution)
        if (!std::isfinite(value))
          return false;
      bool removed = false;
      for (int index : indices) {
        if (solution[index] <= 1e-12) {
          solution[index] = 0.0;
          free_set[index] = 0;
          removed = true;
        }
      }
      if (!removed)
        return false;
    }
  }

  info->kkt_residual = qp_kkt_residual(linear, gram, rho, solution, NULL);
  info->converged = info->kkt_residual <= tolerance;
  return info->converged;
}

struct PsdProjectionDerivative {
  int offset;
  int rank;
  std::vector<double> eigenvalues;
  std::vector<double> eigenvectors;
};

static void
project_psd_and_prepare(const std::vector<double> &input,
                        const std::vector<int> &block_sizes,
                        std::vector<double> &projected,
                        std::vector<PsdProjectionDerivative> *derivatives) {
  projected.assign(input.size(), 0.0);
  if (derivatives != NULL)
    derivatives->clear();
  int offset = 0;
  for (int r : block_sizes) {
    int p = packed_dim(r);
    std::vector<double> matrix((size_t)r * r);
    std::vector<double> work((size_t)r * r);
    std::vector<double> eigenvalues(r);
    std::vector<double> eigenvectors((size_t)r * r);
    unpack_symmetric(input.data() + offset, r, matrix.data());
    work = matrix;
    jacobi_eigendecomposition(work.data(), r, eigenvalues.data(),
                              eigenvectors.data());
    std::fill(matrix.begin(), matrix.end(), 0.0);
    for (int e = 0; e < r; e++) {
      double value = std::max(0.0, eigenvalues[e]);
      for (int i = 0; i < r; i++)
        for (int j = 0; j < r; j++)
          matrix[(size_t)i * r + j] += value * eigenvectors[i + (size_t)e * r] *
                                       eigenvectors[j + (size_t)e * r];
    }
    pack_symmetric(matrix.data(), r, projected.data() + offset);
    if (derivatives != NULL) {
      PsdProjectionDerivative derivative;
      derivative.offset = offset;
      derivative.rank = r;
      derivative.eigenvalues.swap(eigenvalues);
      derivative.eigenvectors.swap(eigenvectors);
      derivatives->push_back(std::move(derivative));
    }
    offset += p;
  }
}

static void apply_psd_projection_derivative(
    const std::vector<PsdProjectionDerivative> &derivatives,
    const std::vector<double> &direction, std::vector<double> &result) {
  result.assign(direction.size(), 0.0);
  for (const PsdProjectionDerivative &derivative : derivatives) {
    int r = derivative.rank;
    std::vector<double> matrix((size_t)r * r);
    std::vector<double> transformed((size_t)r * r, 0.0);
    std::vector<double> projected((size_t)r * r, 0.0);
    unpack_symmetric(direction.data() + derivative.offset, r, matrix.data());

    for (int i = 0; i < r; i++) {
      for (int j = 0; j < r; j++) {
        double value = 0.0;
        for (int a = 0; a < r; a++)
          for (int b = 0; b < r; b++)
            value += derivative.eigenvectors[a + (size_t)i * r] *
                     matrix[(size_t)a * r + b] *
                     derivative.eigenvectors[b + (size_t)j * r];
        double left = derivative.eigenvalues[i];
        double right = derivative.eigenvalues[j];
        double scale = std::max({1.0, fabs(left), fabs(right)});
        double omega = 0.0;
        if (fabs(left - right) > 1e-12 * scale) {
          omega = (std::max(0.0, left) - std::max(0.0, right)) / (left - right);
        } else if (left > 1e-12 * scale) {
          omega = 1.0;
        } else if (left >= -1e-12 * scale) {
          omega = 0.5;
        }
        transformed[(size_t)i * r + j] = omega * value;
      }
    }

    for (int a = 0; a < r; a++)
      for (int b = 0; b < r; b++)
        for (int i = 0; i < r; i++)
          for (int j = 0; j < r; j++)
            projected[(size_t)a * r + b] +=
                derivative.eigenvectors[a + (size_t)i * r] *
                transformed[(size_t)i * r + j] *
                derivative.eigenvectors[b + (size_t)j * r];
    pack_symmetric(projected.data(), r, result.data() + derivative.offset);
  }
}

static double compute_sdp_fixed_point(
    const std::vector<double> &linear, const std::vector<double> &gram,
    double rho, double step, const std::vector<double> &inverse_metric,
    const std::vector<int> &block_sizes, const std::vector<double> &solution,
    std::vector<double> &projected, std::vector<double> &residual,
    std::vector<PsdProjectionDerivative> *derivatives) {
  int n = (int)solution.size();
  std::vector<double> gram_solution;
  dense_matvec(gram, solution, gram_solution);
  std::vector<double> argument(n);
  for (int i = 0; i < n; i++) {
    double gradient = linear[i] + rho * gram_solution[i];
    argument[i] = solution[i] - step * inverse_metric[i] * gradient;
  }
  project_psd_and_prepare(argument, block_sizes, projected, derivatives);
  residual.resize(n);
  double squared_norm = 0.0;
  for (int i = 0; i < n; i++) {
    residual[i] = solution[i] - projected[i];
    squared_norm += residual[i] * residual[i];
  }
  return 0.5 * squared_norm;
}

static bool solve_semismooth_sdp(const std::vector<double> &linear,
                                 const std::vector<double> &gram, double rho,
                                 double tolerance,
                                 const std::vector<int> &block_sizes,
                                 std::vector<double> &solution,
                                 RankLiftSolveInfo *info) {
  int n = (int)linear.size();
  solution.assign(n, 0.0);
  info->iterations = 0;
  info->kkt_residual = std::numeric_limits<double>::infinity();
  info->converged = false;
  if (n == 0 || rho <= 0.0)
    return false;

  double spectral_bound = gram_spectral_upper_bound(gram, n);
  if (!(spectral_bound > 0.0) || !std::isfinite(spectral_bound))
    return false;
  double step = 1.0 / (rho * spectral_bound);
  std::vector<double> inverse_metric(n, 1.0);
  int offset = 0;
  for (int r : block_sizes) {
    for (int j = r; j < packed_dim(r); j++)
      inverse_metric[offset + j] = 0.5;
    offset += packed_dim(r);
  }
  double linear_scale = 1.0;
  for (double value : linear)
    linear_scale = std::max(linear_scale, fabs(value));

  std::vector<double> projected, residual;
  for (int iteration = 0; iteration < kSdpWarmStartIterations; iteration++) {
    double merit = compute_sdp_fixed_point(linear, gram, rho, step,
                                           inverse_metric, block_sizes,
                                           solution, projected, residual, NULL);
    if (!std::isfinite(merit))
      return false;
    solution.swap(projected);
  }

  for (int iteration = 0; iteration < kSdpNewtonMaxIterations; iteration++) {
    info->iterations = iteration + 1;
    std::vector<PsdProjectionDerivative> derivatives;
    double merit = compute_sdp_fixed_point(
        linear, gram, rho, step, inverse_metric, block_sizes, solution,
        projected, residual, &derivatives);
    if (!std::isfinite(merit))
      return false;
    double residual_inf = 0.0;
    for (double value : residual) {
      if (!std::isfinite(value))
        return false;
      residual_inf = std::max(residual_inf, fabs(value));
    }
    info->kkt_residual = residual_inf / (step * linear_scale);
    if (!std::isfinite(info->kkt_residual))
      return false;
    if (info->kkt_residual <= tolerance) {
      info->converged = true;
      return true;
    }

    std::vector<double> jacobian((size_t)n * n);
    std::vector<double> argument_direction(n);
    std::vector<double> projection_direction;
    for (int col = 0; col < n; col++) {
      for (int row = 0; row < n; row++) {
        argument_direction[row] =
            (row == col ? 1.0 : 0.0) -
            step * inverse_metric[row] * rho * gram[(size_t)row * n + col];
      }
      apply_psd_projection_derivative(derivatives, argument_direction,
                                      projection_direction);
      for (int row = 0; row < n; row++)
        jacobian[(size_t)row * n + col] =
            (row == col ? 1.0 : 0.0) - projection_direction[row];
    }

    std::vector<double> rhs(n);
    for (int i = 0; i < n; i++)
      rhs[i] = -residual[i];
    std::vector<double> newton_direction;
    bool have_newton =
        solve_regularized_system(jacobian, rhs, newton_direction);
    bool accepted = false;
    if (have_newton) {
      double line_step = 1.0;
      for (int line_search = 0; line_search < 20; line_search++) {
        std::vector<double> candidate(n);
        for (int i = 0; i < n; i++)
          candidate[i] = solution[i] + line_step * newton_direction[i];
        project_psd_blocks(candidate, block_sizes, 0.0);
        std::vector<double> candidate_projected, candidate_residual;
        double candidate_merit = compute_sdp_fixed_point(
            linear, gram, rho, step, inverse_metric, block_sizes, candidate,
            candidate_projected, candidate_residual, NULL);
        if (std::isfinite(candidate_merit) &&
            candidate_merit <= (1.0 - 1e-4 * line_step) * merit) {
          solution.swap(candidate);
          accepted = true;
          break;
        }
        line_step *= 0.5;
      }
    }
    if (!accepted)
      solution.swap(projected);
  }

  compute_sdp_fixed_point(linear, gram, rho, step, inverse_metric, block_sizes,
                          solution, projected, residual, NULL);
  double residual_inf = 0.0;
  for (double value : residual) {
    if (!std::isfinite(value))
      return false;
    residual_inf = std::max(residual_inf, fabs(value));
  }
  info->kkt_residual = residual_inf / (step * linear_scale);
  info->converged =
      std::isfinite(info->kkt_residual) && info->kkt_residual <= tolerance;
  return info->converged;
}

static void compute_gram_gpu(cardal_sdp_solver_state_t *state,
                             const double *d_constraint_columns, int m, int n,
                             std::vector<double> &gram) {
  gram.assign((size_t)n * n, 0.0);
  if (n == 0 || m == 0)
    return;
  double *d_gram = NULL;
  CUDA_CHECK(cudaMalloc(&d_gram, (size_t)n * n * sizeof(double)));
  double one = 1.0, zero = 0.0;
  CUBLAS_CHECK(cublasDsyrk(state->blas_handle, CUBLAS_FILL_MODE_LOWER,
                           CUBLAS_OP_T, n, m, &one, d_constraint_columns, m,
                           &zero, d_gram, n));
  CUDA_CHECK(cudaMemcpy(gram.data(), d_gram, (size_t)n * n * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_gram));
  // cuBLAS writes column-major lower, which is row-major upper on the host.
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
      gram[(size_t)j * n + i] = gram[(size_t)i * n + j];
}

static bool scale_dense_subproblem(const std::vector<double> &linear,
                                   const std::vector<double> &gram,
                                   bool psd_blocks,
                                   const std::vector<int> &block_sizes,
                                   std::vector<double> &scaled_linear,
                                   std::vector<double> &scaled_gram,
                                   std::vector<double> &variable_scaling) {
  int n = (int)linear.size();
  double max_diagonal = 0.0;
  for (int i = 0; i < n; i++)
    max_diagonal = std::max(max_diagonal, fabs(gram[(size_t)i * n + i]));
  if (!(max_diagonal > 0.0) || !std::isfinite(max_diagonal))
    return false;

  double diagonal_floor = std::max(1e-30, 1e-14 * max_diagonal);
  variable_scaling.assign(n, 1.0);
  if (psd_blocks) {
    int offset = 0;
    for (int r : block_sizes) {
      int p = packed_dim(r);
      double block_diagonal = 0.0;
      for (int i = 0; i < p; i++)
        block_diagonal = std::max(
            block_diagonal, fabs(gram[(size_t)(offset + i) * n + offset + i]));
      double scale = 1.0 / sqrt(std::max(block_diagonal, diagonal_floor));
      scale = std::min(1e12, std::max(1e-12, scale));
      for (int i = 0; i < p; i++)
        variable_scaling[offset + i] = scale;
      offset += p;
    }
  } else {
    for (int i = 0; i < n; i++) {
      double diagonal = fabs(gram[(size_t)i * n + i]);
      double scale = 1.0 / sqrt(std::max(diagonal, diagonal_floor));
      variable_scaling[i] = std::min(1e12, std::max(1e-12, scale));
    }
  }

  scaled_linear.resize(n);
  scaled_gram.resize((size_t)n * n);
  for (int i = 0; i < n; i++) {
    scaled_linear[i] = linear[i] * variable_scaling[i];
    if (!std::isfinite(scaled_linear[i]))
      return false;
    for (int j = 0; j < n; j++)
      scaled_gram[(size_t)i * n + j] =
          gram[(size_t)i * n + j] * variable_scaling[i] * variable_scaling[j];
    for (int j = 0; j < n; j++)
      if (!std::isfinite(scaled_gram[(size_t)i * n + j]))
        return false;
  }
  return true;
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

static void gather_cone_axis(
    cardal_sdp_solver_state_t *state, const std::vector<double> &local_linear,
    const double *d_local_columns, const std::vector<int> &local_block_sizes,
    std::vector<double> &global_linear, double **d_global_columns,
    std::vector<int> &global_block_sizes, int *local_variable_displacement) {
  *d_global_columns = NULL;
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
    *local_variable_displacement = var_displs[state->grid_context->coords[2]];
    global_linear.resize(total_vars);
    MPI_Allgatherv(local_linear.data(), local_vars, MPI_DOUBLE,
                   global_linear.data(), var_counts.data(), var_displs.data(),
                   MPI_DOUBLE, state->grid_context->comm_cone);

    std::vector<int> column_counts(peers), column_displs(peers);
    for (int i = 0; i < peers; i++) {
      column_counts[i] = var_counts[i] * m;
      column_displs[i] = var_displs[i] * m;
    }
    std::vector<double> local_columns((size_t)m * local_vars);
    std::vector<double> global_columns((size_t)m * total_vars);
    if (local_vars > 0)
      CUDA_CHECK(cudaMemcpy(local_columns.data(), d_local_columns,
                            (size_t)m * local_vars * sizeof(double),
                            cudaMemcpyDeviceToHost));
    MPI_Allgatherv(local_columns.data(), local_vars * m, MPI_DOUBLE,
                   global_columns.data(), column_counts.data(),
                   column_displs.data(), MPI_DOUBLE,
                   state->grid_context->comm_cone);
    if (total_vars > 0) {
      CUDA_CHECK(cudaMalloc(d_global_columns,
                            (size_t)m * total_vars * sizeof(double)));
      CUDA_CHECK(cudaMemcpy(*d_global_columns, global_columns.data(),
                            (size_t)m * total_vars * sizeof(double),
                            cudaMemcpyHostToDevice));
    }

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
  global_block_sizes = local_block_sizes;
  *local_variable_displacement = 0;
  if (!local_linear.empty())
    *d_global_columns = const_cast<double *>(d_local_columns);
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
      CUBLAS_CHECK(cublasDdot(state->blas_handle, dim, d_V + (size_t)i * dim, 1,
                              d_out, 1, &value));
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

static int build_qp_or_closed_columns(cardal_sdp_solver_state_t *state,
                                      const std::vector<LocalBlock> &blocks,
                                      const std::vector<double> &local_solution,
                                      double shared_t, bool closed_form,
                                      rank_lift_correction_t *correction) {
  int total_added = 0;
  double tolerance = kRankLiftTolerance;
  int solution_offset = 0;
  for (const LocalBlock &meta : blocks) {
    block_low_rank_state_t *blk = state->block_low_rank_state[meta.block_index];
    int r = meta.direction_count;
    double max_z = 0.0;
    if (!closed_form)
      for (int j = 0; j < r; j++)
        max_z = std::max(max_z, local_solution[solution_offset + j]);
    double cutoff = tolerance * max_z;
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
    block_low_rank_state_t *blk = state->block_low_rank_state[meta.block_index];
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
    double cutoff = kRankLiftTolerance * max_eval;
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
        mix[i + (size_t)out * r] = eigvecs[i + (size_t)e * r] * scale;
    }
    double *d_mix = NULL;
    double *d_cols = NULL;
    CUDA_CHECK(cudaMalloc(&d_mix, (size_t)r * inc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, (size_t)blk->dim * inc * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_mix, mix.data(), (size_t)r * inc * sizeof(double),
                          cudaMemcpyHostToDevice));
    double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasDgemm(state->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             blk->dim, inc, r, &one, meta.d_U, blk->dim, d_mix,
                             r, &zero, d_cols, blk->dim));
    CUDA_CHECK(cudaFree(d_mix));
    correction->rank_incs[meta.block_index] = inc;
    correction->d_columns[meta.block_index] = d_cols;
    total_added += inc;
  }
  return total_added;
}

static int select_strongest_directions(
    cardal_sdp_solver_state_t *state, const int *direction_counts,
    double *const *d_directions, double *const *eigenvalues,
    std::vector<int> &selected_counts,
    std::vector<double *> &selected_directions,
    std::vector<double *> &selected_eigenvalues) {
  std::vector<DirectionCandidate> candidates;
  for (int b = 0; b < state->n_blks; b++) {
    std::vector<int> indices(direction_counts[b]);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int left, int right) {
      if (eigenvalues[b][left] != eigenvalues[b][right])
        return eigenvalues[b][left] < eigenvalues[b][right];
      return left < right;
    });
    int retained = std::min((int)indices.size(), kRankLiftPerConeDirectionCap);
    for (int j = 0; j < retained; j++) {
      DirectionCandidate candidate;
      candidate.eigenvalue = eigenvalues[b][indices[j]];
      candidate.block_index = b;
      candidate.direction_index = indices[j];
      candidates.push_back(candidate);
    }
  }

  std::vector<double> local_values(candidates.size());
  for (int i = 0; i < (int)candidates.size(); i++)
    local_values[i] = candidates[i].eigenvalue;
  std::vector<double> global_values = local_values;
  int local_displacement = 0;
#ifdef USE_MPI
  if (state->grid_context != NULL && state->grid_context->dims[2] > 1) {
    int peers = state->grid_context->dims[2];
    int local_count = (int)local_values.size();
    std::vector<int> counts(peers), displacements(peers);
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                  state->grid_context->comm_cone);
    int total = 0;
    for (int peer = 0; peer < peers; peer++) {
      displacements[peer] = total;
      total += counts[peer];
    }
    local_displacement = displacements[state->grid_context->coords[2]];
    global_values.resize(total);
    MPI_Allgatherv(local_values.data(), local_count, MPI_DOUBLE,
                   global_values.data(), counts.data(), displacements.data(),
                   MPI_DOUBLE, state->grid_context->comm_cone);
  }
#endif

  std::vector<int> order(global_values.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int left, int right) {
    if (global_values[left] != global_values[right])
      return global_values[left] < global_values[right];
    return left < right;
  });
  int global_retained =
      std::min((int)order.size(), kRankLiftGlobalDirectionCap);
  std::vector<unsigned char> keep_local(candidates.size(), 0);
  int local_end = local_displacement + (int)candidates.size();
  for (int position = 0; position < global_retained; position++) {
    int global_index = order[position];
    if (global_index >= local_displacement && global_index < local_end)
      keep_local[global_index - local_displacement] = 1;
  }

  selected_counts.assign(state->n_blks, 0);
  selected_directions.assign(state->n_blks, NULL);
  selected_eigenvalues.assign(state->n_blks, NULL);
  std::vector<std::vector<const DirectionCandidate *>> by_block(state->n_blks);
  for (int i = 0; i < (int)candidates.size(); i++)
    if (keep_local[i])
      by_block[candidates[i].block_index].push_back(&candidates[i]);

  for (int b = 0; b < state->n_blks; b++) {
    auto &block_candidates = by_block[b];
    std::sort(
        block_candidates.begin(), block_candidates.end(),
        [](const DirectionCandidate *left, const DirectionCandidate *right) {
          if (left->eigenvalue != right->eigenvalue)
            return left->eigenvalue < right->eigenvalue;
          return left->direction_index < right->direction_index;
        });
    int count = (int)block_candidates.size();
    if (count == 0)
      continue;
    int dim = state->block_low_rank_state[b]->dim;
    selected_counts[b] = count;
    selected_eigenvalues[b] =
        (double *)safe_malloc((size_t)count * sizeof(double));
    CUDA_CHECK(cudaMalloc(&selected_directions[b],
                          (size_t)dim * count * sizeof(double)));
    for (int j = 0; j < count; j++) {
      int source = block_candidates[j]->direction_index;
      selected_eigenvalues[b][j] = block_candidates[j]->eigenvalue;
      CUDA_CHECK(cudaMemcpy(selected_directions[b] + (size_t)j * dim,
                            d_directions[b] + (size_t)source * dim,
                            (size_t)dim * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    }
  }
  return global_retained;
}

static void
free_selected_directions(std::vector<double *> &selected_directions,
                         std::vector<double *> &selected_eigenvalues) {
  for (double *directions : selected_directions)
    if (directions != NULL)
      CUDA_CHECK(cudaFree(directions));
  for (double *values : selected_eigenvalues)
    free(values);
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
    CUBLAS_CHECK(
        cublasDscal(state->blas_handle, dim * inc, &noise_scale, d_columns, 1));
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

extern "C" int solve_joint_rank_lift(cardal_sdp_solver_state_t *state,
                                     const int *direction_counts,
                                     double *const *d_directions,
                                     double *const *eigenvalues,
                                     rank_lift_correction_t *correction) {
  cublasPointerMode_t saved_mode;
  CUBLAS_CHECK(cublasGetPointerMode(state->blas_handle, &saved_mode));
  CUBLAS_CHECK(
      cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

  if (state->augmentation_mode == AUGMENTATION_MODE_RANDOM) {
    int total_added = build_random_columns(state, direction_counts, correction);
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, saved_mode));
    return total_added;
  }

  using Clock = std::chrono::steady_clock;
  if (state->verbose >= 3)
    CUDA_CHECK(cudaDeviceSynchronize());
  auto total_start = Clock::now();
  std::vector<int> selected_counts;
  std::vector<double *> selected_directions;
  std::vector<double *> selected_eigenvalues;
  int global_direction_count = select_strongest_directions(
      state, direction_counts, d_directions, eigenvalues, selected_counts,
      selected_directions, selected_eigenvalues);
  if (state->verbose >= 3)
    CUDA_CHECK(cudaDeviceSynchronize());
  auto selection_end = Clock::now();

  std::vector<LocalBlock> blocks;
  int local_variable_count = 0;
  bool use_sdp = (state->augmentation_mode == AUGMENTATION_MODE_SDP);
  for (int b = 0; b < state->n_blks; b++) {
    int r = selected_counts[b];
    if (r <= 0)
      continue;
    LocalBlock meta;
    meta.block_index = b;
    meta.direction_count = r;
    meta.variable_offset = local_variable_count;
    meta.variable_count = use_sdp ? packed_dim(r) : r;
    meta.d_U = NULL;
    local_variable_count += meta.variable_count;
    blocks.push_back(meta);
  }

  std::vector<double> local_linear(local_variable_count, 0.0);
  std::vector<int> local_block_sizes;
  double *d_local_columns = NULL;
  if (state->num_constraints > 0 && local_variable_count > 0)
    CUDA_CHECK(cudaMalloc(&d_local_columns, (size_t)state->num_constraints *
                                                local_variable_count *
                                                sizeof(double)));

  for (LocalBlock &meta : blocks) {
    block_low_rank_state_t *blk = state->block_low_rank_state[meta.block_index];
    int dim = blk->dim;
    int r = meta.direction_count;
    CUDA_CHECK(cudaMalloc(&meta.d_U, (size_t)dim * r * sizeof(double)));
    if (blk->psd_cone_rescaling != NULL) {
      CUBLAS_CHECK(cublasDdgmm(state->blas_handle, CUBLAS_SIDE_LEFT, dim, r,
                               selected_directions[meta.block_index], dim,
                               blk->psd_cone_rescaling, 1, meta.d_U, dim));
    } else {
      CUDA_CHECK(cudaMemcpy(meta.d_U, selected_directions[meta.block_index],
                            (size_t)dim * r * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    }

    std::vector<double> H;
    if (use_sdp)
      compute_projected_slack(state, blk, selected_directions[meta.block_index],
                              r, H);

    int variable = meta.variable_offset;
    for (int j = 0; j < r; j++) {
      local_linear[variable] = use_sdp
                                   ? H[(size_t)j * r + j]
                                   : selected_eigenvalues[meta.block_index][j];
      compute_rank_lift_A_ww(state, blk, meta.d_U + (size_t)j * dim,
                             d_local_columns +
                                 (size_t)variable * state->num_constraints);
      variable++;
    }
    if (use_sdp) {
      for (int i = 0; i < r; i++) {
        for (int j = i + 1; j < r; j++) {
          local_linear[variable] = 2.0 * H[(size_t)i * r + j];
          compute_rank_lift_A_uv(state, blk, meta.d_U + (size_t)i * dim,
                                 meta.d_U + (size_t)j * dim,
                                 d_local_columns +
                                     (size_t)variable * state->num_constraints);
          variable++;
        }
      }
      local_block_sizes.push_back(r);
    }
  }

  std::vector<double> global_linear;
  double *d_global_columns = NULL;
  std::vector<int> global_block_sizes;
  int local_variable_displacement = 0;
  gather_cone_axis(state, local_linear, d_local_columns, local_block_sizes,
                   global_linear, &d_global_columns, global_block_sizes,
                   &local_variable_displacement);
  bool global_columns_alias_local =
      d_global_columns != NULL && d_global_columns == d_local_columns;
  if (d_local_columns != NULL && !global_columns_alias_local)
    CUDA_CHECK(cudaFree(d_local_columns));
  if (state->verbose >= 3)
    CUDA_CHECK(cudaDeviceSynchronize());
  auto operator_end = Clock::now();

  int global_variables = (int)global_linear.size();
  std::vector<double> gram;
  compute_gram_gpu(state, d_global_columns, state->num_constraints,
                   global_variables, gram);
  if (d_global_columns != NULL && !global_columns_alias_local)
    CUDA_CHECK(cudaFree(d_global_columns));
  if (d_local_columns != NULL && global_columns_alias_local)
    CUDA_CHECK(cudaFree(d_local_columns));
  allreduce_constraint_gram(state, gram);
  if (state->verbose >= 3)
    CUDA_CHECK(cudaDeviceSynchronize());
  auto gram_end = Clock::now();

  std::vector<double> global_solution;
  bool closed_form =
      (state->augmentation_mode == AUGMENTATION_MODE_CLOSED_FORM);
  bool solved = false;
  double shared_t = 0.0;
  RankLiftSolveInfo solve_info = {0, 0.0, false};
  if (closed_form) {
    solved = solve_shared_scalar(global_linear, gram, state->penalty_coef,
                                 kRankLiftTolerance, false, global_block_sizes,
                                 &shared_t);
    solve_info.converged = solved;
  } else {
    std::vector<double> scaled_linear;
    std::vector<double> scaled_gram;
    std::vector<double> variable_scaling;
    bool scaled =
        scale_dense_subproblem(global_linear, gram, use_sdp, global_block_sizes,
                               scaled_linear, scaled_gram, variable_scaling);
    if (state->verbose >= 3 && scaled) {
      double min_diagonal = std::numeric_limits<double>::infinity();
      double max_diagonal = 0.0;
      double min_scaled_diagonal = std::numeric_limits<double>::infinity();
      double max_scaled_diagonal = 0.0;
      double max_linear = 0.0;
      double max_scaled_linear = 0.0;
      for (int i = 0; i < global_variables; i++) {
        min_diagonal = std::min(min_diagonal,
                                fabs(gram[(size_t)i * global_variables + i]));
        max_diagonal = std::max(max_diagonal,
                                fabs(gram[(size_t)i * global_variables + i]));
        min_scaled_diagonal =
            std::min(min_scaled_diagonal,
                     fabs(scaled_gram[(size_t)i * global_variables + i]));
        max_scaled_diagonal =
            std::max(max_scaled_diagonal,
                     fabs(scaled_gram[(size_t)i * global_variables + i]));
        max_linear = std::max(max_linear, fabs(global_linear[i]));
        max_scaled_linear = std::max(max_scaled_linear, fabs(scaled_linear[i]));
      }
      printf("  [rank lift] scale diag=[%.2e, %.2e] scaled=[%.2e, %.2e] "
             "|linear|max=%.2e scaled=%.2e\n",
             min_diagonal, max_diagonal, min_scaled_diagonal,
             max_scaled_diagonal, max_linear, max_scaled_linear);
    }
    if (scaled) {
      if (use_sdp) {
        solved = solve_semismooth_sdp(
            scaled_linear, scaled_gram, state->penalty_coef, kRankLiftTolerance,
            global_block_sizes, global_solution, &solve_info);
      } else {
        solved = solve_active_set_qp(scaled_linear, scaled_gram,
                                     state->penalty_coef, kRankLiftTolerance,
                                     global_solution, &solve_info);
      }
      if (solved)
        for (int i = 0; i < global_variables; i++)
          global_solution[i] *= variable_scaling[i];
    }
    if (!solved) {
      solved = solve_shared_scalar(global_linear, gram, state->penalty_coef,
                                   kRankLiftTolerance, use_sdp,
                                   global_block_sizes, &shared_t);
      closed_form = solved;
      if (state->verbose >= 3 && solved)
        printf("  [rank lift] joint solver failed; using shared scalar "
               "closed form\n");
    }
  }
  auto solve_end = Clock::now();

  int total_added = 0;
  if (solved) {
    if (closed_form) {
      std::vector<double> unused;
      total_added = build_qp_or_closed_columns(state, blocks, unused, shared_t,
                                               true, correction);
    } else {
      std::vector<double> local_solution(
          global_solution.begin() + local_variable_displacement,
          global_solution.begin() + local_variable_displacement +
              local_variable_count);
      if (use_sdp)
        total_added =
            build_sdp_columns(state, blocks, local_solution, correction);
      else
        total_added = build_qp_or_closed_columns(state, blocks, local_solution,
                                                 0.0, false, correction);
    }
  }
  if (state->verbose >= 3)
    CUDA_CHECK(cudaDeviceSynchronize());
  auto columns_end = Clock::now();

  for (LocalBlock &meta : blocks)
    if (meta.d_U != NULL)
      CUDA_CHECK(cudaFree(meta.d_U));
  free_selected_directions(selected_directions, selected_eigenvalues);
  if (state->verbose >= 3) {
    const char *mode = use_sdp ? "sdp" : "qp";
    if (state->augmentation_mode == AUGMENTATION_MODE_CLOSED_FORM)
      mode = "closed-form";
    double select_ms =
        std::chrono::duration<double, std::milli>(selection_end - total_start)
            .count();
    double operator_ms =
        std::chrono::duration<double, std::milli>(operator_end - selection_end)
            .count();
    double gram_ms =
        std::chrono::duration<double, std::milli>(gram_end - operator_end)
            .count();
    double solve_ms =
        std::chrono::duration<double, std::milli>(solve_end - gram_end).count();
    double columns_ms =
        std::chrono::duration<double, std::milli>(columns_end - solve_end)
            .count();
    double total_ms =
        std::chrono::duration<double, std::milli>(columns_end - total_start)
            .count();
    printf("  [rank lift] mode=%s directions=%d variables=%d "
           "converged=%d iterations=%d kkt=%.2e\n",
           mode, global_direction_count, global_variables,
           solve_info.converged ? 1 : 0, solve_info.iterations,
           solve_info.kkt_residual);
    printf("  [rank lift] time_ms select=%.3f operator=%.3f gram=%.3f "
           "solve=%.3f columns=%.3f total=%.3f\n",
           select_ms, operator_ms, gram_ms, solve_ms, columns_ms, total_ms);
  }
  CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, saved_mode));
  return total_added;
}
