/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

// python_bindings/_core.cpp
//
// Thin pybind11 wrapper around the CARDAL public C ABI (include/cardal.h).
// This TU does not #include any internal solver headers — the C ABI is the
// contract, so internal refactors of cardal_parameters_t / sdp_result_t
// do not force a rebuild here.
//
// The Python-facing package (python/cardal/) provides the ergonomic
// dataclass Result / TerminationReason enum / solve_sdpa() free function
// on top of these low-level bindings.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <csignal>

extern "C" {
#include "cardal.h"
}

namespace py = pybind11;

namespace {

// SIGINT handler installed by the binding while a solve is running.
// Just sets the C-side cooperative-cancel flag; the outer loop polls it at
// each iteration boundary and returns CARDAL_STATUS_USER_INTERRUPT.
extern "C" void cardal_sigint_handler(int) {
    cardal_request_cancel();
}

} // namespace

namespace {

// Wrap a borrowed C pointer as a read-only numpy array whose lifetime is
// tied to the Python owner (typically a _CoreResult handle). A base py::object
// keeps the owner alive as long as the array survives.
py::array_t<double>
borrow_as_numpy(const double *data, py::ssize_t n, py::object owner) {
    if (data == nullptr || n <= 0)
        return py::array_t<double>({0});
    // 4th arg is the "base" — pybind11 stashes it on the array so the C
    // buffer is not freed before the numpy view. No copy is performed.
    return py::array_t<double>(
        {n},                 // shape
        {sizeof(double)},    // strides
        data,                // pointer
        std::move(owner));   // base
}

py::array_t<int>
borrow_int_as_numpy(const int *data, py::ssize_t n, py::object owner) {
    if (data == nullptr || n <= 0)
        return py::array_t<int>({0});
    return py::array_t<int>(
        {n},
        {sizeof(int)},
        data,
        std::move(owner));
}

// Convert a Python dict of {name: value} into a cardal_params struct. Unknown
// keys raise TypeError (contra PDHCG-II A1 — sub-typo protection).
cardal_params
params_from_dict(const py::dict &d) {
    cardal_params p;
    cardal_default_params(&p);

    static const std::vector<std::string> known_keys = {
        "eps_optimal_relative",
        "eps_feasible_relative",
        "time_sec_limit",
        "iteration_limit",
        "initial_rank",
        "max_rank",
        "lbfgs_history_size",
        "penalty_factor",
        "initial_penalty_coef",
        "max_penalty_coef",
        "inner_iterations_limit",
        "verbose",
    };
    auto is_known = [&](const std::string &k) {
        for (auto &s : known_keys) if (s == k) return true;
        return false;
    };

    for (auto item : d) {
        std::string key = py::str(item.first);
        if (!is_known(key)) {
            throw py::type_error("unknown parameter: " + key);
        }
        py::object v = py::reinterpret_borrow<py::object>(item.second);
        if      (key == "eps_optimal_relative")   p.eps_optimal_relative   = v.cast<double>();
        else if (key == "eps_feasible_relative")  p.eps_feasible_relative  = v.cast<double>();
        else if (key == "time_sec_limit")         p.time_sec_limit         = v.cast<double>();
        else if (key == "iteration_limit")        p.iteration_limit        = v.cast<int>();
        else if (key == "initial_rank")           p.initial_rank           = v.cast<int>();
        else if (key == "max_rank")               p.max_rank               = v.cast<int>();
        else if (key == "lbfgs_history_size")     p.lbfgs_history_size     = v.cast<int>();
        else if (key == "penalty_factor")         p.penalty_factor         = v.cast<double>();
        else if (key == "initial_penalty_coef")   p.initial_penalty_coef   = v.cast<double>();
        else if (key == "max_penalty_coef")       p.max_penalty_coef       = v.cast<double>();
        else if (key == "inner_iterations_limit") p.inner_iterations_limit = v.cast<long>();
        else if (key == "verbose")                p.verbose                = v.cast<int>();
    }
    return p;
}

// --------------------------------------------------------------------
// Handle wrappers
// --------------------------------------------------------------------

struct CoreProblem {
    cardal_problem *raw = nullptr;

    CoreProblem() = default;
    explicit CoreProblem(cardal_problem *p) : raw(p) {}
    CoreProblem(const CoreProblem &) = delete;
    CoreProblem &operator=(const CoreProblem &) = delete;
    CoreProblem(CoreProblem &&other) noexcept : raw(other.raw) { other.raw = nullptr; }
    CoreProblem &operator=(CoreProblem &&other) noexcept {
        if (this != &other) {
            if (raw) cardal_problem_free(raw);
            raw = other.raw;
            other.raw = nullptr;
        }
        return *this;
    }
    ~CoreProblem() { if (raw) cardal_problem_free(raw); }
};

struct CoreResult {
    cardal_result *raw = nullptr;

    CoreResult() = default;
    explicit CoreResult(cardal_result *r) : raw(r) {}
    CoreResult(const CoreResult &) = delete;
    CoreResult &operator=(const CoreResult &) = delete;
    CoreResult(CoreResult &&other) noexcept : raw(other.raw) { other.raw = nullptr; }
    CoreResult &operator=(CoreResult &&other) noexcept {
        if (this != &other) {
            if (raw) cardal_result_free(raw);
            raw = other.raw;
            other.raw = nullptr;
        }
        return *this;
    }
    ~CoreResult() { if (raw) cardal_result_free(raw); }
};

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "CARDAL low-level bindings (see python/cardal/ for the "
              "ergonomic API).";

    // ----- Version marker: bump on ABI-visible changes ------------------
    m.attr("__abi_version__") = py::int_(1);

    // ----- Status enum --------------------------------------------------
    py::enum_<cardal_status>(m, "Status")
        .value("UNSPECIFIED",     CARDAL_STATUS_UNSPECIFIED)
        .value("OPTIMAL",         CARDAL_STATUS_OPTIMAL)
        .value("TIME_LIMIT",      CARDAL_STATUS_TIME_LIMIT)
        .value("ITERATION_LIMIT", CARDAL_STATUS_ITERATION_LIMIT)
        .value("USER_INTERRUPT",  CARDAL_STATUS_USER_INTERRUPT)
        .export_values();

    // ----- Problem handle -----------------------------------------------
    py::class_<CoreProblem>(m, "Problem")
        .def_property_readonly("num_constraints",
            [](CoreProblem &self) { return cardal_problem_num_constraints(self.raw); })
        .def_property_readonly("num_cones",
            [](CoreProblem &self) { return cardal_problem_num_cones(self.raw); })
        .def_property_readonly("num_variables",
            [](CoreProblem &self) { return cardal_problem_num_variables(self.raw); })
        .def_property_readonly("lp_dim",
            [](CoreProblem &self) { return cardal_problem_lp_dim(self.raw); })
        .def_property_readonly("block_dims",
            [](CoreProblem &self) {
                int n = cardal_problem_num_cones(self.raw);
                std::vector<int> dims(std::max(n, 1));
                if (n > 0) cardal_problem_get_block_dims(self.raw, dims.data());
                dims.resize(n);
                return dims;
            });

    // ----- Result handle ------------------------------------------------
    py::class_<CoreResult>(m, "Result")
        .def_property_readonly("status",
            [](CoreResult &self) { return cardal_result_status(self.raw); })
        .def_property_readonly("primal_objective",
            [](CoreResult &self) { return cardal_result_primal_objective(self.raw); })
        .def_property_readonly("dual_objective",
            [](CoreResult &self) { return cardal_result_dual_objective(self.raw); })
        .def_property_readonly("objective_gap",
            [](CoreResult &self) { return cardal_result_objective_gap(self.raw); })
        .def_property_readonly("rel_primal_residual",
            [](CoreResult &self) { return cardal_result_rel_primal_residual(self.raw); })
        .def_property_readonly("rel_dual_residual",
            [](CoreResult &self) { return cardal_result_rel_dual_residual(self.raw); })
        .def_property_readonly("rel_objective_gap",
            [](CoreResult &self) { return cardal_result_rel_objective_gap(self.raw); })
        .def_property_readonly("runtime_sec",
            [](CoreResult &self) { return cardal_result_runtime_sec(self.raw); })
        .def_property_readonly("outer_iters",
            [](CoreResult &self) { return cardal_result_outer_iters(self.raw); })
        .def_property_readonly("inner_iters",
            [](CoreResult &self) { return cardal_result_inner_iters(self.raw); })
        .def_property_readonly("num_cones",
            [](CoreResult &self) { return cardal_result_num_cones(self.raw); })
        .def_property_readonly("num_variables",
            [](CoreResult &self) { return cardal_result_num_variables(self.raw); })
        .def_property_readonly("num_constraints",
            [](CoreResult &self) { return cardal_result_num_constraints(self.raw); })
        .def_property_readonly("total_rank",
            [](CoreResult &self) { return cardal_result_total_rank(self.raw); })
        .def("primal_factor",
             [](py::object self) {
                 CoreResult &r = self.cast<CoreResult &>();
                 int n = 0;
                 const double *p = cardal_result_primal_factor(r.raw, &n);
                 return borrow_as_numpy(p, n, self);
             },
             "Return the low-rank primal factor as a zero-copy numpy view.")
        .def("dual",
             [](py::object self) {
                 CoreResult &r = self.cast<CoreResult &>();
                 int n = 0;
                 const double *p = cardal_result_dual(r.raw, &n);
                 return borrow_as_numpy(p, n, self);
             },
             "Return the dual solution y as a zero-copy numpy view.")
        .def("rank_list",
             [](py::object self) {
                 CoreResult &r = self.cast<CoreResult &>();
                 int n = 0;
                 const int *p = cardal_result_rank_list(r.raw, &n);
                 return borrow_int_as_numpy(p, n, self);
             },
             "Return the per-cone BM rank list as a zero-copy numpy view.");

    // ----- Free functions -----------------------------------------------
    m.def("default_params",
          []() {
              cardal_params p;
              cardal_default_params(&p);
              py::dict d;
              d["eps_optimal_relative"]   = p.eps_optimal_relative;
              d["eps_feasible_relative"]  = p.eps_feasible_relative;
              d["time_sec_limit"]         = p.time_sec_limit;
              d["iteration_limit"]        = p.iteration_limit;
              d["initial_rank"]           = p.initial_rank;
              d["max_rank"]               = p.max_rank;
              d["lbfgs_history_size"]     = p.lbfgs_history_size;
              d["penalty_factor"]         = p.penalty_factor;
              d["initial_penalty_coef"]   = p.initial_penalty_coef;
              d["max_penalty_coef"]       = p.max_penalty_coef;
              d["inner_iterations_limit"] = p.inner_iterations_limit;
              d["verbose"]                = p.verbose;
              return d;
          },
          "Return the CARDAL default parameter values as a dict.");

    m.def("read_sdpa",
          [](const std::string &path) {
              cardal_error err = CARDAL_OK;
              cardal_problem *raw = cardal_read_sdpa(path.c_str(), &err);
              if (raw == nullptr) {
                  switch (err) {
                  case CARDAL_E_NULL_ARG:
                      throw py::value_error("path is empty");
                  case CARDAL_E_FILE_IO:
                      throw py::value_error("cannot open " + path);
                  case CARDAL_E_PARSE:
                      throw py::value_error(
                          "failed to parse " + path +
                          " (unsupported or corrupt format)");
                  default:
                      throw std::runtime_error("internal error loading " + path);
                  }
              }
              return CoreProblem(raw);
          },
          py::arg("path"),
          "Load an SDPA/MATLAB/PDSDP file and return a Problem handle.");

    // Build a Problem from COO triplet numpy arrays (no file involved).
    // Ownership: C API copies every array internally, so the numpy inputs
    // may go out of scope as soon as this function returns.
    m.def("build_problem",
          [](int num_constraints,
             int num_cones,
             int lp_dim,
             py::array_t<int, py::array::c_style | py::array::forcecast> blk_dims,
             // C
             py::array_t<int, py::array::c_style | py::array::forcecast> c_cone_ind,
             py::array_t<int, py::array::c_style | py::array::forcecast> c_row_ind,
             py::array_t<int, py::array::c_style | py::array::forcecast> c_col_ind,
             py::array_t<double, py::array::c_style | py::array::forcecast> c_val,
             // A
             py::array_t<int, py::array::c_style | py::array::forcecast> a_constr_ind,
             py::array_t<int, py::array::c_style | py::array::forcecast> a_cone_ind,
             py::array_t<int, py::array::c_style | py::array::forcecast> a_row_ind,
             py::array_t<int, py::array::c_style | py::array::forcecast> a_col_ind,
             py::array_t<double, py::array::c_style | py::array::forcecast> a_val,
             // LP
             py::array_t<double, py::array::c_style | py::array::forcecast> lp_obj,
             py::array_t<int, py::array::c_style | py::array::forcecast> lp_constr_ind,
             py::array_t<int, py::array::c_style | py::array::forcecast> lp_col_ind,
             py::array_t<double, py::array::c_style | py::array::forcecast> lp_val,
             // b
             py::array_t<double, py::array::c_style | py::array::forcecast> b)
          {
              // Size checks that map cleanly to a Python ValueError.
              auto check_size = [](const char *name, py::ssize_t got, py::ssize_t want) {
                  if (got != want) {
                      throw py::value_error(std::string("array '") + name +
                          "' has length " + std::to_string(got) +
                          ", expected " + std::to_string(want));
                  }
              };
              if (num_constraints < 0 || num_cones < 0 || lp_dim < 0)
                  throw py::value_error("num_constraints, num_cones, lp_dim must all be >= 0");
              if (num_cones == 0 && lp_dim == 0)
                  throw py::value_error("problem must have at least one cone or an LP block");
              check_size("blk_dims", blk_dims.size(), num_cones);
              check_size("b",        b.size(),        num_constraints);
              const py::ssize_t nnz_c = c_val.size();
              check_size("c_cone_ind", c_cone_ind.size(), nnz_c);
              check_size("c_row_ind",  c_row_ind.size(),  nnz_c);
              check_size("c_col_ind",  c_col_ind.size(),  nnz_c);
              const py::ssize_t nnz_a = a_val.size();
              check_size("a_constr_ind", a_constr_ind.size(), nnz_a);
              check_size("a_cone_ind",   a_cone_ind.size(),   nnz_a);
              check_size("a_row_ind",    a_row_ind.size(),    nnz_a);
              check_size("a_col_ind",    a_col_ind.size(),    nnz_a);
              check_size("lp_obj", lp_obj.size(), lp_dim);
              const py::ssize_t nnz_lp = lp_val.size();
              check_size("lp_constr_ind", lp_constr_ind.size(), nnz_lp);
              check_size("lp_col_ind",    lp_col_ind.size(),    nnz_lp);

              cardal_problem_data d = {};
              d.num_constraints = num_constraints;
              d.num_cones       = num_cones;
              d.lp_dim          = lp_dim;
              d.blk_dims        = num_cones > 0 ? blk_dims.data() : nullptr;
              d.nnz_c        = (int)nnz_c;
              d.c_cone_ind   = nnz_c > 0 ? c_cone_ind.data() : nullptr;
              d.c_row_ind    = nnz_c > 0 ? c_row_ind.data()  : nullptr;
              d.c_col_ind    = nnz_c > 0 ? c_col_ind.data()  : nullptr;
              d.c_val        = nnz_c > 0 ? c_val.data()      : nullptr;
              d.nnz_a        = (int)nnz_a;
              d.a_constr_ind = nnz_a > 0 ? a_constr_ind.data() : nullptr;
              d.a_cone_ind   = nnz_a > 0 ? a_cone_ind.data()   : nullptr;
              d.a_row_ind    = nnz_a > 0 ? a_row_ind.data()    : nullptr;
              d.a_col_ind    = nnz_a > 0 ? a_col_ind.data()    : nullptr;
              d.a_val        = nnz_a > 0 ? a_val.data()        : nullptr;
              d.lp_obj       = lp_dim > 0 ? lp_obj.data() : nullptr;
              d.nnz_lp       = (int)nnz_lp;
              d.lp_constr_ind = nnz_lp > 0 ? lp_constr_ind.data() : nullptr;
              d.lp_col_ind    = nnz_lp > 0 ? lp_col_ind.data()    : nullptr;
              d.lp_val        = nnz_lp > 0 ? lp_val.data()        : nullptr;
              d.b            = num_constraints > 0 ? b.data() : nullptr;

              cardal_error err = CARDAL_OK;
              cardal_problem *raw = cardal_build_problem(&d, &err);
              if (raw == nullptr) {
                  switch (err) {
                  case CARDAL_E_NULL_ARG:
                      throw py::value_error("invalid problem data (null arg or empty shape)");
                  default:
                      throw std::runtime_error("internal error building problem");
                  }
              }
              return CoreProblem(raw);
          },
          py::arg("num_constraints"),
          py::arg("num_cones"),
          py::arg("lp_dim"),
          py::arg("blk_dims"),
          py::arg("c_cone_ind"),
          py::arg("c_row_ind"),
          py::arg("c_col_ind"),
          py::arg("c_val"),
          py::arg("a_constr_ind"),
          py::arg("a_cone_ind"),
          py::arg("a_row_ind"),
          py::arg("a_col_ind"),
          py::arg("a_val"),
          py::arg("lp_obj"),
          py::arg("lp_constr_ind"),
          py::arg("lp_col_ind"),
          py::arg("lp_val"),
          py::arg("b"),
          "Build a Problem from COO triplet numpy arrays. All int arrays are\n"
          "0-indexed. C copies internally; the numpy inputs may be freed after\n"
          "this call returns.");

    m.def("solve",
          [](CoreProblem &prob, const py::dict &params) {
              cardal_params p = params_from_dict(params);
              cardal_result *raw = nullptr;

              // Install a cooperative SIGINT handler that sets the C cancel
              // flag; restore the previous handler on exit so we do not
              // leak process-wide state.
              cardal_clear_cancel();
              auto prev_sigint = std::signal(SIGINT, cardal_sigint_handler);

              {
                  // Long-running CUDA work — release the GIL so other
                  // Python threads can make progress, and so that Ctrl-C
                  // in another thread reaches this process's signal
                  // handler promptly.
                  py::gil_scoped_release release;
                  raw = cardal_solve(prob.raw, &p);
              }

              std::signal(SIGINT, prev_sigint);
              int was_cancelled = cardal_cancel_requested();
              cardal_clear_cancel();

              if (raw == nullptr) {
                  if (was_cancelled)
                      throw py::error_already_set{
                          (PyErr_SetString(PyExc_KeyboardInterrupt,
                              "solve interrupted before producing a result"),
                           py::error_already_set())};
                  throw std::runtime_error("solve failed");
              }

              // If the user hit Ctrl-C, the solver returned a partial-state
              // result with USER_INTERRUPT status. We free that result and
              // raise KeyboardInterrupt so the Python side matches user
              // expectation (contra PDHCG-II A5, whose PyErr_SetInterrupt
              // is famously commented out and Ctrl-C silently returns a
              // partial result).
              if (was_cancelled) {
                  cardal_result_free(raw);
                  PyErr_SetString(PyExc_KeyboardInterrupt,
                                  "solve interrupted by SIGINT");
                  throw py::error_already_set();
              }

              return CoreResult(raw);
          },
          py::arg("problem"),
          py::arg("params") = py::dict(),
          "Solve a Problem and return a Result.\n\n"
          "``params`` accepts the same keys as ``default_params()``; unknown\n"
          "keys raise ``TypeError``. Ctrl-C during the solve raises\n"
          "``KeyboardInterrupt`` cleanly (the C solver polls the cancel\n"
          "flag at each outer iteration).");

    m.def("request_cancel", &cardal_request_cancel,
          "Manually request cancellation of any in-flight solve. The C-side\n"
          "outer loop polls this flag at each iteration boundary.");
    m.def("clear_cancel",   &cardal_clear_cancel,
          "Clear the cancellation flag (called automatically before each solve).");
    m.def("cancel_requested",
          []() { return cardal_cancel_requested() != 0; },
          "Return True if a cancellation is currently pending.");
}
