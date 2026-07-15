/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "qubo_io.h"
#include "utils.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void skip_ws_and_comments(const char **cursor) {
  const char *p = *cursor;
  for (;;) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
      p++;
    if (*p == '\0')
      break;
    if (*p == 'c' || *p == 'C' || *p == '#' || *p == '%' || *p == '!' ||
        (*p == '/' && p[1] == '/')) {
      while (*p && *p != '\n')
        p++;
      continue;
    }
    break;
  }
  *cursor = p;
}

static int read_int(const char **cursor, int *out) {
  skip_ws_and_comments(cursor);
  if (**cursor == '\0')
    return 0;
  char *end;
  long v = strtol(*cursor, &end, 10);
  if (end == *cursor)
    return 0;
  *cursor = end;
  *out = (int)v;
  return 1;
}

static int read_double(const char **cursor, double *out) {
  skip_ws_and_comments(cursor);
  if (**cursor == '\0')
    return 0;
  char *end;
  double v = strtod(*cursor, &end);
  if (end == *cursor)
    return 0;
  *cursor = end;
  *out = v;
  return 1;
}

static int read_word(const char **cursor, char *buf, int bufsz) {
  skip_ws_and_comments(cursor);
  int i = 0;
  while (**cursor && !(**cursor == ' ' || **cursor == '\t' ||
                       **cursor == '\n' || **cursor == '\r') &&
         i < bufsz - 1) {
    buf[i++] = **cursor;
    (*cursor)++;
  }
  buf[i] = '\0';
  return i;
}

// =============================================================================
// D-Wave qbsolv .qubo format
//
//   c <comment lines, optional>
//   p qubo 0 <maxDiagonals> <nDiagonals> <nElements>
//   <i> <i> <v>     x nDiagonals      (diagonal, linear-in-{0,1} coefficient)
//   <i> <j> <v>     x nElements       (off-diagonal, polynomial coef on x_i x_j)
//
// 'target' (the field shown as '0' above) is ignored. Off-diagonals are
// converted from polynomial coefficient to symmetric-matrix entry by
// dividing by 2.
// =============================================================================

static qubo_problem_t *parse_dwave(const char *data) {
  const char *p = data;
  char tok[64];

  if (!read_word(&p, tok, sizeof(tok)) || strcmp(tok, "p") != 0) {
    fprintf(stderr, "qubo_io: D-Wave header missing 'p'\n");
    return NULL;
  }
  if (!read_word(&p, tok, sizeof(tok)) || strcmp(tok, "qubo") != 0) {
    fprintf(stderr, "qubo_io: D-Wave header expects 'qubo' after 'p'\n");
    return NULL;
  }
  // Target (e.g. '0' or 'unconstrained') is ignored.
  if (!read_word(&p, tok, sizeof(tok))) {
    fprintf(stderr, "qubo_io: D-Wave header truncated (target)\n");
    return NULL;
  }

  int n = 0, n_diag = 0, n_off = 0;
  if (!read_int(&p, &n) || !read_int(&p, &n_diag) || !read_int(&p, &n_off)) {
    fprintf(stderr, "qubo_io: D-Wave header truncated (n / nDiag / nOff)\n");
    return NULL;
  }
  if (n <= 0 || n_diag < 0 || n_off < 0) {
    fprintf(stderr,
            "qubo_io: D-Wave header invalid (n=%d nDiag=%d nOff=%d)\n", n,
            n_diag, n_off);
    return NULL;
  }

  int nnz_cap = n_diag + n_off;
  if (nnz_cap == 0) {
    qubo_problem_t *q = (qubo_problem_t *)safe_malloc(sizeof(qubo_problem_t));
    q->n = n;
    q->nnz = 0;
    q->row = q->col = NULL;
    q->val = NULL;
    q->linear = NULL;
    q->sense = 1;
    q->obj_const = 0.0;
    return q;
  }

  qubo_problem_t *q = (qubo_problem_t *)safe_malloc(sizeof(qubo_problem_t));
  q->n = n;
  q->row = (int *)safe_malloc((size_t)nnz_cap * sizeof(int));
  q->col = (int *)safe_malloc((size_t)nnz_cap * sizeof(int));
  q->val = (double *)safe_malloc((size_t)nnz_cap * sizeof(double));
  q->linear = NULL;

  int idx = 0;
  for (int k = 0; k < n_diag; k++) {
    int i, j;
    double v;
    if (!read_int(&p, &i) || !read_int(&p, &j) || !read_double(&p, &v)) {
      fprintf(stderr, "qubo_io: D-Wave diagonal entry #%d truncated\n", k);
      free(q->row);
      free(q->col);
      free(q->val);
      free(q);
      return NULL;
    }
    if (i < 0 || i >= n || j < 0 || j >= n) {
      fprintf(stderr,
              "qubo_io: diagonal index out of range at #%d: (%d,%d)\n", k, i,
              j);
      free(q->row);
      free(q->col);
      free(q->val);
      free(q);
      return NULL;
    }
    if (i != j) {
      fprintf(stderr,
              "qubo_io: D-Wave diagonal entry #%d has i=%d != j=%d\n", k, i,
              j);
    }
    q->row[idx] = i;
    q->col[idx] = i;
    q->val[idx] = v;
    idx++;
  }
  for (int k = 0; k < n_off; k++) {
    int i, j;
    double v;
    if (!read_int(&p, &i) || !read_int(&p, &j) || !read_double(&p, &v)) {
      fprintf(stderr, "qubo_io: D-Wave off-diag entry #%d truncated\n", k);
      free(q->row);
      free(q->col);
      free(q->val);
      free(q);
      return NULL;
    }
    if (i < 0 || i >= n || j < 0 || j >= n) {
      fprintf(stderr,
              "qubo_io: off-diag index out of range at #%d: (%d,%d)\n", k, i,
              j);
      free(q->row);
      free(q->col);
      free(q->val);
      free(q);
      return NULL;
    }
    if (i == j) {
      fprintf(stderr,
              "qubo_io: off-diag entry #%d has i==j=%d (treated as diag)\n", k,
              i);
      q->row[idx] = i;
      q->col[idx] = i;
      q->val[idx] = v;
    } else {
      if (i > j) {
        int t = i;
        i = j;
        j = t;
      }
      q->row[idx] = i;
      q->col[idx] = j;
      q->val[idx] = v * 0.5;
    }
    idx++;
  }
  q->nnz = idx;
  q->sense = 1;
  q->obj_const = 0.0;
  return q;
}

// =============================================================================
// Simple triplet format (OR-Library Beasley bqp style):
//
//   n nnz
//   i j v          x nnz
//
// 1-indexed if any index equals n (rare); we autodetect by checking whether
// any index hits n exactly (then treat as 1-indexed). Polynomial coef
// convention same as D-Wave: off-diag values are coefficients on x_i x_j and
// get halved to match our symmetric internal convention.
// =============================================================================

static qubo_problem_t *parse_triplet(const char *data) {
  const char *p = data;
  int n = 0, nnz = 0;
  if (!read_int(&p, &n) || !read_int(&p, &nnz)) {
    fprintf(stderr, "qubo_io: triplet header truncated (n nnz)\n");
    return NULL;
  }
  if (n <= 0 || nnz < 0) {
    fprintf(stderr, "qubo_io: triplet header invalid (n=%d nnz=%d)\n", n,
            nnz);
    return NULL;
  }

  int *raw_row = (int *)safe_malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
  int *raw_col = (int *)safe_malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
  double *raw_val =
      (double *)safe_malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
  int max_idx_seen = -1;

  for (int k = 0; k < nnz; k++) {
    int i, j;
    double v;
    if (!read_int(&p, &i) || !read_int(&p, &j) || !read_double(&p, &v)) {
      fprintf(stderr, "qubo_io: triplet entry #%d truncated\n", k);
      free(raw_row);
      free(raw_col);
      free(raw_val);
      return NULL;
    }
    raw_row[k] = i;
    raw_col[k] = j;
    raw_val[k] = v;
    if (i > max_idx_seen)
      max_idx_seen = i;
    if (j > max_idx_seen)
      max_idx_seen = j;
  }

  int one_indexed = (max_idx_seen >= n);
  int offset = one_indexed ? 1 : 0;
  if (one_indexed)
    fprintf(stderr,
            "qubo_io: triplet looks 1-indexed (max idx=%d, n=%d); "
            "converting to 0-indexed.\n",
            max_idx_seen, n);

  qubo_problem_t *q = (qubo_problem_t *)safe_malloc(sizeof(qubo_problem_t));
  q->n = n;
  q->nnz = nnz;
  q->row = (int *)safe_malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
  q->col = (int *)safe_malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
  q->val = (double *)safe_malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
  q->linear = NULL;

  for (int k = 0; k < nnz; k++) {
    int i = raw_row[k] - offset;
    int j = raw_col[k] - offset;
    double v = raw_val[k];
    if (i < 0 || i >= n || j < 0 || j >= n) {
      fprintf(stderr,
              "qubo_io: triplet entry #%d out of range after offset: (%d,%d) "
              "n=%d\n",
              k, i, j, n);
      free(q->row);
      free(q->col);
      free(q->val);
      free(q);
      free(raw_row);
      free(raw_col);
      free(raw_val);
      return NULL;
    }
    if (i == j) {
      q->row[k] = i;
      q->col[k] = i;
      q->val[k] = v;
    } else {
      if (i > j) {
        int t = i;
        i = j;
        j = t;
      }
      q->row[k] = i;
      q->col[k] = j;
      q->val[k] = v * 0.5;
    }
  }
  free(raw_row);
  free(raw_col);
  free(raw_val);
  q->sense = 1;
  q->obj_const = 0.0;
  return q;
}

// =============================================================================
// QPLIB 2014/2019 .qplib format (binary QP subset only).
//
//   <problem_name>
//   <type>            3-char code: <obj><con><var>
//                       obj: L=linear, D=diag-Q, C=convex-Q, Q=general-Q
//                       con: N=none, B=box-bounds, L=linear, Q=quad, R=conic
//                       var: C=continuous, B=binary, I=integer, M=mixed
//   <sense>           min/max/MINIMIZE/MAXIMIZE
//   <n>               number of variables
//   [<m>]             number of linear constraints (only if con=L/Q/R)
//   <nQ> + <i j v> x nQ                 objective Q (lower-tri, symmetric)
//   <b_default> <nB> + <i v> x nB        objective linear (default + deltas)
//   <q_0>                                objective constant
//   <infinity>
//   <lb_def> <nLB> + <i v> x nLB         lower bounds
//   <ub_def> <nUB> + <i v> x nUB         upper bounds
//   [mixed-type section, starting points, names — all skipped for QUBO]
//
// Objective convention used by QPLIB binary instances (and matched by QuBowl
// / standard QUBO Python readers): each file entry (i, j, v) — diagonal or
// not — contributes
//      0.5 * v * x_i * x_j
// to f(x). Lower-triangular entries are NOT implicitly mirrored. Equivalently
// f(x) = 0.5 * sum_{file entries} v * x_i * x_j + b^T x + q_0
// (this differs from "f = 1/2 x^T Q x with Q symmetric" by a factor of 2 on
// off-diagonal entries: those QPLIB docs assume Q is symmetrized, but the
// stored instances in practice are not).
//
// Linear part b: written verbatim into q->linear. Comments start with '!'
// (also c/C/#/%) anywhere; trailing-after-value comments work since '!' is in
// the comment-skip set.
// =============================================================================

static int qplib_validate_type(const char *type, int *out_has_lin_con) {
  if (strlen(type) != 3) {
    fprintf(stderr,
            "qubo_io: QPLIB type '%s' is not 3 chars\n", type);
    return 0;
  }
  // QPLIB 2019 convention: <objective><variable><constraint>.
  char obj = type[0], var = type[1], con = type[2];
  if (var != 'B') {
    fprintf(stderr,
            "qubo_io: QPLIB type '%s' has non-binary variables ('%c'); "
            "QUBO subset requires 'B'.\n",
            type, var);
    return 0;
  }
  if (obj != 'L' && obj != 'D' && obj != 'C' && obj != 'Q') {
    fprintf(stderr,
            "qubo_io: QPLIB type '%s' has unknown objective code '%c'.\n",
            type, obj);
    return 0;
  }
  if (con == 'Q' || con == 'R') {
    fprintf(stderr,
            "qubo_io: QPLIB type '%s' has quadratic/conic constraints; "
            "only box-bound (N/B) or pure linear (L w/ m=0) QUBO is "
            "supported.\n",
            type);
    return 0;
  }
  *out_has_lin_con = (con == 'L');
  return 1;
}

static qubo_problem_t *parse_qplib(const char *data) {
  const char *p = data;
  char tok[256];

  // 1. Problem name (ignored).
  if (!read_word(&p, tok, sizeof(tok))) {
    fprintf(stderr, "qubo_io: QPLIB: missing problem name\n");
    return NULL;
  }

  // 2. Type code.
  char type_code[8];
  if (!read_word(&p, type_code, sizeof(type_code))) {
    fprintf(stderr, "qubo_io: QPLIB: missing type code\n");
    return NULL;
  }
  int has_lin_con = 0;
  if (!qplib_validate_type(type_code, &has_lin_con))
    return NULL;

  // 3. Sense.
  if (!read_word(&p, tok, sizeof(tok))) {
    fprintf(stderr, "qubo_io: QPLIB: missing sense\n");
    return NULL;
  }
  int maximize = 0;
  if (tok[0] == 'm' || tok[0] == 'M') {
    if (tok[1] == 'a' || tok[1] == 'A')
      maximize = 1;
    else if (tok[1] == 'i' || tok[1] == 'I')
      maximize = 0;
    else {
      fprintf(stderr, "qubo_io: QPLIB: unknown sense '%s'\n", tok);
      return NULL;
    }
  } else {
    fprintf(stderr, "qubo_io: QPLIB: unknown sense '%s'\n", tok);
    return NULL;
  }
  double sign = maximize ? -1.0 : 1.0;

  // 4. n.
  int n = 0;
  if (!read_int(&p, &n) || n <= 0) {
    fprintf(stderr, "qubo_io: QPLIB: bad n\n");
    return NULL;
  }

  // 5. m (if linear constraints declared by type).
  if (has_lin_con) {
    int m = 0;
    if (!read_int(&p, &m)) {
      fprintf(stderr, "qubo_io: QPLIB: missing m\n");
      return NULL;
    }
    if (m > 0) {
      fprintf(stderr,
              "qubo_io: QPLIB has %d linear constraints; not supported by "
              "QUBO subset.\n",
              m);
      return NULL;
    }
  }

  // 6. Objective Q (lower-triangular symmetric).
  int q_nnz = 0;
  if (!read_int(&p, &q_nnz) || q_nnz < 0) {
    fprintf(stderr, "qubo_io: QPLIB: bad obj Q nnz\n");
    return NULL;
  }
  int *q_row = NULL, *q_col = NULL;
  double *q_val = NULL;
  if (q_nnz > 0) {
    q_row = (int *)safe_malloc((size_t)q_nnz * sizeof(int));
    q_col = (int *)safe_malloc((size_t)q_nnz * sizeof(int));
    q_val = (double *)safe_malloc((size_t)q_nnz * sizeof(double));
  }
  for (int k = 0; k < q_nnz; k++) {
    int i, j;
    double v;
    if (!read_int(&p, &i) || !read_int(&p, &j) || !read_double(&p, &v)) {
      fprintf(stderr, "qubo_io: QPLIB: obj Q entry #%d truncated\n", k);
      free(q_row); free(q_col); free(q_val);
      return NULL;
    }
    i--; j--; // QPLIB is 1-indexed
    if (i < 0 || i >= n || j < 0 || j >= n) {
      fprintf(stderr,
              "qubo_io: QPLIB: obj Q entry #%d index out of range "
              "(%d,%d), n=%d\n",
              k, i + 1, j + 1, n);
      free(q_row); free(q_col); free(q_val);
      return NULL;
    }
    if (i > j) { int t = i; i = j; j = t; } // normalize to upper-tri
    q_row[k] = i;
    q_col[k] = j;
    q_val[k] = v;
  }

  // 7. Linear: default, n_changes, (i v) x.
  double lin_default = 0.0;
  int lin_changes = 0;
  if (!read_double(&p, &lin_default) || !read_int(&p, &lin_changes) ||
      lin_changes < 0) {
    fprintf(stderr, "qubo_io: QPLIB: linear section truncated\n");
    free(q_row); free(q_col); free(q_val);
    return NULL;
  }
  double *linear = (double *)safe_malloc((size_t)n * sizeof(double));
  for (int i = 0; i < n; i++)
    linear[i] = lin_default;
  for (int k = 0; k < lin_changes; k++) {
    int i;
    double v;
    if (!read_int(&p, &i) || !read_double(&p, &v)) {
      fprintf(stderr, "qubo_io: QPLIB: linear change #%d truncated\n", k);
      free(linear); free(q_row); free(q_col); free(q_val);
      return NULL;
    }
    i--;
    if (i < 0 || i >= n) {
      fprintf(stderr, "qubo_io: QPLIB: linear change idx out of range %d\n",
              i + 1);
      free(linear); free(q_row); free(q_col); free(q_val);
      return NULL;
    }
    linear[i] = v;
  }

  // 8. Objective constant (ignored; only shifts f by a constant).
  double obj_const = 0.0;
  if (!read_double(&p, &obj_const)) {
    fprintf(stderr, "qubo_io: QPLIB: missing obj constant\n");
    free(linear); free(q_row); free(q_col); free(q_val);
    return NULL;
  }
  if (obj_const != 0.0)
    LOG_DBG("[QUBO read] QPLIB obj_const = %g (added back in reported f)\n",
            obj_const);

  qubo_problem_t *q = (qubo_problem_t *)safe_malloc(sizeof(qubo_problem_t));
  q->n = n;
  q->nnz = q_nnz;
  q->row =
      (int *)safe_malloc((size_t)(q_nnz > 0 ? q_nnz : 1) * sizeof(int));
  q->col =
      (int *)safe_malloc((size_t)(q_nnz > 0 ? q_nnz : 1) * sizeof(int));
  q->val =
      (double *)safe_malloc((size_t)(q_nnz > 0 ? q_nnz : 1) * sizeof(double));
  q->linear = (double *)safe_malloc((size_t)n * sizeof(double));

  // QPLIB / QuBowl convention for binary instances: each file entry (i, j, v)
  // contributes 0.5 * v * x_i * x_j to f (entries are NOT implicitly mirrored
  // across the diagonal). To match this internally:
  //   diag (i==j):  val[k] = 0.5 * v  (SDP contributes val * Y_{ii} = val * x_i)
  //   off-diag:     val[k] = 0.25 * v (SDP doubles off-diag -> 2 * val * x_i x_j)
  for (int k = 0; k < q_nnz; k++) {
    q->row[k] = q_row[k];
    q->col[k] = q_col[k];
    double f = (q_row[k] == q_col[k]) ? 0.5 : 0.25;
    q->val[k] = sign * q_val[k] * f;
  }
  for (int i = 0; i < n; i++)
    q->linear[i] = sign * linear[i];
  q->sense = maximize ? -1 : 1;
  q->obj_const = obj_const;

  free(linear);
  free(q_row);
  free(q_col);
  free(q_val);
  return q;
}

static qubo_file_format_t sniff_format(const char *data) {
  const char *p = data;
  for (;;) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
      p++;
    if (*p == '\0')
      return QUBO_FMT_TRIPLET;
    if (*p == 'c' || *p == 'C' || *p == '#' || *p == '%' || *p == '!' ||
        (*p == '/' && p[1] == '/')) {
      while (*p && *p != '\n')
        p++;
      continue;
    }
    break;
  }
  if ((*p == 'p' || *p == 'P') &&
      (p[1] == ' ' || p[1] == '\t' || p[1] == '\n' || p[1] == '\r' ||
       p[1] == '\0'))
    return QUBO_FMT_DWAVE;
  if (isdigit((unsigned char)*p) || *p == '+' || *p == '-' || *p == '.')
    return QUBO_FMT_TRIPLET;
  return QUBO_FMT_QPLIB;
}

qubo_problem_t *qubo_read_file(const char *path, qubo_file_format_t fmt) {
  if (!path) {
    fprintf(stderr, "qubo_read_file: NULL path\n");
    return NULL;
  }
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    fprintf(stderr, "qubo_read_file: cannot open '%s': %s\n", path,
            strerror(errno));
    return NULL;
  }
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    fprintf(stderr, "qubo_read_file: fseek failed on '%s'\n", path);
    return NULL;
  }
  long sz = ftell(fp);
  if (sz < 0) {
    fclose(fp);
    fprintf(stderr, "qubo_read_file: ftell failed on '%s'\n", path);
    return NULL;
  }
  rewind(fp);
  char *data = (char *)safe_malloc((size_t)sz + 1);
  size_t got = fread(data, 1, (size_t)sz, fp);
  fclose(fp);
  data[got] = '\0';

  if (fmt == QUBO_FMT_AUTO)
    fmt = sniff_format(data);

  qubo_problem_t *q = NULL;
  if (fmt == QUBO_FMT_DWAVE) {
    LOG_DBG("[QUBO read] %s (D-Wave qbsolv format)\n", path);
    q = parse_dwave(data);
  } else if (fmt == QUBO_FMT_QPLIB) {
    LOG_DBG("[QUBO read] %s (QPLIB format)\n", path);
    q = parse_qplib(data);
  } else {
    LOG_DBG("[QUBO read] %s (triplet format)\n", path);
    q = parse_triplet(data);
  }

  free(data);
  if (q)
    LOG_DBG("[QUBO read] n=%d nnz=%d\n", q->n, q->nnz);
  return q;
}
