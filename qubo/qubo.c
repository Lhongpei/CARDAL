/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "qubo.h"
#include "utils.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int n;
  int bs_words;
  uint64_t **rows;
  int *deg;
} bs_graph_t;

#define BS_WORDS(n) (((n) + 63) >> 6)

static bs_graph_t *bs_alloc(int n) {
  bs_graph_t *g = (bs_graph_t *)safe_malloc(sizeof(bs_graph_t));
  g->n = n;
  g->bs_words = BS_WORDS(n);
  g->rows = (uint64_t **)safe_malloc((size_t)n * sizeof(uint64_t *));
  g->deg = (int *)safe_calloc((size_t)n, sizeof(int));
  for (int i = 0; i < n; i++) {
    g->rows[i] = (uint64_t *)safe_calloc((size_t)g->bs_words, sizeof(uint64_t));
  }
  return g;
}

static void bs_free(bs_graph_t *g) {
  if (!g)
    return;
  for (int i = 0; i < g->n; i++)
    free(g->rows[i]);
  free(g->rows);
  free(g->deg);
  free(g);
}

static inline int bs_test(const bs_graph_t *g, int u, int v) {
  return (int)((g->rows[u][v >> 6] >> (v & 63)) & 1ULL);
}

static inline void bs_add_edge(bs_graph_t *g, int u, int v) {
  if (u == v)
    return;
  uint64_t mu = 1ULL << (v & 63);
  if (g->rows[u][v >> 6] & mu)
    return;
  g->rows[u][v >> 6] |= mu;
  g->rows[v][u >> 6] |= 1ULL << (u & 63);
  g->deg[u]++;
  g->deg[v]++;
}

static int *min_degree_elimination(bs_graph_t *g) {
  int n = g->n;
  int *sigma = (int *)safe_malloc((size_t)n * sizeof(int));
  int *alive = (int *)safe_malloc((size_t)n * sizeof(int));
  int *cur_deg = (int *)safe_malloc((size_t)n * sizeof(int));
  int *nb = (int *)safe_malloc((size_t)n * sizeof(int));
  for (int v = 0; v < n; v++) {
    alive[v] = 1;
    cur_deg[v] = g->deg[v];
  }

  for (int step = 0; step < n; step++) {
    int best = -1;
    int best_deg = INT_MAX;
    for (int v = 0; v < n; v++) {
      if (!alive[v])
        continue;
      if (cur_deg[v] < best_deg) {
        best_deg = cur_deg[v];
        best = v;
        if (best_deg == 0)
          break;
      }
    }
    sigma[step] = best;

    int nb_count = 0;
    for (int w = 0; w < g->bs_words; w++) {
      uint64_t bits = g->rows[best][w];
      while (bits) {
        int bit = __builtin_ctzll(bits);
        int v = (w << 6) + bit;
        if (alive[v])
          nb[nb_count++] = v;
        bits &= bits - 1;
      }
    }

    for (int i = 0; i < nb_count; i++) {
      int u = nb[i];
      for (int j = i + 1; j < nb_count; j++) {
        int w = nb[j];
        if (!bs_test(g, u, w)) {
          uint64_t mu = 1ULL << (w & 63);
          g->rows[u][w >> 6] |= mu;
          g->rows[w][u >> 6] |= 1ULL << (u & 63);
          g->deg[u]++;
          g->deg[w]++;
          cur_deg[u]++;
          cur_deg[w]++;
        }
      }
    }

    for (int i = 0; i < nb_count; i++)
      cur_deg[nb[i]]--;
    alive[best] = 0;
  }

  free(alive);
  free(cur_deg);
  free(nb);
  return sigma;
}

typedef struct {
  int *verts;
  int size;
} clique_t;

static int cmp_int(const void *a, const void *b) {
  int x = *(const int *)a, y = *(const int *)b;
  return (x > y) - (x < y);
}

static clique_t *enumerate_maximal_cliques(const bs_graph_t *g,
                                           const int *sigma, int *out_K) {
  int n = g->n;
  int bw = g->bs_words;

  int *pos = (int *)safe_malloc((size_t)n * sizeof(int));
  for (int k = 0; k < n; k++)
    pos[sigma[k]] = k;

  uint64_t **npbs = (uint64_t **)safe_malloc((size_t)n * sizeof(uint64_t *));
  int *np_sz = (int *)safe_calloc((size_t)n, sizeof(int));
  for (int v = 0; v < n; v++) {
    npbs[v] = (uint64_t *)safe_calloc((size_t)bw, sizeof(uint64_t));
    for (int w = 0; w < bw; w++) {
      uint64_t bits = g->rows[v][w];
      while (bits) {
        int bit = __builtin_ctzll(bits);
        int u = (w << 6) + bit;
        if (pos[u] > pos[v]) {
          npbs[v][u >> 6] |= 1ULL << (u & 63);
          np_sz[v]++;
        }
        bits &= bits - 1;
      }
    }
  }

  int *maximal = (int *)safe_malloc((size_t)n * sizeof(int));
  for (int v = 0; v < n; v++)
    maximal[v] = 1;

  for (int v = 0; v < n; v++) {
    int dropped = 0;
    for (int w = 0; w < bw && !dropped; w++) {
      uint64_t bits = g->rows[v][w];
      while (bits) {
        int bit = __builtin_ctzll(bits);
        int u = (w << 6) + bit;
        bits &= bits - 1;
        if (pos[u] >= pos[v])
          continue;
        if (np_sz[u] < np_sz[v])
          continue;
        int subset = 1;
        for (int q = 0; q < bw; q++) {
          if (npbs[v][q] & ~npbs[u][q]) {
            subset = 0;
            break;
          }
        }
        if (subset) {
          maximal[v] = 0;
          dropped = 1;
          break;
        }
      }
    }
  }

  int K = 0;
  for (int v = 0; v < n; v++)
    if (maximal[v])
      K++;

  clique_t *cliques = (clique_t *)safe_malloc((size_t)K * sizeof(clique_t));
  int idx = 0;
  for (int v = 0; v < n; v++) {
    if (!maximal[v])
      continue;
    int sz = np_sz[v] + 1;
    int *vs = (int *)safe_malloc((size_t)sz * sizeof(int));
    int j = 0;
    vs[j++] = v;
    for (int w = 0; w < bw; w++) {
      uint64_t bits = npbs[v][w];
      while (bits) {
        int bit = __builtin_ctzll(bits);
        vs[j++] = (w << 6) + bit;
        bits &= bits - 1;
      }
    }
    qsort(vs, (size_t)sz, sizeof(int), cmp_int);
    cliques[idx].verts = vs;
    cliques[idx].size = sz;
    idx++;
  }

  for (int v = 0; v < n; v++)
    free(npbs[v]);
  free(npbs);
  free(np_sz);
  free(maximal);
  free(pos);

  *out_K = K;
  return cliques;
}

typedef struct {
  int u, v;
  int weight;
} clique_edge_t;

static clique_edge_t *build_clique_tree(uint64_t **cl_bs, int K, int bw_q,
                                        int *out_E) {
  if (K <= 1) {
    *out_E = 0;
    return NULL;
  }

  int *in_tree = (int *)safe_calloc((size_t)K, sizeof(int));
  int *best_nb = (int *)safe_calloc((size_t)K, sizeof(int));
  int *best_w = (int *)safe_malloc((size_t)K * sizeof(int));
  for (int k = 0; k < K; k++)
    best_w[k] = -1;

  clique_edge_t *edges =
      (clique_edge_t *)safe_malloc((size_t)(K - 1) * sizeof(clique_edge_t));
  int E = 0;

  in_tree[0] = 1;
  for (int k = 1; k < K; k++) {
    int w = 0;
    for (int j = 0; j < bw_q; j++)
      w += __builtin_popcountll(cl_bs[0][j] & cl_bs[k][j]);
    best_w[k] = w;
    best_nb[k] = 0;
  }

  for (int it = 1; it < K; it++) {
    int pick = -1;
    int pick_w = -1;
    for (int k = 0; k < K; k++) {
      if (in_tree[k])
        continue;
      if (best_w[k] > pick_w) {
        pick_w = best_w[k];
        pick = k;
      }
    }
    edges[E].u = best_nb[pick];
    edges[E].v = pick;
    edges[E].weight = pick_w;
    E++;
    in_tree[pick] = 1;
    for (int k = 0; k < K; k++) {
      if (in_tree[k])
        continue;
      int w = 0;
      for (int j = 0; j < bw_q; j++)
        w += __builtin_popcountll(cl_bs[pick][j] & cl_bs[k][j]);
      if (w > best_w[k]) {
        best_w[k] = w;
        best_nb[k] = pick;
      }
    }
  }

  free(in_tree);
  free(best_nb);
  free(best_w);
  *out_E = E;
  return edges;
}

static inline uint64_t xs_next(uint64_t *st) {
  uint64_t x = *st;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  *st = x;
  return x;
}

static inline double xs_unit(uint64_t *st) {
  return ((xs_next(st) >> 11) + 1.0) / 9007199254740992.0;
}

static inline double xs_pm1(uint64_t *st) { return 2.0 * xs_unit(st) - 1.0; }

void free_qubo_problem(qubo_problem_t *q) {
  if (!q)
    return;
  free(q->row);
  free(q->col);
  free(q->val);
  free(q->linear);
  free(q);
}

qubo_problem_t *generate_random_qubo(int n, double density, uint64_t seed) {
  if (n <= 0 || density < 0.0 || density > 1.0) {
    fprintf(stderr,
            "generate_random_qubo: invalid args (n=%d, density=%.3f)\n", n,
            density);
    return NULL;
  }
  uint64_t st = seed ? seed : 0xC0FFEE15600D7EAULL;

  double expected_off = density * 0.5 * (double)n * (double)(n - 1);
  size_t cap = (size_t)n + (size_t)(expected_off * 1.1) + 64;
  int *row = (int *)safe_malloc(cap * sizeof(int));
  int *col = (int *)safe_malloc(cap * sizeof(int));
  double *val = (double *)safe_malloc(cap * sizeof(double));
  int nnz = 0;

  for (int i = 0; i < n; i++) {
    row[nnz] = i;
    col[nnz] = i;
    val[nnz] = xs_pm1(&st);
    nnz++;
  }

  if (density >= 1.0) {
    for (int i = 0; i < n - 1; i++) {
      for (int j = i + 1; j < n; j++) {
        if ((size_t)nnz >= cap) {
          cap = cap * 3 / 2 + 16;
          row = (int *)safe_realloc(row, cap * sizeof(int));
          col = (int *)safe_realloc(col, cap * sizeof(int));
          val = (double *)safe_realloc(val, cap * sizeof(double));
        }
        row[nnz] = i;
        col[nnz] = j;
        val[nnz] = xs_pm1(&st);
        nnz++;
      }
    }
  } else if (density > 0.0) {
    double log_p = log(1.0 - density);
    int v = 1, w = -1;
    while (v < n) {
      double r = xs_unit(&st);
      int skip = (int)(log(r) / log_p);
      w = w + 1 + skip;
      while (w >= v && v < n) {
        w = w - v;
        v = v + 1;
      }
      if (v < n) {
        if ((size_t)nnz >= cap) {
          cap = cap * 3 / 2 + 16;
          row = (int *)safe_realloc(row, cap * sizeof(int));
          col = (int *)safe_realloc(col, cap * sizeof(int));
          val = (double *)safe_realloc(val, cap * sizeof(double));
        }
        row[nnz] = w;
        col[nnz] = v;
        val[nnz] = xs_pm1(&st);
        nnz++;
      }
    }
  }

  qubo_problem_t *q = (qubo_problem_t *)safe_malloc(sizeof(qubo_problem_t));
  q->n = n;
  q->nnz = nnz;
  q->row = row;
  q->col = col;
  q->val = val;
  q->linear = NULL;
  q->sense = 1;
  q->obj_const = 0.0;
  LOG_DBG("[QUBO gen] n=%d density=%.3e -> nnz=%d (diag=%d off-diag=%d)\n", n,
          density, nnz, n, nnz - n);
  return q;
}

basic_sdp_t *qubo_to_sdp_chordal(const qubo_problem_t *q) {
  if (q == NULL || q->n <= 0) {
    fprintf(stderr, "qubo_to_sdp_chordal: invalid QUBO (n=%d)\n",
            q ? q->n : -1);
    return NULL;
  }

  int n = q->n;
  LOG_DBG("[QUBO->SDP] n=%d, Q nnz=%d (upper-tri incl. diag)\n", n, q->nnz);

  bs_graph_t *g = bs_alloc(n);
  for (int k = 0; k < q->nnz; k++) {
    int i = q->row[k], j = q->col[k];
    if (i != j && q->val[k] != 0.0)
      bs_add_edge(g, i, j);
  }

  int *sigma = min_degree_elimination(g);

  int K = 0;
  clique_t *cliques = enumerate_maximal_cliques(g, sigma, &K);
  LOG_DBG("[QUBO->SDP] %d maximal clique(s)\n", K);

  int **pos_in = (int **)safe_malloc((size_t)K * sizeof(int *));
  for (int k = 0; k < K; k++) {
    pos_in[k] = (int *)safe_calloc((size_t)n, sizeof(int));
    for (int i = 0; i < cliques[k].size; i++)
      pos_in[k][cliques[k].verts[i]] = i + 1;
  }

  int bw_q = BS_WORDS(n);
  uint64_t **cl_bs = (uint64_t **)safe_malloc((size_t)K * sizeof(uint64_t *));
  for (int k = 0; k < K; k++) {
    cl_bs[k] = (uint64_t *)safe_calloc((size_t)bw_q, sizeof(uint64_t));
    for (int i = 0; i < cliques[k].size; i++) {
      int v = cliques[k].verts[i];
      cl_bs[k][v >> 6] |= 1ULL << (v & 63);
    }
  }

  int E = 0;
  clique_edge_t *tree_edges = build_clique_tree(cl_bs, K, bw_q, &E);

  int *home_vert = (int *)safe_malloc((size_t)n * sizeof(int));
  for (int i = 0; i < n; i++)
    home_vert[i] = -1;
  for (int k = 0; k < K; k++) {
    for (int idx = 0; idx < cliques[k].size; idx++) {
      int v = cliques[k].verts[idx];
      if (home_vert[v] == -1)
        home_vert[v] = k;
    }
  }

  basic_sdp_t *sdp = (basic_sdp_t *)safe_malloc(sizeof(basic_sdp_t));
  sdp->n_cones = K;
  sdp->blk_dims = (int *)safe_malloc((size_t)K * sizeof(int));
  for (int k = 0; k < K; k++)
    sdp->blk_dims[k] = cliques[k].size + 1;
  sdp->lp_dim = 0;
  sdp->lp_constraints = NULL;
  sdp->lp_objective = NULL;
  sdp->nnz_lp_constr = 0;
  sdp->nnz_lp_obj = 0;

  double *diag = (double *)safe_calloc((size_t)n, sizeof(double));
  for (int k = 0; k < q->nnz; k++) {
    int i = q->row[k], j = q->col[k];
    if (i == j)
      diag[i] += q->val[k];
  }
  if (q->linear) {
    for (int i = 0; i < n; i++)
      diag[i] += q->linear[i];
  }

  int n_diag_nnz = 0;
  for (int i = 0; i < n; i++)
    if (diag[i] != 0.0)
      n_diag_nnz++;
  int n_off_nnz = 0;
  for (int k = 0; k < q->nnz; k++) {
    if (q->row[k] != q->col[k] && q->val[k] != 0.0)
      n_off_nnz++;
  }

  int obj_cap = n_diag_nnz + n_off_nnz;
  sdp->psd_cone_objective =
      (psd_cone_objective_t *)safe_malloc(sizeof(psd_cone_objective_t));
  sdp->psd_cone_objective->cone_ind =
      (int *)safe_malloc((size_t)obj_cap * sizeof(int));
  sdp->psd_cone_objective->row_ind =
      (int *)safe_malloc((size_t)obj_cap * sizeof(int));
  sdp->psd_cone_objective->col_ind =
      (int *)safe_malloc((size_t)obj_cap * sizeof(int));
  sdp->psd_cone_objective->val =
      (double *)safe_malloc((size_t)obj_cap * sizeof(double));

  int obj_idx = 0;
  for (int i = 0; i < n; i++) {
    if (diag[i] == 0.0)
      continue;
    int kh = home_vert[i];
    int il = pos_in[kh][i];
    sdp->psd_cone_objective->cone_ind[obj_idx] = kh;
    sdp->psd_cone_objective->row_ind[obj_idx] = il;
    sdp->psd_cone_objective->col_ind[obj_idx] = il;
    sdp->psd_cone_objective->val[obj_idx] = diag[i];
    obj_idx++;
  }
  free(diag);

  for (int k = 0; k < q->nnz; k++) {
    int i = q->row[k], j = q->col[k];
    if (i == j || q->val[k] == 0.0)
      continue;
    if (i > j) {
      int t = i;
      i = j;
      j = t;
    }
    int home = -1;
    int best_size = INT_MAX;
    for (int kk = 0; kk < K; kk++) {
      if (pos_in[kk][i] > 0 && pos_in[kk][j] > 0 &&
          cliques[kk].size < best_size) {
        best_size = cliques[kk].size;
        home = kk;
      }
    }
    if (home == -1) {
      fprintf(stderr,
              "qubo_to_sdp_chordal: edge (%d,%d) not covered by any clique\n",
              i, j);
      continue;
    }
    int il = pos_in[home][i];
    int jl = pos_in[home][j];
    if (il > jl) {
      int t = il;
      il = jl;
      jl = t;
    }
    sdp->psd_cone_objective->cone_ind[obj_idx] = home;
    sdp->psd_cone_objective->row_ind[obj_idx] = il;
    sdp->psd_cone_objective->col_ind[obj_idx] = jl;
    sdp->psd_cone_objective->val[obj_idx] = q->val[k];
    obj_idx++;
  }
  sdp->nnz_psd_obj = obj_idx;

  int mA = K;
  int mB = 0;
  for (int k = 0; k < K; k++)
    mB += cliques[k].size;

  int *inter_sz = E > 0 ? (int *)safe_malloc((size_t)E * sizeof(int)) : NULL;
  int mC_v = 0, mC_e = 0;
  for (int e = 0; e < E; e++) {
    int kk = tree_edges[e].u, ll = tree_edges[e].v;
    int sz = 0;
    for (int w = 0; w < bw_q; w++)
      sz += __builtin_popcountll(cl_bs[kk][w] & cl_bs[ll][w]);
    inter_sz[e] = sz;
    mC_v += sz;
    mC_e += sz * (sz - 1) / 2;
  }
  int m_total = mA + mB + mC_v + mC_e;

  int constr_nnz_cap = K + 2 * mB + 2 * mC_v + 2 * mC_e;
  sdp->psd_cone_constraints =
      (psd_cone_constraint_t *)safe_malloc(sizeof(psd_cone_constraint_t));
  sdp->psd_cone_constraints->constr_ind =
      (int *)safe_malloc((size_t)constr_nnz_cap * sizeof(int));
  sdp->psd_cone_constraints->cone_ind =
      (int *)safe_malloc((size_t)constr_nnz_cap * sizeof(int));
  sdp->psd_cone_constraints->row_ind =
      (int *)safe_malloc((size_t)constr_nnz_cap * sizeof(int));
  sdp->psd_cone_constraints->col_ind =
      (int *)safe_malloc((size_t)constr_nnz_cap * sizeof(int));
  sdp->psd_cone_constraints->val =
      (double *)safe_malloc((size_t)constr_nnz_cap * sizeof(double));

  sdp->right_hand_side = (double *)safe_calloc((size_t)m_total, sizeof(double));

  int ci = 0;
  int row_id = 0;

  for (int k = 0; k < K; k++) {
    sdp->psd_cone_constraints->constr_ind[ci] = row_id;
    sdp->psd_cone_constraints->cone_ind[ci] = k;
    sdp->psd_cone_constraints->row_ind[ci] = 0;
    sdp->psd_cone_constraints->col_ind[ci] = 0;
    sdp->psd_cone_constraints->val[ci] = 1.0;
    ci++;
    sdp->right_hand_side[row_id++] = 1.0;
  }

  for (int k = 0; k < K; k++) {
    for (int idx = 0; idx < cliques[k].size; idx++) {
      int il = idx + 1;
      sdp->psd_cone_constraints->constr_ind[ci] = row_id;
      sdp->psd_cone_constraints->cone_ind[ci] = k;
      sdp->psd_cone_constraints->row_ind[ci] = il;
      sdp->psd_cone_constraints->col_ind[ci] = il;
      sdp->psd_cone_constraints->val[ci] = 1.0;
      ci++;
      sdp->psd_cone_constraints->constr_ind[ci] = row_id;
      sdp->psd_cone_constraints->cone_ind[ci] = k;
      sdp->psd_cone_constraints->row_ind[ci] = 0;
      sdp->psd_cone_constraints->col_ind[ci] = il;
      sdp->psd_cone_constraints->val[ci] = -0.5;
      ci++;
      row_id++;
    }
  }

  int *shared = (int *)safe_malloc((size_t)n * sizeof(int));
  for (int e = 0; e < E; e++) {
    int kk = tree_edges[e].u, ll = tree_edges[e].v;
    int sz = inter_sz[e];
    int s = 0;
    for (int w = 0; w < bw_q; w++) {
      uint64_t bits = cl_bs[kk][w] & cl_bs[ll][w];
      while (bits) {
        int bit = __builtin_ctzll(bits);
        shared[s++] = (w << 6) + bit;
        bits &= bits - 1;
      }
    }
    (void)sz;

    for (int i = 0; i < s; i++) {
      int v = shared[i];
      int ilk = pos_in[kk][v];
      int ill = pos_in[ll][v];
      sdp->psd_cone_constraints->constr_ind[ci] = row_id;
      sdp->psd_cone_constraints->cone_ind[ci] = kk;
      sdp->psd_cone_constraints->row_ind[ci] = ilk;
      sdp->psd_cone_constraints->col_ind[ci] = ilk;
      sdp->psd_cone_constraints->val[ci] = 1.0;
      ci++;
      sdp->psd_cone_constraints->constr_ind[ci] = row_id;
      sdp->psd_cone_constraints->cone_ind[ci] = ll;
      sdp->psd_cone_constraints->row_ind[ci] = ill;
      sdp->psd_cone_constraints->col_ind[ci] = ill;
      sdp->psd_cone_constraints->val[ci] = -1.0;
      ci++;
      row_id++;
    }

    for (int i = 0; i < s; i++) {
      for (int j = i + 1; j < s; j++) {
        int u = shared[i], v = shared[j];
        int uk = pos_in[kk][u], vk = pos_in[kk][v];
        int ul = pos_in[ll][u], vl = pos_in[ll][v];
        int rk = uk, ck = vk;
        if (rk > ck) {
          int t = rk;
          rk = ck;
          ck = t;
        }
        int rl = ul, cl = vl;
        if (rl > cl) {
          int t = rl;
          rl = cl;
          cl = t;
        }
        sdp->psd_cone_constraints->constr_ind[ci] = row_id;
        sdp->psd_cone_constraints->cone_ind[ci] = kk;
        sdp->psd_cone_constraints->row_ind[ci] = rk;
        sdp->psd_cone_constraints->col_ind[ci] = ck;
        sdp->psd_cone_constraints->val[ci] = 0.5;
        ci++;
        sdp->psd_cone_constraints->constr_ind[ci] = row_id;
        sdp->psd_cone_constraints->cone_ind[ci] = ll;
        sdp->psd_cone_constraints->row_ind[ci] = rl;
        sdp->psd_cone_constraints->col_ind[ci] = cl;
        sdp->psd_cone_constraints->val[ci] = -0.5;
        ci++;
        row_id++;
      }
    }
  }
  free(shared);

  sdp->m = m_total;
  sdp->nnz_psd_constr = ci;

  LOG_DBG("[QUBO->SDP] cones=%d  constraints=%d  obj nnz=%d  constr nnz=%d\n",
          K, m_total, obj_idx, ci);

  // Cleanup.
  for (int k = 0; k < K; k++) {
    free(cl_bs[k]);
    free(pos_in[k]);
    free(cliques[k].verts);
  }
  free(cl_bs);
  free(pos_in);
  free(cliques);
  free(tree_edges);
  free(inter_sz);
  free(home_vert);
  free(sigma);
  bs_free(g);

  return sdp;
}

basic_sdp_t *qubo_to_sdp_dense(const qubo_problem_t *q) {
  if (q == NULL || q->n <= 0) {
    fprintf(stderr, "qubo_to_sdp_dense: invalid QUBO (n=%d)\n",
            q ? q->n : -1);
    return NULL;
  }
  int n = q->n;
  LOG_DBG("[QUBO->SDP/dense] n=%d, single cone of size %d\n", n, n + 1);

  basic_sdp_t *sdp = (basic_sdp_t *)safe_malloc(sizeof(basic_sdp_t));
  sdp->n_cones = 1;
  sdp->blk_dims = (int *)safe_malloc(sizeof(int));
  sdp->blk_dims[0] = n + 1;
  sdp->lp_dim = 0;
  sdp->lp_constraints = NULL;
  sdp->lp_objective = NULL;
  sdp->nnz_lp_constr = 0;
  sdp->nnz_lp_obj = 0;

  double *diag = (double *)safe_calloc((size_t)n, sizeof(double));
  for (int k = 0; k < q->nnz; k++) {
    if (q->row[k] == q->col[k])
      diag[q->row[k]] += q->val[k];
  }
  if (q->linear) {
    for (int i = 0; i < n; i++)
      diag[i] += q->linear[i];
  }

  int n_diag_nnz = 0;
  for (int i = 0; i < n; i++)
    if (diag[i] != 0.0)
      n_diag_nnz++;
  int n_off_nnz = 0;
  for (int k = 0; k < q->nnz; k++)
    if (q->row[k] != q->col[k] && q->val[k] != 0.0)
      n_off_nnz++;
  int obj_cap = n_diag_nnz + n_off_nnz;

  sdp->psd_cone_objective =
      (psd_cone_objective_t *)safe_malloc(sizeof(psd_cone_objective_t));
  sdp->psd_cone_objective->cone_ind =
      (int *)safe_calloc((size_t)obj_cap, sizeof(int));
  sdp->psd_cone_objective->row_ind =
      (int *)safe_malloc((size_t)obj_cap * sizeof(int));
  sdp->psd_cone_objective->col_ind =
      (int *)safe_malloc((size_t)obj_cap * sizeof(int));
  sdp->psd_cone_objective->val =
      (double *)safe_malloc((size_t)obj_cap * sizeof(double));

  int oi = 0;
  for (int i = 0; i < n; i++) {
    if (diag[i] == 0.0)
      continue;
    sdp->psd_cone_objective->row_ind[oi] = i + 1;
    sdp->psd_cone_objective->col_ind[oi] = i + 1;
    sdp->psd_cone_objective->val[oi] = diag[i];
    oi++;
  }
  free(diag);
  for (int k = 0; k < q->nnz; k++) {
    int i = q->row[k], j = q->col[k];
    if (i == j || q->val[k] == 0.0)
      continue;
    if (i > j) {
      int t = i;
      i = j;
      j = t;
    }
    sdp->psd_cone_objective->row_ind[oi] = i + 1;
    sdp->psd_cone_objective->col_ind[oi] = j + 1;
    sdp->psd_cone_objective->val[oi] = q->val[k];
    oi++;
  }
  sdp->nnz_psd_obj = oi;

  int m_total = 1 + n;
  int constr_nnz_cap = 1 + 2 * n;
  sdp->psd_cone_constraints =
      (psd_cone_constraint_t *)safe_malloc(sizeof(psd_cone_constraint_t));
  sdp->psd_cone_constraints->constr_ind =
      (int *)safe_malloc((size_t)constr_nnz_cap * sizeof(int));
  sdp->psd_cone_constraints->cone_ind =
      (int *)safe_calloc((size_t)constr_nnz_cap, sizeof(int));
  sdp->psd_cone_constraints->row_ind =
      (int *)safe_malloc((size_t)constr_nnz_cap * sizeof(int));
  sdp->psd_cone_constraints->col_ind =
      (int *)safe_malloc((size_t)constr_nnz_cap * sizeof(int));
  sdp->psd_cone_constraints->val =
      (double *)safe_malloc((size_t)constr_nnz_cap * sizeof(double));
  sdp->right_hand_side = (double *)safe_calloc((size_t)m_total, sizeof(double));

  int ci = 0;
  int row_id = 0;

  sdp->psd_cone_constraints->constr_ind[ci] = row_id;
  sdp->psd_cone_constraints->row_ind[ci] = 0;
  sdp->psd_cone_constraints->col_ind[ci] = 0;
  sdp->psd_cone_constraints->val[ci] = 1.0;
  ci++;
  sdp->right_hand_side[row_id++] = 1.0;

  for (int i = 0; i < n; i++) {
    int il = i + 1;
    sdp->psd_cone_constraints->constr_ind[ci] = row_id;
    sdp->psd_cone_constraints->row_ind[ci] = il;
    sdp->psd_cone_constraints->col_ind[ci] = il;
    sdp->psd_cone_constraints->val[ci] = 1.0;
    ci++;
    sdp->psd_cone_constraints->constr_ind[ci] = row_id;
    sdp->psd_cone_constraints->row_ind[ci] = 0;
    sdp->psd_cone_constraints->col_ind[ci] = il;
    sdp->psd_cone_constraints->val[ci] = -0.5;
    ci++;
    row_id++;
  }

  sdp->m = m_total;
  sdp->nnz_psd_constr = ci;

  LOG_DBG("[QUBO->SDP/dense] m=%d  obj nnz=%d  constr nnz=%d\n", m_total, oi,
          ci);
  return sdp;
}

void free_qubo_layout(qubo_layout_t *l) {
  if (!l)
    return;
  free(l->home_cone);
  free(l->home_lidx);
  free(l);
}

qubo_layout_t *qubo_compute_layout(const qubo_problem_t *q, int chordal) {
  if (!q || q->n <= 0)
    return NULL;
  int n = q->n;
  qubo_layout_t *l = (qubo_layout_t *)safe_malloc(sizeof(qubo_layout_t));
  l->n = n;
  l->home_cone = (int *)safe_malloc((size_t)n * sizeof(int));
  l->home_lidx = (int *)safe_malloc((size_t)n * sizeof(int));

  if (!chordal) {
    l->n_cones = 1;
    for (int i = 0; i < n; i++) {
      l->home_cone[i] = 0;
      l->home_lidx[i] = i + 1;
    }
    return l;
  }

  bs_graph_t *g = bs_alloc(n);
  for (int k = 0; k < q->nnz; k++) {
    int i = q->row[k], j = q->col[k];
    if (i != j && q->val[k] != 0.0)
      bs_add_edge(g, i, j);
  }
  int *sigma = min_degree_elimination(g);
  int K = 0;
  clique_t *cliques = enumerate_maximal_cliques(g, sigma, &K);

  for (int i = 0; i < n; i++)
    l->home_cone[i] = -1;
  for (int k = 0; k < K; k++) {
    for (int idx = 0; idx < cliques[k].size; idx++) {
      int v = cliques[k].verts[idx];
      if (l->home_cone[v] == -1) {
        l->home_cone[v] = k;
        l->home_lidx[v] = idx + 1;
      }
    }
  }
  l->n_cones = K;

  for (int k = 0; k < K; k++)
    free(cliques[k].verts);
  free(cliques);
  free(sigma);
  bs_free(g);
  return l;
}
