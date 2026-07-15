/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "parser.h"
#include "utils.h"
#include <ctype.h>
#include <string.h>
#include <zlib.h>

static char *read_file_to_memory_transparent(const char *filename,
                                             size_t *out_size) {
  gzFile file = gzopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error: Cannot open %s\n", filename);
    return NULL;
  }

  size_t capacity = 10 * 1024 * 1024;
  size_t total_read = 0;
  char *buffer = (char *)safe_malloc(capacity);

  while (true) {
    if (total_read + 1024 * 1024 > capacity) {
      capacity *= 2;
      buffer = (char *)realloc(buffer, capacity);
    }
    int bytes = gzread(file, buffer + total_read, 1024 * 1024);
    if (bytes < 0) {
      fprintf(stderr, "Decompression error in %s\n", filename);
      free(buffer);
      gzclose(file);
      return NULL;
    }
    if (bytes == 0)
      break;
    total_read += bytes;
  }
  gzclose(file);
  *out_size = total_read;
  return buffer;
}

static sdp_format_t detect_format_from_memory(const char *data, size_t size) {
  if (size < 128)
    return FORMAT_UNKNOWN;

  if (memcmp(data, "MATLAB 5.0", 10) == 0) {
    return FORMAT_MAT;
  }

  if (memcmp(data, "VER.", 4) == 0) {
    return FORMAT_CBF;
  }

  const char *p = data;
  while (p < data + size && isspace(*p))
    p++;
  if (p < data + size && (*p == '"' || *p == '*' || isdigit(*p))) {
    return FORMAT_SDPA;
  }

  return FORMAT_UNKNOWN;
}

static int has_suffix(const char *s, const char *suf) {
  size_t ls = strlen(s), lt = strlen(suf);
  if (lt > ls)
    return 0;
  return strcmp(s + ls - lt, suf) == 0;
}

basic_sdp_t *sdp_problem_parse(const char *filename) {
  if (has_suffix(filename, ".npz")) {
    LOG_DBG("[Parser] Logic: PDSDP-style NPZ\n");
    basic_sdp_t *sdp = read_pdsdp_npz(filename);
    if (sdp && sdp->m > 0) {
      LOG_DBG("[Parser] Successfully loaded problem: m=%d, n_cones=%d\n",
              sdp->m, sdp->n_cones);
    }
    return sdp;
  }

  size_t size = 0;
  char *data = read_file_to_memory_transparent(filename, &size);
  if (!data)
    return NULL;

  sdp_format_t format = detect_format_from_memory(data, size);
  basic_sdp_t *sdp = NULL;

  switch (format) {
  case FORMAT_SDPA:
    LOG_DBG("[Parser] Logic: SDPA (Text/GZ)\n");
    sdp = parse_sdpa_from_memory(data, size);
    break;

  case FORMAT_MAT:
    LOG_DBG("[Parser] Logic: MATLAB v5 Container\n");
    sdp = handle_mat_from_memory(filename, data, size);
    break;

  case FORMAT_CBF:
    LOG_DBG("[Parser] Logic: CBF (Conic Benchmark)\n");
    break;

  default:
    fprintf(stderr, "[Error] Unknown format for: %s\n", filename);
    break;
  }

  free(data);

  if (sdp && sdp->m > 0) {
    LOG_DBG("[Parser] Successfully loaded problem: m=%d, n_cones=%d\n", sdp->m,
           sdp->n_cones);
  }

  return sdp;
}

void free_basic_sdp(basic_sdp_t *sdp) {
  if (!sdp)
    return;

  if (sdp->psd_cone_constraints) {
    free(sdp->psd_cone_constraints->constr_ind);
    free(sdp->psd_cone_constraints->cone_ind);
    free(sdp->psd_cone_constraints->row_ind);
    free(sdp->psd_cone_constraints->col_ind);
    free(sdp->psd_cone_constraints->val);
    free(sdp->psd_cone_constraints);
  }

  if (sdp->psd_cone_objective) {
    free(sdp->psd_cone_objective->cone_ind);
    free(sdp->psd_cone_objective->row_ind);
    free(sdp->psd_cone_objective->col_ind);
    free(sdp->psd_cone_objective->val);
    free(sdp->psd_cone_objective);
  }

  if (sdp->lp_constraints) {
    free(sdp->lp_constraints->row_ind);
    free(sdp->lp_constraints->col_ind);
    free(sdp->lp_constraints->val);
    free(sdp->lp_constraints);
  }

  free(sdp->lp_objective);
  free(sdp->right_hand_side);
  free(sdp->blk_dims);
  free(sdp);
}