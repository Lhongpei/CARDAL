/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NPY_DTYPE_UNKNOWN = 0,
  NPY_DTYPE_F64,
  NPY_DTYPE_F32,
  NPY_DTYPE_I64,
  NPY_DTYPE_I32,
  NPY_DTYPE_U8,
  NPY_DTYPE_I8,
} npy_dtype_t;

typedef struct {
  char name[256];     // filename inside the archive (without trailing .npy)
  npy_dtype_t dtype;
  int n_dim;          // 1 or 2 (higher-dim ignored / flattened)
  long long shape[2]; // shape[0] = rows; shape[1] = cols (1 if 1D)
  int fortran_order;  // 1 if column-major
  void *data;         // raw buffer in dtype-sized elements
  long long n_elements;
} npz_entry_t;

typedef struct {
  int n_entries;
  npz_entry_t *entries;
} npz_archive_t;

npz_archive_t *npz_read(const char *path);

const npz_entry_t *npz_find(const npz_archive_t *arc, const char *name);

void npz_free(npz_archive_t *arc);

#ifdef __cplusplus
}
#endif
