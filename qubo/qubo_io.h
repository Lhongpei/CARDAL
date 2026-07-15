/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#pragma once

#include "qubo.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  QUBO_FMT_AUTO = 0,    // sniff from content
  QUBO_FMT_DWAVE = 1,   // D-Wave qbsolv .qubo with "p qubo 0 n nDiag nOff" header
  QUBO_FMT_TRIPLET = 2, // "n nnz\\n i j v\\n..." (e.g. OR-Library Beasley bqp)
  QUBO_FMT_QPLIB = 3,   // QPLIB 2014/2019 .qplib (binary QP subset only)
} qubo_file_format_t;

qubo_problem_t *qubo_read_file(const char *path, qubo_file_format_t fmt);

#ifdef __cplusplus
}
#endif
