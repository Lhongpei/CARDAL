/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "npz_reader.h"
#include "utils.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

// =============================================================================
// Tiny ZIP-archive reader (Stored / Deflate) sufficient for numpy savez output.
//
// Layout of a ZIP file (we only need a subset):
//   <local_file_header_1><file_data_1>... <local_file_header_n><file_data_n>
//   <central_dir_entry_1>... <central_dir_entry_n>
//   <end_of_central_dir_record>
// We scan for EOCDR at end of file, then walk the central directory, then for
// each entry seek to its local file header and decompress its data.
// =============================================================================

static uint32_t le32(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}
static uint16_t le16(const unsigned char *p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

#define EOCDR_SIG 0x06054b50u
#define CDIR_SIG 0x02014b50u
#define LFH_SIG 0x04034b50u

typedef struct {
  char name[256];
  uint16_t method;
  uint32_t comp_size;
  uint32_t uncomp_size;
  uint32_t local_header_off;
} zip_entry_t;

static int read_whole_file(const char *path, unsigned char **out_buf,
                           long *out_sz) {
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    fprintf(stderr, "npz_reader: cannot open '%s': %s\n", path,
            strerror(errno));
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  long sz = ftell(fp);
  if (sz < 22) { // smaller than EOCDR
    fclose(fp);
    fprintf(stderr, "npz_reader: '%s' too small (%ld bytes)\n", path, sz);
    return -1;
  }
  rewind(fp);
  unsigned char *buf = (unsigned char *)safe_malloc((size_t)sz);
  if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
    fclose(fp);
    free(buf);
    fprintf(stderr, "npz_reader: short read on '%s'\n", path);
    return -1;
  }
  fclose(fp);
  *out_buf = buf;
  *out_sz = sz;
  return 0;
}

// Find EOCDR by scanning backwards. Returns offset, or -1.
static long find_eocdr(const unsigned char *buf, long sz) {
  // EOCDR is at least 22 bytes; can have up to 65535-byte comment after.
  long min_off = (sz - (long)((1L << 16) + 22) > 0) ? sz - ((1L << 16) + 22) : 0;
  for (long i = sz - 22; i >= min_off; i--) {
    if (le32(buf + i) == EOCDR_SIG)
      return i;
  }
  return -1;
}

static int parse_zip(const unsigned char *buf, long sz, zip_entry_t **out_entries,
                     int *out_count) {
  long eocdr = find_eocdr(buf, sz);
  if (eocdr < 0) {
    fprintf(stderr, "npz_reader: EOCDR not found\n");
    return -1;
  }
  uint16_t total = le16(buf + eocdr + 10);
  uint32_t cdir_size = le32(buf + eocdr + 12);
  uint32_t cdir_off = le32(buf + eocdr + 16);
  (void)cdir_size;

  zip_entry_t *ents = (zip_entry_t *)safe_calloc(total, sizeof(zip_entry_t));
  long off = cdir_off;
  for (int i = 0; i < total; i++) {
    if (off + 46 > sz || le32(buf + off) != CDIR_SIG) {
      fprintf(stderr,
              "npz_reader: bad central directory entry #%d at off %ld\n", i,
              off);
      free(ents);
      return -1;
    }
    uint16_t method = le16(buf + off + 10);
    uint32_t comp_size = le32(buf + off + 20);
    uint32_t uncomp_size = le32(buf + off + 24);
    uint16_t name_len = le16(buf + off + 28);
    uint16_t extra_len = le16(buf + off + 30);
    uint16_t comment_len = le16(buf + off + 32);
    uint32_t lfh_off = le32(buf + off + 42);

    if (name_len >= sizeof(ents[i].name)) {
      fprintf(stderr, "npz_reader: filename too long (%u bytes)\n", name_len);
      free(ents);
      return -1;
    }
    memcpy(ents[i].name, buf + off + 46, name_len);
    ents[i].name[name_len] = '\0';
    ents[i].method = method;
    ents[i].comp_size = comp_size;
    ents[i].uncomp_size = uncomp_size;
    ents[i].local_header_off = lfh_off;

    off += 46 + name_len + extra_len + comment_len;
  }
  *out_entries = ents;
  *out_count = total;
  return 0;
}

static int read_entry_data(const unsigned char *buf, long sz,
                           const zip_entry_t *ent, unsigned char **out_data) {
  long off = ent->local_header_off;
  if (off + 30 > sz || le32(buf + off) != LFH_SIG) {
    fprintf(stderr, "npz_reader: bad local file header for '%s'\n",
            ent->name);
    return -1;
  }
  uint16_t name_len = le16(buf + off + 26);
  uint16_t extra_len = le16(buf + off + 28);
  long data_off = off + 30 + name_len + extra_len;
  if (data_off + ent->comp_size > (uint32_t)sz) {
    fprintf(stderr, "npz_reader: '%s' data extends past EOF\n", ent->name);
    return -1;
  }

  unsigned char *out = (unsigned char *)safe_malloc(ent->uncomp_size + 1);
  if (ent->method == 0) {
    if (ent->comp_size != ent->uncomp_size) {
      fprintf(stderr,
              "npz_reader: STORED entry '%s' size mismatch %u != %u\n",
              ent->name, ent->comp_size, ent->uncomp_size);
      free(out);
      return -1;
    }
    memcpy(out, buf + data_off, ent->comp_size);
  } else if (ent->method == 8) {
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    if (inflateInit2(&strm, -15) != Z_OK) {
      fprintf(stderr, "npz_reader: inflateInit2 failed for '%s'\n", ent->name);
      free(out);
      return -1;
    }
    strm.next_in = (Bytef *)(buf + data_off);
    strm.avail_in = ent->comp_size;
    strm.next_out = (Bytef *)out;
    strm.avail_out = ent->uncomp_size;
    int rc = inflate(&strm, Z_FINISH);
    inflateEnd(&strm);
    if (rc != Z_STREAM_END) {
      fprintf(stderr, "npz_reader: inflate failed (rc=%d) for '%s'\n", rc,
              ent->name);
      free(out);
      return -1;
    }
  } else {
    fprintf(stderr, "npz_reader: unsupported method %u for '%s'\n",
            ent->method, ent->name);
    free(out);
    return -1;
  }
  *out_data = out;
  return 0;
}

// =============================================================================
// .npy header parser. Format:
//   bytes  0..5 : magic "\x93NUMPY"
//   byte   6    : version_major
//   byte   7    : version_minor
//   bytes  8..9 (v1) or 8..11 (v2/3) : little-endian header_len
//   bytes ...   : ASCII Python-dict header, e.g.
//                   {'descr': '<f8', 'fortran_order': False, 'shape': (3, 5), }
//   then raw data.
// =============================================================================

static int parse_npy_header(const unsigned char *data, long data_sz,
                            npz_entry_t *out, long *out_data_off) {
  if (data_sz < 10 || memcmp(data, "\x93NUMPY", 6) != 0) {
    fprintf(stderr, "npz_reader: bad NPY magic for '%s'\n", out->name);
    return -1;
  }
  unsigned major = data[6];
  long hlen_off, hlen;
  if (major == 1) {
    hlen = (long)le16(data + 8);
    hlen_off = 10;
  } else if (major == 2 || major == 3) {
    hlen = (long)le32(data + 8);
    hlen_off = 12;
  } else {
    fprintf(stderr, "npz_reader: unknown NPY major %u for '%s'\n", major,
            out->name);
    return -1;
  }
  long header_end = hlen_off + hlen;
  if (header_end > data_sz) {
    fprintf(stderr, "npz_reader: NPY header truncated for '%s'\n", out->name);
    return -1;
  }
  const char *hdr = (const char *)(data + hlen_off);
  // Look for descr
  const char *p = strstr(hdr, "'descr'");
  if (!p)
    p = strstr(hdr, "\"descr\"");
  if (!p) {
    fprintf(stderr, "npz_reader: no descr field for '%s'\n", out->name);
    return -1;
  }
  p = strchr(p, ':');
  if (!p)
    return -1;
  p++;
  while (*p == ' ' || *p == '\t')
    p++;
  char quote = *p;
  if (quote != '\'' && quote != '"') {
    fprintf(stderr, "npz_reader: bad descr quote for '%s'\n", out->name);
    return -1;
  }
  p++;
  char descr[16];
  int di = 0;
  while (*p != quote && di < (int)sizeof(descr) - 1)
    descr[di++] = *p++;
  descr[di] = '\0';

  out->dtype = NPY_DTYPE_UNKNOWN;
  // Accept '<', '|', or '=' byte order (we don't byteswap; assume little
  // endian host).
  if (strcmp(descr, "<f8") == 0 || strcmp(descr, "|f8") == 0)
    out->dtype = NPY_DTYPE_F64;
  else if (strcmp(descr, "<f4") == 0 || strcmp(descr, "|f4") == 0)
    out->dtype = NPY_DTYPE_F32;
  else if (strcmp(descr, "<i8") == 0 || strcmp(descr, "|i8") == 0)
    out->dtype = NPY_DTYPE_I64;
  else if (strcmp(descr, "<i4") == 0 || strcmp(descr, "|i4") == 0)
    out->dtype = NPY_DTYPE_I32;
  else if (strcmp(descr, "|u1") == 0 || strcmp(descr, "|b1") == 0 ||
           strcmp(descr, "<u1") == 0)
    out->dtype = NPY_DTYPE_U8; // also covers numpy bool
  else if (strcmp(descr, "|i1") == 0 || strcmp(descr, "<i1") == 0)
    out->dtype = NPY_DTYPE_I8;
  // Anything else (strings, structured, complex, big-endian, ...) is left as
  // NPY_DTYPE_UNKNOWN; the entry is still discoverable by name but its data
  // payload is not loaded. This lets metadata-style entries (e.g. a 'mode'
  // string) coexist with the actual numerical arrays.

  // fortran_order
  out->fortran_order = 0;
  if (strstr(hdr, "'fortran_order': True") ||
      strstr(hdr, "\"fortran_order\": True"))
    out->fortran_order = 1;

  // shape: (a, b) or (a,) or (a, b, c) -> we keep first two dims and flatten.
  const char *sp = strstr(hdr, "'shape'");
  if (!sp)
    sp = strstr(hdr, "\"shape\"");
  if (!sp) {
    fprintf(stderr, "npz_reader: no shape field for '%s'\n", out->name);
    return -1;
  }
  sp = strchr(sp, '(');
  if (!sp) {
    fprintf(stderr, "npz_reader: bad shape paren for '%s'\n", out->name);
    return -1;
  }
  sp++;
  out->n_dim = 0;
  out->shape[0] = 1;
  out->shape[1] = 1;
  long long total_elems = 1;
  while (out->n_dim < 2 && *sp && *sp != ')') {
    while (*sp == ' ' || *sp == '\t' || *sp == ',')
      sp++;
    if (!isdigit((unsigned char)*sp))
      break;
    char *end;
    long long v = strtoll(sp, &end, 10);
    sp = end;
    out->shape[out->n_dim++] = v;
    total_elems *= v;
  }
  // If there are extra dims, multiply them into n_elements but ignore for
  // shape (the consumer can iterate flat).
  while (*sp && *sp != ')') {
    while (*sp == ' ' || *sp == '\t' || *sp == ',')
      sp++;
    if (!isdigit((unsigned char)*sp))
      break;
    char *end;
    long long v = strtoll(sp, &end, 10);
    sp = end;
    total_elems *= v;
  }
  if (out->n_dim == 0)
    out->n_dim = 1;

  out->n_elements = total_elems;
  *out_data_off = header_end;
  return 0;
}

static int dtype_size(npy_dtype_t d) {
  switch (d) {
  case NPY_DTYPE_F64:
  case NPY_DTYPE_I64:
    return 8;
  case NPY_DTYPE_F32:
  case NPY_DTYPE_I32:
    return 4;
  case NPY_DTYPE_U8:
  case NPY_DTYPE_I8:
    return 1;
  default:
    return 0;
  }
}

npz_archive_t *npz_read(const char *path) {
  unsigned char *zbuf = NULL;
  long zsz = 0;
  if (read_whole_file(path, &zbuf, &zsz) != 0)
    return NULL;

  zip_entry_t *ents = NULL;
  int n_ent = 0;
  if (parse_zip(zbuf, zsz, &ents, &n_ent) != 0) {
    free(zbuf);
    return NULL;
  }

  npz_archive_t *arc = (npz_archive_t *)safe_malloc(sizeof(npz_archive_t));
  arc->n_entries = n_ent;
  arc->entries = (npz_entry_t *)safe_calloc(n_ent, sizeof(npz_entry_t));

  for (int i = 0; i < n_ent; i++) {
    // Strip ".npy" suffix from name for friendly lookup
    const char *zip_name = ents[i].name;
    size_t L = strlen(zip_name);
    size_t stripped =
        (L >= 4 && strcmp(zip_name + L - 4, ".npy") == 0) ? L - 4 : L;
    if (stripped >= sizeof(arc->entries[i].name))
      stripped = sizeof(arc->entries[i].name) - 1;
    memcpy(arc->entries[i].name, zip_name, stripped);
    arc->entries[i].name[stripped] = '\0';

    unsigned char *raw = NULL;
    if (read_entry_data(zbuf, zsz, &ents[i], &raw) != 0) {
      free(zbuf);
      free(ents);
      npz_free(arc);
      return NULL;
    }
    long data_off = 0;
    if (parse_npy_header(raw, ents[i].uncomp_size, &arc->entries[i],
                         &data_off) != 0) {
      free(raw);
      free(zbuf);
      free(ents);
      npz_free(arc);
      return NULL;
    }
    int es = dtype_size(arc->entries[i].dtype);
    if (es == 0) {
      // Unknown dtype: skip data payload but keep the entry so callers can
      // detect presence by name. Useful for metadata like 'mode' strings.
      arc->entries[i].data = NULL;
      free(raw);
      continue;
    }
    long n_bytes = (long)arc->entries[i].n_elements * es;
    if (data_off + n_bytes > ents[i].uncomp_size) {
      fprintf(stderr,
              "npz_reader: data of '%s' exceeds entry size "
              "(data_off=%ld + %ld > %u)\n",
              arc->entries[i].name, data_off, n_bytes, ents[i].uncomp_size);
      free(raw);
      free(zbuf);
      free(ents);
      npz_free(arc);
      return NULL;
    }
    arc->entries[i].data = safe_malloc((size_t)n_bytes);
    memcpy(arc->entries[i].data, raw + data_off, (size_t)n_bytes);
    free(raw);
  }
  free(zbuf);
  free(ents);
  return arc;
}

const npz_entry_t *npz_find(const npz_archive_t *arc, const char *name) {
  if (!arc)
    return NULL;
  for (int i = 0; i < arc->n_entries; i++)
    if (strcmp(arc->entries[i].name, name) == 0)
      return &arc->entries[i];
  return NULL;
}

void npz_free(npz_archive_t *arc) {
  if (!arc)
    return;
  for (int i = 0; i < arc->n_entries; i++)
    free(arc->entries[i].data);
  free(arc->entries);
  free(arc);
}
