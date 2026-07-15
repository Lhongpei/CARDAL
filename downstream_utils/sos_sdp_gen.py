#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

"""Generate SOS-SDP from FCIDUMP integrals per Low et al. PRX 15, 041016 (2025).

Implements the spin-free level-2 SDP from Appendix F:
  - Generators: O_SF = sum_{ij} (g_ij ph_{ji} + gbar_ij hp_{ji})
                O_Dsigma = sum_i d_isigma a_isigma
                O_Qsigma = sum_i q_isigma a_isigma^dag
  - H_SOS = O_SF^dag O_SF + sum_sigma (O_Dsigma^dag O_Dsigma + O_Qsigma^dag O_Qsigma)
  - Variables: G_SF (2N^2 x 2N^2, PSD), D (N x N, PSD), Q (N x N, PSD)
  - Maximize E_SOS = 4 sum_pq G'''^{pp}_{qq} + 2 Tr(Q)
  - Subject to:
      * 2-body match (Eq. F9): G^{ik}_{lj} - G'^{ik}_{jl} - G''^{ki}_{lj} + G'''^{ki}_{jl}
        Sum over (i,k,j,l) of operator a^dag_isigma a_ksigma a^dag_jtau a_ltau coefficient
        equals (1/2) h2_{i,k,j,l}   (chemistry notation (ik|jl))
      * 1-body match (Eq. F10): coefficient of a^dag_isigma a_jsigma equals h1_ij + 1/2 sum_r h2_{i,r,r,j}
      * constant absorbed into E_SOS
Output: SDPA dat-s.

Sign convention: cardal's parser flips F0 sign internally; we want to MAXIMIZE E_SOS,
and SDPA standard is "min <C,X> s.t. <Ai,X>=bi, X>=0", so set C = -dE_SOS/dX
"""
import sys, os, re, struct
import numpy as np


# ----- FCIDUMP parser -----

def parse_fcidump(path):
    """Return (N, h0, h1[N,N], h2[N,N,N,N]) with 8-fold symmetry expanded."""
    with open(path) as f:
        # Header
        header = ""
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF")
            header += line
            if "&END" in line or "/" == line.strip():
                break
        m = re.search(r"NORB\s*=\s*(\d+)", header)
        if not m:
            raise ValueError("Cannot find NORB")
        N = int(m.group(1))
        h0 = 0.0
        h1 = np.zeros((N, N))
        h2 = np.zeros((N, N, N, N))
        # Body lines: val p q r s
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                val = float(parts[0])
                p, q, r, s = [int(x) for x in parts[1:5]]
            except ValueError:
                continue
            if p == 0 and q == 0 and r == 0 and s == 0:
                h0 = val
            elif r == 0 and s == 0:
                # 1-electron h1[p-1, q-1]
                i, j = p - 1, q - 1
                h1[i, j] = val
                h1[j, i] = val
            else:
                # 2-electron, chemistry notation (pq|rs) symmetric under
                # (p<->q), (r<->s), (pq)<->(rs). Expand 8-fold.
                i, j, k, l = p - 1, q - 1, r - 1, s - 1
                # set all 8 orbits
                for (a, b, c, d) in [(i, j, k, l), (j, i, k, l),
                                    (i, j, l, k), (j, i, l, k),
                                    (k, l, i, j), (l, k, i, j),
                                    (k, l, j, i), (l, k, j, i)]:
                    h2[a, b, c, d] = val
    return N, h0, h1, h2


# ----- SDP builder -----

class SdpaWriter:
    """Builds SDPA dat-s format incrementally with sparse F matrices.

    SDPA format:
      m
      nblocks
      block_sizes (space-separated; negative = LP)
      b vector (length m)
      For each (mat, blk, i, j) nonzero:
        mat blk i j val
      where mat = 0..m, blk = 1..nblocks, i,j 1-based (UPPER triangle only).
    """

    def __init__(self, m, block_sizes):
        self.m = m
        self.block_sizes = list(block_sizes)
        self.nblocks = len(block_sizes)
        self.b = [0.0] * m
        self.entries = []  # list of (mat, blk, i, j, val) with i<=j, 1-based

    def set_b(self, i, val):
        self.b[i] = val

    def add_entry(self, mat, blk, i, j, val):
        """mat 0..m, blk 1..nblocks, i,j 0-based; we store upper-triangle 1-based.

        Caller passes a "raw" coefficient meaning <F, X> should pick up val * X[i,j]
        (where i,j are the FULL symmetric matrix indices). SDPA convention expands
        upper-tri file entries to (r,c) AND (c,r), giving 2·F[r,c]·X[r,c] for off-diag.
        So we store val/2 on off-diag (and aggregate by summing val/2 for repeated entries),
        or val on diagonal.
        """
        if i == j:
            stored = val
        else:
            if i > j:
                i, j = j, i
            stored = val * 0.5
        self.entries.append((mat, blk, i + 1, j + 1, stored))

    def write(self, path):
        # Combine duplicate entries (same mat,blk,i,j) by summing values
        combined = {}
        for mat, blk, i, j, val in self.entries:
            key = (mat, blk, i, j)
            combined[key] = combined.get(key, 0.0) + val
        with open(path, "w") as f:
            f.write(f"{self.m}\n")
            f.write(f"{self.nblocks}\n")
            f.write(" ".join(str(s) for s in self.block_sizes) + "\n")
            f.write(" ".join(f"{x:.16g}" for x in self.b) + "\n")
            for (mat, blk, i, j), val in combined.items():
                if val == 0.0:
                    continue
                f.write(f"{mat} {blk} {i} {j} {val:.16g}\n")


def build_sdp(N, h0, h1, h2, out_path):
    """Generate the SOS SDP per Low et al. Appendix F.

    Block layout (SDPA blocks 1..3):
      blk 1: G_SF, size 2N^2.
             Linearized index ph(i,k)=i*N+k for first half (0..N^2),
             linearized index hp(k,i)=N^2 + k*N+i for second half.
      blk 2: D, size N.   (symmetric PSD; for spin-free D^{i,sigma}_{j,sigma} same for both sigma)
      blk 3: Q, size N.

    Variables (entries of X) on which constraints are linear:
      X[blk1, a, b] = (G_SF)[a,b]
      X[blk2, i, j] = D[i,j]
      X[blk3, i, j] = Q[i,j]

    Objective: maximize E_SOS
      E_SOS = 4 sum_{p,q} G'''^{pp}_{qq} + 2 sum_p Q[p,p]
            = 4 sum_{p,q} G_SF[N^2 + p*N + p, N^2 + q*N + q]   (G''' is bottom-right sub-block)
            + 2 sum_p Q[p,p]

    In SDPA primal (min <F0, X>), we want min -E_SOS:
      F0 = matrix with -4 at positions (N^2+p*N+p, N^2+q*N+q) of blk 1 (G''' diagonal corners)
                       -2 at positions (p,p) of blk 3 (Q diagonal).
    Note: SDPA's internal sign flip on F0 (the parser inverts the sign at load) means
    we should write +4 and +2 here; cardal internally will treat -F0 as the actual C. We
    confirm convention by mirroring biomedP setup.

    Two-body constraint (Eq. F9):
      For each tuple (i, k, j, l):
        G^{ik}_{lj} - G'^{ik}_{jl} - G''^{ki}_{lj} + G'''^{ki}_{jl} = (1/2) h2[i,k,j,l]

      In G_SF blocks (using my linearized layout):
        G^{ik}_{lj}     = G_SF[ph(i,k), ph(l,j)]   = G_SF[i*N+k, l*N+j]
        G'^{ik}_{jl}    = G_SF[ph(i,k), hp(j,l)]   = G_SF[i*N+k, N^2 + j*N+l]
        G''^{ki}_{lj}   = G_SF[hp(k,i), ph(l,j)]   = G_SF[N^2 + k*N+i, l*N+j]
        G'''^{ki}_{jl}  = G_SF[hp(k,i), hp(j,l)]   = G_SF[N^2 + k*N+i, N^2 + j*N+l]

    One-body constraint (Eq. F10):
      For each (i, j):
        Coefficient of a^dag_isigma a_jsigma after spin trace = h1[i,j] + 0.5 sum_r h2[i,r,r,j]

      Per paper Eq. F10 (after careful reading and translating the indices), the coefficient is:
        sum over p of (2 G'^{ij}_{pp} + G''^{pp}_{ji} - G'''^{pp}_{ij} - G'''^{ji}_{pp})
        + 0.5 (D[i,j] - Q[j,i])

      We use the spin-free convention where D and Q are N x N symmetric (so D[i,j]=D[j,i]).
      Map to my blocks:
        G'^{ij}_{pp}    = G_SF[i*N+j, N^2 + p*N+p]
        G''^{pp}_{ji}   = G_SF[N^2 + p*N+p, j*N+i]
        G'''^{pp}_{ij}  = G_SF[N^2 + p*N+p, N^2 + i*N+j]
        G'''^{ji}_{pp}  = G_SF[N^2 + j*N+i, N^2 + p*N+p]
        D[i,j]          = X[blk2, i, j]
        Q[j,i]          = X[blk3, j, i]    (== Q[i,j] for symmetric Q)

    Constant term:
      E_SOS = h0 - g00 contribution; since we picked E_SOS = 4 sum G'''_pp,qq + 2 Tr Q
      and constant part of H is h0, we have additional constraint:
        h0 + 0 = 0   (no constant variable; absorb into the gap reporting)
      Actually paper writes E_SOS just from G''' and Q only; h0 is added back after.
    """
    NN = N * N
    L = 2 * NN  # G_SF size
    block_sizes = [L, N, N]  # blk 1: G_SF, blk 2: D, blk 3: Q

    # Constraint enumeration:
    #   Two-body: m_2 = N^4 unique constraints (we don't dedupe by symmetry to keep code simple)
    #   One-body: m_1 = N^2 constraints
    # In SDPA, we collect upper-triangular only and use symmetric entries.
    m_2 = N * N * N * N
    m_1 = N * N
    m = m_2 + m_1
    w = SdpaWriter(m, block_sizes)

    # --- Objective F0 ---
    # SOS lower bound: H - E_SOS*I = O_SF^dag G O_SF + sum D + sum Q  (PSD on RHS)
    # Constant part of RHS = 4 sum_{p,q} G'''^{pp}_{qq} + 2 sum_p Q[p,p]  (>= 0)
    # Constant part of LHS = h0 - E_SOS
    # So E_SOS = h0 - (4 sum G''' + 2 Tr Q). Maximizing E_SOS means MINIMIZING the cost.
    # cardal's sdpa_parser.c:212 stores internal_C = -F0_file, then solves min <internal_C, X>.
    # We want min <internal_C, X> = 4 sum G''' + 2 Tr Q  =>  internal_C has +4/+2 at diagonals
    # =>  F0_file has -4/-2.
    # Returned pobj == cost; E_SOS = h0 - pobj.
    for p in range(N):
        for q in range(N):
            r = NN + p * N + p
            c = NN + q * N + q
            w.add_entry(0, 1, r, c, -4.0)
    for p in range(N):
        w.add_entry(0, 3, p, p, -2.0)

    # --- Two-body constraints (Eq. F9, with F5 per-block sub-indexing) ---
    # Coefficient of Σ_στ a†_iσ a_kσ a†_jτ a_lτ = E_ik E_jl in H_SOS^(2):
    #   G^{ik}_{lj} - G'^{ik}_{jl} - G''^{ki}_{lj} + G'''^{ki}_{jl}
    #   (each block's (super)_(sub) follows F5; F9's shorthand "(...)^{ik}_{lj}" is loose.)
    # Match: = (1/2) h2[i,k,j,l] for chemistry H written in (E, EE) basis.
    ci = 0
    for i in range(N):
        for k in range(N):
            for j in range(N):
                for l in range(N):
                    # +G^{ik}_{lj} = +X[i*N+k, l*N+j]
                    w.add_entry(ci + 1, 1, i * N + k, l * N + j, 1.0)
                    # -G'^{ik}_{jl} = -X[i*N+k, NN+j*N+l]   (G' col label (j,l) → NN+j*N+l)
                    w.add_entry(ci + 1, 1, i * N + k, NN + j * N + l, -1.0)
                    # -G''^{ki}_{lj} = -X[NN+k*N+i, l*N+j] (G'' row label (k,i) → NN+k*N+i)
                    w.add_entry(ci + 1, 1, NN + k * N + i, l * N + j, -1.0)
                    # +G'''^{ki}_{jl} = +X[NN+k*N+i, NN+j*N+l]
                    w.add_entry(ci + 1, 1, NN + k * N + i, NN + j * N + l, 1.0)
                    w.set_b(ci, 0.5 * h2[i, k, j, l])
                    ci += 1
    assert ci == m_2

    # --- One-body constraints (Eq. F10) ---
    # Paper: H_SOS^(1) = 2 Σ_ij Σ_σ [Σ_p (G'^{ij}_{pp} + G''^{pp}_{ji} - G'''^{pp}_{ij}
    #                              - G'''^{ji}_{pp}) + (1/2)(D^{iσ}_{jσ} - Q^{iσ}_{jσ})] a†_jσ a_iσ
    # Σ_σ a†_jσ a_iσ = E_ji  (σ-indep G terms get factor 2 via Σ_σ on the operator).
    # For spin-free D, Q (D^{iσ}_{jσ}=D_ij both σ): Σ_σ(1/2)(D-Q)·a†_jσ a_iσ = (1/2)(D-Q) E_ji.
    # So coefficient of E_ji in H_SOS^(1):
    #   2·Σ_p (G'^{ij}_{pp} + G''^{pp}_{ji} - G'''^{pp}_{ij} - G'''^{ji}_{pp}) + (D_ij - Q_ij)
    # Match to chemistry H^(1) (e_pqrs→EE shift): coefficient of E_ji = h1[i,j] - 0.5 Σ_r h2[i,r,r,j].
    for i in range(N):
        for j in range(N):
            for p in range(N):
                # +2 G'^{ij}_{pp}  = +2 X[i*N+j, NN+p*N+p]
                w.add_entry(ci + 1, 1, i * N + j, NN + p * N + p, 2.0)
                # +2 G''^{pp}_{ji} = +2 X[NN+p*N+p, j*N+i]
                w.add_entry(ci + 1, 1, NN + p * N + p, j * N + i, 2.0)
                # -2 G'''^{pp}_{ij} = -2 X[NN+p*N+p, NN+i*N+j]
                w.add_entry(ci + 1, 1, NN + p * N + p, NN + i * N + j, -2.0)
                # -2 G'''^{ji}_{pp} = -2 X[NN+j*N+i, NN+p*N+p]
                w.add_entry(ci + 1, 1, NN + j * N + i, NN + p * N + p, -2.0)
            # +D[i,j]
            w.add_entry(ci + 1, 2, i, j, 1.0)
            # -Q[i,j]
            w.add_entry(ci + 1, 3, i, j, -1.0)
            # Chemist FCIDUMP convention: 2-body op is (1/2)(pq|rs) a^dag_p a^dag_r a_s a_q
            # (the e_pqrs operator). To match E_ik E_jl in the SOS basis, use
            # E_pq E_rs = e_pqrs + delta_qr E_ps, which shifts h1 by -(1/2) sum_r (ir|rj).
            rhs = h1[i, j] - 0.5 * sum(h2[i, r, r, j] for r in range(N))
            w.set_b(ci, rhs)
            ci += 1
    assert ci == m

    w.write(out_path)
    print(f"  wrote {out_path}")
    print(f"  m = {m}, blocks = {block_sizes}, h0 = {h0}")
    return h0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: sos_sdp_gen.py <fcidump> <out.dat-s>")
        sys.exit(1)
    fcidump = sys.argv[1]
    out = sys.argv[2]
    print(f"reading {fcidump}")
    N, h0, h1, h2 = parse_fcidump(fcidump)
    print(f"  N = {N} orbitals, h0 = {h0}")
    print(f"  |h1|_max = {np.max(np.abs(h1)):.4f}, |h2|_max = {np.max(np.abs(h2)):.4f}")
    build_sdp(N, h0, h1, h2, out)
