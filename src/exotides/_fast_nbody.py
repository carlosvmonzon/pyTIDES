"""
exotides/_fast_nbody.py

Numba-accelerated float64-only Newtonian N-body Taylor-coefficient
computation -- functionally identical to
``exotides.nbody._nbody_mincseries_core`` with ``include_pn=False``, but with
the generic mixed-type (float64 vs. gmpy2.mpfr) series-arithmetic helpers
from ``exotides/core.py`` (which branch dynamically on type at every call and
aren't JIT-compatible) replaced by inlined float64-only convolution loops.

Numba is an *optional* dependency (mirrors the existing ``HAS_GMPY2``
pattern in ``exotides/core.py``): if it isn't installed, ``HAS_NUMBA`` is
False and callers fall back to the pure-Python core transparently --
nothing in this module is imported/executed in that case.

Scoped to the plain-Newtonian path only (no 1PN correction) -- that keeps
using the pure-Python core, which can still express any physics choice;
this module exists purely for the common-case speed-up.

Copyright (C) 2026  Carlos Vazquez Monzon

This file is part of pyTIDES (the ``exotides`` package).

pyTIDES is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version.

pyTIDES is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along
with pyTIDES. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:

    @njit(cache=True)
    def _mul_mc(u, v, k):
        w = u[0] * v[k]
        for j in range(1, k + 1):
            w += u[j] * v[k - j]
        return w

    @njit(cache=True)
    def _pow_mc_c(u, c, w, k):
        if k == 0:
            return u[0] ** c
        if u[0] != 0.0:
            ww = c * k * w[0] * u[k]
            for j in range(1, k):
                ww += (c * (k - j) - j) * w[j] * u[k - j]
            return ww / (k * u[0])
        return 0.0

    @njit(cache=True)
    def _fast_newtonian_core(XX_flat, stride, VAR, N, ORDER, G, masses):
        for i in range(ORDER):
            # 1. Per-pair auxiliary series: dx, dy, dz, r^2, r^-3.
            for j in range(N):
                for k in range(j + 1, N):
                    pair_idx = j * (2 * N - 1 - j) // 2 + k - j - 1
                    base = VAR + 5 * pair_idx

                    dx_row = (base + 1) * stride
                    dy_row = (base + 2) * stride
                    dz_row = (base + 3) * stride
                    r2_row = (base + 4) * stride
                    r3inv_row = (base + 5) * stride

                    u_dx = XX_flat[dx_row: dx_row + stride]
                    u_dy = XX_flat[dy_row: dy_row + stride]
                    u_dz = XX_flat[dz_row: dz_row + stride]
                    u_r2 = XX_flat[r2_row: r2_row + stride]
                    u_r3inv = XX_flat[r3inv_row: r3inv_row + stride]

                    xj = (6 * j + 1) * stride
                    xk = (6 * k + 1) * stride
                    yj = (6 * j + 2) * stride
                    yk = (6 * k + 2) * stride
                    zj = (6 * j + 3) * stride
                    zk = (6 * k + 3) * stride

                    u_dx[i] = XX_flat[xk + i] - XX_flat[xj + i]
                    u_dy[i] = XX_flat[yk + i] - XX_flat[yj + i]
                    u_dz[i] = XX_flat[zk + i] - XX_flat[zj + i]

                    u_r2[i] = _mul_mc(u_dx, u_dx, i) + _mul_mc(u_dy, u_dy, i) + _mul_mc(u_dz, u_dz, i)
                    u_r3inv[i] = _pow_mc_c(u_r2, -1.5, u_r3inv, i)

            # 2. Acceleration accumulation and next-order state.
            for j in range(N):
                ax = 0.0
                ay = 0.0
                az = 0.0
                for k in range(N):
                    if k == j:
                        continue
                    pj = j if j < k else k
                    pk = k if j < k else j
                    pair_idx = pj * (2 * N - 1 - pj) // 2 + pk - pj - 1
                    base = VAR + 5 * pair_idx

                    u_dx = XX_flat[(base + 1) * stride: (base + 1) * stride + stride]
                    u_dy = XX_flat[(base + 2) * stride: (base + 2) * stride + stride]
                    u_dz = XX_flat[(base + 3) * stride: (base + 3) * stride + stride]
                    u_r3inv = XX_flat[(base + 5) * stride: (base + 5) * stride + stride]

                    sign = 1.0 if j < k else -1.0
                    factor = G * masses[k]

                    ax += sign * factor * _mul_mc(u_dx, u_r3inv, i)
                    ay += sign * factor * _mul_mc(u_dy, u_r3inv, i)
                    az += sign * factor * _mul_mc(u_dz, u_r3inv, i)

                inext = i + 1
                XX_flat[(6 * j + 1) * stride + inext] = XX_flat[(6 * j + 4) * stride + i] / inext
                XX_flat[(6 * j + 2) * stride + inext] = XX_flat[(6 * j + 5) * stride + i] / inext
                XX_flat[(6 * j + 3) * stride + inext] = XX_flat[(6 * j + 6) * stride + i] / inext
                XX_flat[(6 * j + 4) * stride + inext] = ax / inext
                XX_flat[(6 * j + 5) * stride + inext] = ay / inext
                XX_flat[(6 * j + 6) * stride + inext] = az / inext

    _cache = {}

    def run_fast_newtonian(t, v, p, XVAR, ORDER, MO):
        """
        Full setup/call/extract cycle for the Numba Newtonian core --
        callable directly as a TIDES ``mincseries_func``. Maintains its own
        static buffer (keyed by shape), independent of
        ``exotides.nbody._nbody_mincseries_core``'s buffer (which reserves
        extra slots for the PN path this module doesn't need).
        """
        VAR = len(v)
        N = VAR // 6
        P = N * (N - 1) // 2
        TT = VAR + 5 * P
        stride = MO + 1

        key = (TT, MO)
        buf = _cache.get(key)
        if buf is None or buf.shape[0] != (TT + 1) * stride:
            buf = np.empty((TT + 1) * stride, dtype=np.float64)
            _cache[key] = buf
        XX_flat = buf
        XX_flat.fill(0.0)

        XX_flat[0] = t
        XX_flat[1] = 1.0
        for idx in range(1, VAR + 1):
            XX_flat[idx * stride] = v[idx - 1]

        G = float(p[0])
        masses = np.asarray(p[1:1 + N], dtype=np.float64)

        _fast_newtonian_core(XX_flat, stride, VAR, N, ORDER, G, masses)

        for j in range(VAR + 1):
            XVAR[:ORDER + 1, j] = XX_flat[j * stride: j * stride + ORDER + 1]
