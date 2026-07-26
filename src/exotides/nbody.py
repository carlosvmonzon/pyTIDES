"""
exotides/nbody.py

N-body dynamics for the TIDES integrator.

Contents
--------
- nbody_mincseries  - TIDES-compatible Taylor-series generator for the
                      gravitational N-body equations of motion.
- pack_state        - flatten (positions, velocities) → 1-D state vector.
- unpack_state      - split 1-D state vector → (positions, velocities).
- compute_energy    - total mechanical energy (T + V) for N-body systems.

Initial conditions built here are always plain float64 -- to integrate at
arbitrary precision, pass is_mpfr=True to exotides.core.TidesSolver, the
sole place precision is chosen (it converts v_init/p_init via
exotides.core.to_mpfr regardless of the dtype they were built in here).

See also exotides.core for as_float64/vector_norm, the mpfr-aware array
helpers used throughout this module's callers (plotting, tests).
"""

import logging

import numpy as np
from .core import mul_mc, pow_mc_c, HAS_GMPY2, gmpy2, vector_norm
from .relativity import has_pn_params, pn_speed_of_light
from . import _fast_nbody

# Library code logs, it doesn't print: an unconfigured logger is silent by
# default (Python's logging convention), so this never writes to stdout
# unless the calling application opts in, e.g. via
# ``logging.basicConfig(level=logging.INFO)``.
logger = logging.getLogger(__name__)


# Per-pair auxiliary Taylor series slots: dx,dy,dz,r2,r3inv (Newtonian,
# always used) plus dvx,dvy,dvz,rinv,rv_dot,D,E,C (1PN-only, see
# _nbody_mincseries_core's PN block and exotides/relativity.py). Always
# reserved regardless of include_pn so the shared static buffer never needs
# resizing when switching between Newtonian/PN calls in the same process.
_PAIR_SLOTS = 13


# ---------------------------------------------------------------------------
# TIDES N-body series generator
# ---------------------------------------------------------------------------

def _nbody_mincseries_core(t, v, p, XVAR, ORDER, MO, include_pn=False):
    """
    TIDES-compatible Taylor-series generator for the N-body problem.

    Supports both standard ``float64`` arrays and object arrays of
    ``gmpy2.mpfr`` for arbitrary multiple precision.

    State layout (per body j, 0-indexed):
        v[6j+0] = x_j,  v[6j+1] = y_j,  v[6j+2] = z_j
        v[6j+3] = vx_j, v[6j+4] = vy_j, v[6j+5] = vz_j

    Parameters
    ----------
    t : scalar
        Current time.
    v : array-like, length 6*N
        Current state.
    p : array-like, length 1+N (+1 if include_pn)
        Parameters: p[0]=G, p[1..N]=masses, plus an optional trailing speed
        of light (exotides/relativity.py) if include_pn.
    XVAR : ndarray, shape (MO+1, nvar+1)
        Output array filled with Taylor coefficients.
    ORDER : int
        Number of Taylor orders to compute.
    MO : int
        Maximum order (pre-allocated buffer size hint).
    include_pn : bool
        If True, add the pairwise "dominant mass" 1PN correction (see
        exotides/relativity.py) to the lighter body of each pair.
    """
    VAR = len(v)
    N   = VAR // 6
    if not hasattr(_nbody_mincseries_core, "_last_reported_n") or _nbody_mincseries_core._last_reported_n != N:
        logger.info("%d bodies involved", N)
        _nbody_mincseries_core._last_reported_n = N
    P   = N * (N - 1) // 2
    TT  = VAR + _PAIR_SLOTS * P

    stride = MO + 1

    is_mpfr = HAS_GMPY2 and isinstance(v[0], gmpy2.mpfr)
    dtype   = object if is_mpfr else np.float64

    # Buffer estático: se reutiliza entre llamadas para evitar reservar memoria
    # en cada paso del integrador. Solo crece si cambia el tamaño necesario.
    cache = _nbody_mincseries_core
    if (
        not hasattr(cache, "_XX_flat")
        or cache._XX_flat is None
        or TT > cache._alloc_TT
        or MO > cache._alloc_MO
        or not hasattr(cache, "_alloc_dtype")
        or cache._alloc_dtype is not dtype
    ):
        cache._XX_flat = np.empty((TT + 1) * stride, dtype=dtype)
        cache._alloc_TT = TT
        cache._alloc_MO = MO
        cache._alloc_dtype = dtype

    XX_flat = cache._XX_flat
    if is_mpfr:
        XX_flat.fill(gmpy2.mpfr("0.0"))
    else:
        XX_flat.fill(0.0)

    # Serie temporal: t(tau) = t0 + tau, por eso el coeficiente de orden 1 es 1.
    XX_flat[0] = t
    XX_flat[1] = gmpy2.mpfr("1.0") if is_mpfr else 1.0   # d(t)/dt = 1

    # Estado inicial en orden 0: posiciones y velocidades actuales.
    for idx in range(1, VAR + 1):
        XX_flat[idx * stride] = v[idx - 1]

    G = p[0]

    for i in range(ORDER):
        # ------------------------------------------------------------------
        # 1. Para cada pareja se construyen dx, dy, dz, r^2 y r^-3.
        #    Esas series auxiliares se reutilizan al sumar aceleraciones.
        # ------------------------------------------------------------------
        for j in range(N):
            for k in range(j + 1, N):
                pair_idx = j * (2 * N - 1 - j) // 2 + k - j - 1
                base = VAR + _PAIR_SLOTS * pair_idx

                dx_row   = base + 1
                dy_row   = base + 2
                dz_row   = base + 3
                r2_row   = base + 4
                r3inv_row = base + 5

                u_dx    = XX_flat[dx_row    * stride : (dx_row    + 1) * stride]
                u_dy    = XX_flat[dy_row    * stride : (dy_row    + 1) * stride]
                u_dz    = XX_flat[dz_row    * stride : (dz_row    + 1) * stride]
                u_r2    = XX_flat[r2_row    * stride : (r2_row    + 1) * stride]
                u_r3inv = XX_flat[r3inv_row * stride : (r3inv_row + 1) * stride]

                u_xj = XX_flat[(6 * j + 1) * stride : (6 * j + 2) * stride]
                u_xk = XX_flat[(6 * k + 1) * stride : (6 * k + 2) * stride]
                u_yj = XX_flat[(6 * j + 2) * stride : (6 * j + 3) * stride]
                u_yk = XX_flat[(6 * k + 2) * stride : (6 * k + 3) * stride]
                u_zj = XX_flat[(6 * j + 3) * stride : (6 * j + 4) * stride]
                u_zk = XX_flat[(6 * k + 3) * stride : (6 * k + 4) * stride]

                u_dx[i] = u_xk[i] - u_xj[i]
                u_dy[i] = u_yk[i] - u_yj[i]
                u_dz[i] = u_zk[i] - u_zj[i]

                u_r2[i] = (
                    mul_mc(u_dx, u_dx, i)
                    + mul_mc(u_dy, u_dy, i)
                    + mul_mc(u_dz, u_dz, i)
                )

                u_r3inv[i] = pow_mc_c(u_r2, -1.5, u_r3inv, i)

                if include_pn:
                    # 1PN auxiliary series (see exotides/relativity.py for the
                    # derivation): relative velocity, r^-1, v.r_vec (rv_dot),
                    # D = 4*GM_heavy/r - v^2, E = D*r^-3, C = rv_dot*r^-3.
                    # v^2 itself doesn't need its own persistent slot -- it's
                    # only ever used (as a scalar) to build D at this same
                    # order, never as an array input to a later convolution.
                    u_dvx  = XX_flat[(base +  6) * stride : (base +  7) * stride]
                    u_dvy  = XX_flat[(base +  7) * stride : (base +  8) * stride]
                    u_dvz  = XX_flat[(base +  8) * stride : (base +  9) * stride]
                    u_rinv = XX_flat[(base +  9) * stride : (base + 10) * stride]
                    u_rvd  = XX_flat[(base + 10) * stride : (base + 11) * stride]
                    u_d    = XX_flat[(base + 11) * stride : (base + 12) * stride]
                    u_e    = XX_flat[(base + 12) * stride : (base + 13) * stride]
                    u_c    = XX_flat[(base + 13) * stride : (base + 14) * stride]

                    u_vxj = XX_flat[(6 * j + 4) * stride : (6 * j + 5) * stride]
                    u_vxk = XX_flat[(6 * k + 4) * stride : (6 * k + 5) * stride]
                    u_vyj = XX_flat[(6 * j + 5) * stride : (6 * j + 6) * stride]
                    u_vyk = XX_flat[(6 * k + 5) * stride : (6 * k + 6) * stride]
                    u_vzj = XX_flat[(6 * j + 6) * stride : (6 * j + 7) * stride]
                    u_vzk = XX_flat[(6 * k + 6) * stride : (6 * k + 7) * stride]

                    u_dvx[i] = u_vxk[i] - u_vxj[i]
                    u_dvy[i] = u_vyk[i] - u_vyj[i]
                    u_dvz[i] = u_vzk[i] - u_vzj[i]

                    u_rinv[i] = pow_mc_c(u_r2, -0.5, u_rinv, i)

                    vv_i = (
                        mul_mc(u_dvx, u_dvx, i)
                        + mul_mc(u_dvy, u_dvy, i)
                        + mul_mc(u_dvz, u_dvz, i)
                    )
                    u_rvd[i] = (
                        mul_mc(u_dvx, u_dx, i)
                        + mul_mc(u_dvy, u_dy, i)
                        + mul_mc(u_dvz, u_dz, i)
                    )

                    gm_heavy = G * (p[1 + j] if p[1 + j] >= p[1 + k] else p[1 + k])
                    u_d[i] = 4.0 * gm_heavy * u_rinv[i] - vv_i

                    u_e[i] = mul_mc(u_d, u_r3inv, i)
                    u_c[i] = mul_mc(u_rvd, u_r3inv, i)

        # ------------------------------------------------------------------
        # 2. Suma de aceleraciones gravitatorias y escritura del siguiente
        #    coeficiente de Taylor para cada variable de estado.
        # ------------------------------------------------------------------
        if include_pn:
            if not has_pn_params(p, N):
                raise ValueError(
                    "include_pn=True requires p to carry the trailing speed-of-light "
                    "parameter -- see exotides.relativity.append_pn_params"
                )
            pn_c = pn_speed_of_light(p, N)

        for j in range(N):
            ax = gmpy2.mpfr("0.0") if is_mpfr else 0.0
            ay = gmpy2.mpfr("0.0") if is_mpfr else 0.0
            az = gmpy2.mpfr("0.0") if is_mpfr else 0.0

            for k in range(N):
                if k == j:
                    continue
                p_j = j if j < k else k
                p_k = k if j < k else j
                pair_idx  = p_j * (2 * N - 1 - p_j) // 2 + p_k - p_j - 1
                base = VAR + _PAIR_SLOTS * pair_idx

                dx_row    = base + 1
                dy_row    = base + 2
                dz_row    = base + 3
                r3inv_row = base + 5

                u_dx    = XX_flat[dx_row    * stride : (dx_row    + 1) * stride]
                u_dy    = XX_flat[dy_row    * stride : (dy_row    + 1) * stride]
                u_dz    = XX_flat[dz_row    * stride : (dz_row    + 1) * stride]
                u_r3inv = XX_flat[r3inv_row * stride : (r3inv_row + 1) * stride]

                sign   = 1.0 if j < k else -1.0
                factor = G * p[1 + k]

                ax += sign * factor * mul_mc(u_dx,    u_r3inv, i)
                ay += sign * factor * mul_mc(u_dy,    u_r3inv, i)
                az += sign * factor * mul_mc(u_dz,    u_r3inv, i)

                if include_pn and p[1 + j] <= p[1 + k]:
                    # Only the lighter body of the pair gets the correction
                    # (the "dominant mass" approximation -- see
                    # exotides/relativity.py). Tied masses get it both ways,
                    # which degrades gracefully into a symmetric-ish
                    # approximation rather than an arbitrary one-sided pick.
                    u_dvx = XX_flat[(base +  6) * stride : (base +  7) * stride]
                    u_dvy = XX_flat[(base +  7) * stride : (base +  8) * stride]
                    u_dvz = XX_flat[(base +  8) * stride : (base +  9) * stride]
                    u_e   = XX_flat[(base + 12) * stride : (base + 13) * stride]
                    u_c   = XX_flat[(base + 13) * stride : (base + 14) * stride]

                    gm_heavy = G * (p[1 + j] if p[1 + j] >= p[1 + k] else p[1 + k])
                    pn_scale = (-sign) * gm_heavy / (pn_c * pn_c)

                    ax += pn_scale * (mul_mc(u_e, u_dx, i) + 4.0 * mul_mc(u_c, u_dvx, i))
                    ay += pn_scale * (mul_mc(u_e, u_dy, i) + 4.0 * mul_mc(u_c, u_dvy, i))
                    az += pn_scale * (mul_mc(u_e, u_dz, i) + 4.0 * mul_mc(u_c, u_dvz, i))

            inext = i + 1

            # Derivadas cinemáticas: dx/dt = vx, dy/dt = vy, dz/dt = vz.
            XX_flat[(6 * j + 1) * stride + inext] = XX_flat[(6 * j + 4) * stride + i] / inext
            XX_flat[(6 * j + 2) * stride + inext] = XX_flat[(6 * j + 5) * stride + i] / inext
            XX_flat[(6 * j + 3) * stride + inext] = XX_flat[(6 * j + 6) * stride + i] / inext

            # Derivadas dinámicas: dv/dt = aceleración gravitatoria.
            XX_flat[(6 * j + 4) * stride + inext] = ax / inext
            XX_flat[(6 * j + 5) * stride + inext] = ay / inext
            XX_flat[(6 * j + 6) * stride + inext] = az / inext

    # Copia únicamente tiempo y variables de estado al formato que espera
    # TidesSolver; las series auxiliares quedan como detalle interno.
    for j in range(VAR + 1):
        XVAR[:ORDER + 1, j] = XX_flat[j * stride : j * stride + ORDER + 1]


def nbody_mincseries(t, v, p, XVAR, ORDER, MO, use_numba=None):
    """
    TIDES-compatible Taylor-series generator for the gravitational N-body
    problem.

    This keeps the original point-mass Newtonian dynamics. Use
    ``nbody_pn_mincseries`` when the parameter vector includes a 1PN
    correction.

    ``use_numba`` explicitly selects the float64 Numba-JIT core
    (``exotides/_fast_nbody.py``) instead of the pure-Python one --
    numerically identical either way, just faster when available:

    - ``None`` (default): use it when available and the state isn't
      arbitrary-precision (``gmpy2.mpfr``).
    - ``True``: force it. Raises ``ValueError`` if Numba isn't installed, or
      if ``v`` is already ``gmpy2.mpfr`` -- Numba's JIT only compiles
      fixed-width native types, not arbitrary-precision Python objects, so
      that combination is never possible, not just undesirable. This
      function only sees the state vector it was handed, so it can't
      silently downgrade it to float64 -- callers that want "Numba wins,
      just use double precision" should decide that before building ``v``
      (see ``HierarchicalSystemTemplates.solve_nbody``, which does exactly
      that: ``use_numba=True`` overrides ``is_mpfr`` to ``False`` there).
    - ``False``: force the pure-Python core even when Numba is available
      (e.g. to benchmark against it, or work around a JIT issue).
    """
    is_mpfr = HAS_GMPY2 and isinstance(v[0], gmpy2.mpfr)
    if use_numba is None:
        use_numba = not is_mpfr and _fast_nbody.HAS_NUMBA
    elif use_numba:
        if is_mpfr:
            raise ValueError(
                "use_numba=True is incompatible with arbitrary-precision "
                "(gmpy2.mpfr) state -- Numba cannot JIT-compile mpfr objects"
            )
        if not _fast_nbody.HAS_NUMBA:
            raise ValueError("use_numba=True requested but numba is not installed")

    # Reporta precisión/backend una sola vez por combinación (igual que el
    # aviso de nº de cuerpos en _nbody_mincseries_core), no en cada llamada
    # -- este generador se invoca una vez por orden de Taylor por paso.
    settings = (is_mpfr, use_numba)
    if getattr(nbody_mincseries, "_last_reported_settings", None) != settings:
        precision = "mpfr (arbitrary precision)" if is_mpfr else "float64 (double precision)"
        backend = "numba" if use_numba else "pure Python"
        logger.info("precision=%s, backend=%s", precision, backend)
        nbody_mincseries._last_reported_settings = settings

    if use_numba:
        return _fast_nbody.run_fast_newtonian(t, v, p, XVAR, ORDER, MO)
    return _nbody_mincseries_core(t, v, p, XVAR, ORDER, MO)


def nbody_pn_mincseries(t, v, p, XVAR, ORDER, MO):
    """
    TIDES-compatible Taylor-series generator for N-body dynamics with the
    pairwise "dominant mass" 1PN relativistic correction (see
    exotides/relativity.py).

    Parameter layout:
        p = [G, m0, ..., mN-1, c]
    """
    return _nbody_mincseries_core(t, v, p, XVAR, ORDER, MO, include_pn=True)


# ---------------------------------------------------------------------------
# State packing / unpacking helpers
# ---------------------------------------------------------------------------

def pack_state(positions, velocities):
    """
    Flatten (positions, velocities) arrays into a single state vector.

    Parameters
    ----------
    positions : array-like, shape (N, 3)
    velocities : array-like, shape (N, 3)

    Returns
    -------
    ndarray, shape (6*N,)
        Interleaved as [x0,y0,z0,vx0,vy0,vz0, x1,y1,z1, ...].
    """
    positions  = np.asarray(positions)
    velocities = np.asarray(velocities)
    N = len(positions)

    is_mpfr = HAS_GMPY2 and (
        any(isinstance(x, gmpy2.mpfr) for x in positions.flat)
        or any(isinstance(x, gmpy2.mpfr) for x in velocities.flat)
    )
    dtype = object if is_mpfr else np.float64

    # Intercala coordenadas y velocidades por cuerpo para coincidir con el
    # convenio usado en el código C de TIDES.
    flat = np.empty(6 * N, dtype=dtype)
    for i in range(N):
        flat[6 * i     : 6 * i + 3] = positions[i]
        flat[6 * i + 3 : 6 * i + 6] = velocities[i]
    return flat


def unpack_state(flat_state):
    """
    Split a flat state vector into position and velocity arrays.

    Parameters
    ----------
    flat_state : array-like
        Either shape (6*N,) for a single snapshot or (steps, 6*N) for a
        trajectory.

    Returns
    -------
    positions : ndarray
        Shape (N, 3) or (steps, N, 3).
    velocities : ndarray
        Shape (N, 3) or (steps, N, 3).
    """
    flat_state = np.asarray(flat_state)
    is_mpfr    = HAS_GMPY2 and any(isinstance(x, gmpy2.mpfr) for x in flat_state.flat)
    dtype      = object if is_mpfr else np.float64

    # Admite tanto un único estado como una trayectoria completa.
    if flat_state.ndim == 1:
        N          = len(flat_state) // 6
        positions  = np.empty((N, 3), dtype=dtype)
        velocities = np.empty((N, 3), dtype=dtype)
        for i in range(N):
            positions[i]  = flat_state[6 * i     : 6 * i + 3]
            velocities[i] = flat_state[6 * i + 3 : 6 * i + 6]
        return positions, velocities

    steps, VAR = flat_state.shape
    N          = VAR // 6
    positions  = np.empty((steps, N, 3), dtype=dtype)
    velocities = np.empty((steps, N, 3), dtype=dtype)
    for i in range(N):
        positions[:, i, :]  = flat_state[:, 6 * i     : 6 * i + 3]
        velocities[:, i, :] = flat_state[:, 6 * i + 3 : 6 * i + 6]
    return positions, velocities


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------

def compute_energy(state, masses, G):
    """
    Total mechanical energy E = T + V of an N-body system.

    Parameters
    ----------
    state : ndarray
        Shape (6*N,) for a single snapshot or (steps, 6*N) for a trajectory.
    masses : array-like, length N
        Masses of the bodies.
    G : scalar
        Gravitational constant.

    Returns
    -------
    ndarray of length *steps* (or a scalar for a single snapshot).
    """
    # Normaliza a formato trayectoria para usar el mismo cálculo en ambos casos.
    is_trajectory = state.ndim == 2
    if not is_trajectory:
        state = state[np.newaxis, :]

    steps, VAR = state.shape
    N          = VAR // 6

    is_mpfr = HAS_GMPY2 and isinstance(state[0, 0], gmpy2.mpfr)
    dtype   = object if is_mpfr else np.float64

    T = np.empty(steps, dtype=dtype)
    V = np.empty(steps, dtype=dtype)
    T.fill(gmpy2.mpfr("0.0") if is_mpfr else 0.0)
    V.fill(gmpy2.mpfr("0.0") if is_mpfr else 0.0)

    for i in range(N):
        # Energía cinética de cada cuerpo.
        vx = state[:, 6 * i + 3]
        vy = state[:, 6 * i + 4]
        vz = state[:, 6 * i + 5]
        T += 0.5 * masses[i] * (vx**2 + vy**2 + vz**2)

        # Energía potencial gravitatoria de cada pareja, contada una sola vez.
        for j in range(i + 1, N):
            dx = state[:, 6 * j]     - state[:, 6 * i]
            dy = state[:, 6 * j + 1] - state[:, 6 * i + 1]
            dz = state[:, 6 * j + 2] - state[:, 6 * i + 2]

            r = vector_norm(np.stack([dx, dy, dz], axis=-1), axis=-1)
            V -= G * masses[i] * masses[j] / r

    energy = T + V
    return energy if is_trajectory else energy[0]
