"""
exotides/core.py

Pure Python implementation of the TIDES (Taylor Integration of Differential EquationS)
Minimal Double and Multiple Precision Integrator.

Contents
--------
- to_mpfr, as_float64, vector_norm - mpfr/float64 precision helpers, at the
  scalar (to_mpfr) and array (as_float64, vector_norm) level -- shared by
  every mincseries_func (N-body, Kepler, Lorenz, ...), not just N-body, so
  they live here rather than in exotides.nbody.
- Type-agnostic scalar math helpers (_exp, _log, _sin, _cos)
- Taylor series algebra functions:
    mul_mc, div_mc, inv_mc, exp_mc, pow_mc_c, log_mc, sin_mc, cos_mc
- TidesSolver - variable-stepsize, variable-order Taylor integrator that
  supports both standard float64 (double precision) and arbitrary-precision
  floats via gmpy2 (multiple precision).

This file is a Python port of the Taylor Series Method core of the original
C TIDES library (libTIDES/dp_tides.c, mp_tides.c, minc_tides.c -- see
../C/libTIDES), from the TIDES project (http://gme.unizar.es/software/tides):

    Copyright (C) 2010  A. Abad, R. Barrio, F. Blesa, M. Rodriguez
    Grupo de Mecanica Espacial, University of Zaragoza, SPAIN
    Contact: <tides@unizar.es>

Python port and subsequent modifications:

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

import math
import numpy as np

# ---------------------------------------------------------------------------
# Optional multiple-precision back-end
# ---------------------------------------------------------------------------

try:
    import gmpy2
    HAS_GMPY2 = True
except ImportError:
    # gmpy2 es opcional: si no está instalado, el paquete sigue funcionando
    # con float64 y desactiva únicamente el modo de multiprecisión.
    HAS_GMPY2 = False

    class gmpy2:  # noqa: N801 - dummy stub
        """Stub so that ``isinstance(x, gmpy2.mpfr)`` is always False."""
        mpfr = type(None)


def to_mpfr(value):
    """Convert ``value`` to ``gmpy2.mpfr`` at the currently configured precision.

    Goes through ``str(value)`` rather than a direct ``gmpy2.mpfr(value)``
    so a plain Python float's usual decimal literal is what gets
    reinterpreted at the target precision, instead of that float's exact
    (but visually surprising) binary64 value padded with zero bits.
    ``gmpy2.get_context().precision`` controls the target precision --
    unset, that's gmpy2's own default. Used internally by ``TidesSolver``
    to convert ``v_init``/``p_init`` up to arbitrary precision when
    ``is_mpfr=True``, regardless of what dtype they were built in.
    """
    if not HAS_GMPY2:
        raise ValueError("to_mpfr requires gmpy2 to be installed")
    return gmpy2.mpfr(str(value))


def as_float64(array: np.ndarray) -> np.ndarray:
    """Cast an mpfr (``dtype=object``) array to plain ``float64``.

    ``array`` is ``dtype=object`` holding ``gmpy2.mpfr`` values when the
    system was integrated at arbitrary precision (``is_mpfr=True``). Several
    things downstream of ``exotides.nbody.unpack_state``/``compute_energy``
    can't handle that directly:

    - NumPy ufuncs (``np.sqrt``, and anything built on it like
      ``np.linalg.norm``) called on an ``object`` array look for a
      ``.sqrt()`` *method* on each element, which ``gmpy2.mpfr`` doesn't
      have (unlike the module-level ``gmpy2.sqrt(x)``), so they raise
      instead of computing anything.
    - matplotlib's 3D machinery (``mpl_toolkits.mplot3d``) calls NumPy
      ufuncs like ``np.isfinite`` directly on plotted data during
      rendering.

    None of that is precision display/analysis code could show or check
    anyway, so casting right after ``unpack_state`` -- before any further
    plain-NumPy arithmetic -- loses nothing while sidestepping all of the
    above. A no-op (returns ``array`` unchanged) when it's already a
    float dtype.
    """
    return array.astype(np.float64) if array.dtype == object else array


def vector_norm(vec, axis=-1):
    """
    Euclidean norm along ``axis``, for either plain ``float64`` or
    ``gmpy2.mpfr`` (``dtype=object``) arrays.

    ``np.linalg.norm`` calls ``np.sqrt`` internally, which -- like any
    NumPy ufunc -- looks for a ``.sqrt()`` *method* on each element when
    given an ``object``-dtype array; ``gmpy2.mpfr`` doesn't have one (only
    the module-level ``gmpy2.sqrt(x)``), so it raises instead of computing
    anything (the same issue ``as_float64`` works around for
    display/plotting code). This dispatches to an elementwise
    ``gmpy2.sqrt`` in that case instead, so quantities derived from an
    ``is_mpfr=True`` integration (e.g. body separations) can be measured
    without dropping to ``float64`` first -- unlike ``as_float64``, this
    keeps full precision.

    Parameters
    ----------
    vec : array-like
        E.g. shape (3,) for one vector, or (steps, 3) for a trajectory of
        them (with ``axis=1``/``-1``, the default).
    axis : int, optional
        Axis to reduce over.

    Returns
    -------
    scalar or ndarray
    """
    vec = np.asarray(vec)
    is_mpfr = HAS_GMPY2 and vec.dtype == object and any(
        isinstance(x, gmpy2.mpfr) for x in vec.flat
    )
    if not is_mpfr:
        return np.linalg.norm(vec, axis=axis)

    sq_sum = np.sum(vec * vec, axis=axis)
    if np.ndim(sq_sum) == 0:
        return gmpy2.sqrt(sq_sum)
    return np.array([gmpy2.sqrt(x) for x in sq_sum], dtype=object)


# ---------------------------------------------------------------------------
# Type-agnostic scalar math helpers
# ---------------------------------------------------------------------------

def _exp(x):
    if HAS_GMPY2 and isinstance(x, gmpy2.mpfr):
        return gmpy2.exp(x)
    return math.exp(x)


def _log(x):
    if HAS_GMPY2 and isinstance(x, gmpy2.mpfr):
        return gmpy2.log(x)
    return math.log(x)


def _sin(x):
    if HAS_GMPY2 and isinstance(x, gmpy2.mpfr):
        return gmpy2.sin(x)
    return math.sin(x)


def _cos(x):
    if HAS_GMPY2 and isinstance(x, gmpy2.mpfr):
        return gmpy2.cos(x)
    return math.cos(x)


# ---------------------------------------------------------------------------
# Taylor series algebra functions
# ---------------------------------------------------------------------------

def mul_mc(u, v, k):
    """
    k-th Taylor coefficient of W = U * V  (discrete convolution).

    Parameters
    ----------
    u, v : array-like
        Coefficient arrays of U and V (length >= k+1).
    k : int
        Target order.

    Returns
    -------
    scalar
        The k-th coefficient of the product series.
    """
    # Producto de series de Taylor: el coeficiente k se obtiene por
    # convolución de todos los pares de órdenes que suman k.
    w = u[0] * v[k]
    for j in range(1, k + 1):
        w += u[j] * v[k - j]
    return w


def div_mc(u, v, w, k):
    """
    k-th Taylor coefficient of W = U / V.

    Parameters
    ----------
    u, v : array-like
        Coefficient arrays of numerator U and denominator V.
    w : array-like
        Already-computed coefficients of the quotient W (orders 0..k-1).
    k : int
        Target order.

    Returns
    -------
    scalar

    Raises
    ------
    ZeroDivisionError
        If v[0] == 0.
    """
    # División recurrente: usa los coeficientes de w ya calculados en órdenes
    # inferiores para despejar el nuevo coeficiente.
    if v[0] == 0.0:
        raise ZeroDivisionError("div_mc: denominator leading coefficient v[0] is zero")
    ww = u[k]
    for j in range(1, k + 1):
        ww -= v[j] * w[k - j]
    return ww / v[0]


def inv_mc(p, u, w, k):
    """
    k-th Taylor coefficient of W = p / U.

    Parameters
    ----------
    p : scalar
        Constant numerator.
    u : array-like
        Coefficient array of denominator U.
    w : array-like
        Already-computed coefficients of W (orders 0..k-1).
    k : int
        Target order.

    Returns
    -------
    scalar

    Raises
    ------
    ZeroDivisionError
        If u[0] == 0.
    """
    # Caso p / U. Los ceros se construyen desde u para conservar el tipo
    # numérico, tanto en float como en mpfr.
    if u[0] == 0.0:
        raise ZeroDivisionError("inv_mc: denominator leading coefficient u[0] is zero")
    # Si V = exp(U), entonces V' = U' * V; esta recurrencia evita derivación
    # simbólica y solo usa coeficientes previos.
    if k == 0:
        ww = u[0] * 0.0 + 1.0   # preserves mpfr type if needed
    else:
        ww = u[k] * 0.0          # zero with correct type
        for j in range(k):
            ww -= u[k - j] * w[j]
    ww /= u[0]
    return ww * p


def exp_mc(u, v, k):
    """
    k-th Taylor coefficient of V = exp(U).

    Parameters
    ----------
    u : array-like
        Coefficient array of U.
    v : array-like
        Already-computed coefficients of exp(U) (orders 0..k-1).
    k : int
        Target order.

    Returns
    -------
    scalar
    """
    # Potencia con exponente constante: el orden 0 fija la escala inicial y
    # los demás órdenes se obtienen por recurrencia.
    if k == 0:
        return _exp(u[0])
    w = k * v[0] * u[k]
    for j in range(1, k):
        w += (k - j) * v[j] * u[k - j]
    return w / k


def pow_mc_c(u, c, w, k):
    """
    k-th Taylor coefficient of W = U^c  (c is a real constant exponent).

    Parameters
    ----------
    u : array-like
        Coefficient array of U.
    c : float
        Exponent.
    w : array-like
        Already-computed coefficients of W (orders 0..k-1).
    k : int
        Target order.

    Returns
    -------
    scalar

    Raises
    ------
    ZeroDivisionError
        If k==0 and u[0]==0.
    """
    # Para W = log(U), se usa U * W' = U' como identidad de recurrencia.
    if k == 0:
        if u[0] == 0.0:
            raise ZeroDivisionError("pow_mc_c: u[0] is zero, cannot raise to power c")
        return u[0] ** c
    if u[0] != 0.0:
        ww = c * k * w[0] * u[k]
        for j in range(1, k):
            ww += (c * (k - j) - j) * w[j] * u[k - j]
        return ww / (k * u[0])
    return u[0] * 0.0


def log_mc(u, w, k):
    """
    k-th Taylor coefficient of W = log(U).

    Parameters
    ----------
    u : array-like
        Coefficient array of U.
    w : array-like
        Already-computed coefficients of W (orders 0..k-1).
    k : int
        Target order.

    Returns
    -------
    scalar

    Raises
    ------
    ZeroDivisionError
        If k==0 and u[0]==0.
    """
    # sin(U) se calcula acoplado con los coeficientes ya disponibles de cos(U).
    if k == 0:
        if u[0] == 0.0:
            raise ZeroDivisionError("log_mc: u[0] is zero")
        return _log(u[0])
    ww = k * u[k]
    for j in range(1, k):
        ww -= (k - j) * u[j] * w[k - j]
    return ww / (k * u[0])


def sin_mc(u, cos_u, k):
    """
    k-th Taylor coefficient of sin(U).

    Parameters
    ----------
    u : array-like
        Coefficient array of U.
    cos_u : array-like
        Already-computed Taylor coefficients of cos(U) (orders 0..k-1).
    k : int
        Target order.

    Returns
    -------
    scalar
    """
    # cos(U) se calcula acoplado con los coeficientes ya disponibles de sin(U).
    if k == 0:
        return _sin(u[0])
    ww = u[1] * cos_u[k - 1]
    for j in range(2, k + 1):
        ww += j * u[j] * cos_u[k - j]
    return ww / k


def cos_mc(u, sin_u, k):
    """
    k-th Taylor coefficient of cos(U).

    Parameters
    ----------
    u : array-like
        Coefficient array of U.
    sin_u : array-like
        Already-computed Taylor coefficients of sin(U) (orders 0..k-1).
    k : int
        Target order.

    Returns
    -------
    scalar
    """
    if k == 0:
        return _cos(u[0])
    ww = -u[1] * sin_u[k - 1]
    for j in range(2, k + 1):
        ww -= j * u[j] * sin_u[k - j]
    return ww / k


# ---------------------------------------------------------------------------
# TIDES Integrator
# ---------------------------------------------------------------------------

class TidesSolver:
    """
    Variable-stepsize, variable-order Taylor integrator.

    Mirrors the behaviour of the TIDES C minimal integrator library.
    Supports both ``float`` (double precision) and ``gmpy2.mpfr``
    (arbitrary multiple precision), chosen by ``is_mpfr`` -- by default
    (``is_mpfr=None``) auto-detected from whether ``v_init``/``p_init``
    passed to ``solve`` already hold ``gmpy2.mpfr`` values.

    Parameters
    ----------
    mincseries_func : callable
        Function ``f(t, v, p, XVAR, ORDER, MO)`` that fills *XVAR* with
        Taylor coefficients up to *ORDER* for each state variable.
    nvar : int
        Number of state variables.
    npar : int
        Number of parameters.
    tolrel : float or mpfr
        Relative tolerance.
    tolabs : float or mpfr
        Absolute tolerance.
    maxord : int
        Maximum Taylor order allowed.
    minord : int
        Minimum Taylor order allowed.
    nordinc : int
        Extra order increment added on top of the tolerance-derived estimate.
    defect_error_control : bool
        Whether to apply defect-error-control step rejection (DEC).
    stepsize_controller : {"pytides", "original"}
        Which stepsize-estimate formula ``get_step`` uses, both sharing the
        same last-nonzero-coefficient search, [rmin, rmax]-relative
        clamping, and safety factor. ``"pytides"`` (default) is the
        variant actually shipped: each retained coefficient at order k is
        paired with the tolerance exponent of the *next*, untruncated
        order (see the comment in ``get_step`` below). ``"original"``
        instead reproduces Abad et al. (2012, Algorithm 924)'s own
        same-index pairing, where a term at order k contributes
        ``(TOL/|y^[k]|)^(1/k)``. Exists so the two controllers can be
        run back to back on the same problem for a direct comparison
        (steps taken, mean order, achieved error) -- see
        ``experiments/exp8_stepsize_controller.py``.
    is_mpfr : bool or None
        The sole place integration precision is chosen -- independent of
        how ``v_init``/``p_init`` were built (``exotides.orbital.
        HierarchicalSystem``/``exotides.nbody.pack_state`` always build
        plain ``float64``, regardless of this setting). ``None`` (default)
        auto-detects from whether ``v_init``/``p_init`` passed to ``solve``
        already hold ``gmpy2.mpfr`` values. ``True``/``False`` forces
        ``solve`` to convert them to that precision up front (via
        ``exotides.core.to_mpfr`` or plain ``float``, respectively) instead
        of trusting their existing dtype -- e.g. run a plain-``float64``-built
        system through an arbitrary-precision integration, without having
        to build the initial conditions in ``gmpy2.mpfr`` at all.
    """

    def __init__(
        self,
        mincseries_func,
        nvar,
        npar,
        tolrel=1e-16,
        tolabs=1e-16,
        maxord=26,
        minord=6,
        nordinc=5,
        defect_error_control=False,
        stepsize_controller="pytides",
        is_mpfr=None,
    ):
        if is_mpfr and not HAS_GMPY2:
            raise ValueError("is_mpfr=True requires gmpy2 to be installed")
        if stepsize_controller not in ("pytides", "original"):
            raise ValueError(
                f"stepsize_controller must be 'pytides' or 'original', got {stepsize_controller!r}"
            )
        self.mincseries = mincseries_func
        self.nvar = nvar
        self.npar = npar
        self.tolrel = tolrel
        self.tolabs = tolabs
        self.maxord = maxord
        self.minord = minord
        self.nordinc = nordinc
        self.defect_error_control = defect_error_control
        self.stepsize_controller = stepsize_controller
        self.is_mpfr = is_mpfr
        self.last_events = []
        self.last_run_stats = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, v_init, p_init, tini, tend, dt, events=None):
        """
        Integrate the system from *tini* to *tend*, recording state at
        multiples of *dt*.

        Parameters
        ----------
        v_init : array-like
            Initial state vector (length ``nvar``).
        p_init : array-like
            Parameter vector (length ``npar``).
        tini, tend, dt : float or mpfr
            Start time, end time, output step size.
        events : sequence of exotides.events.Event, optional
            Zero-crossing events to watch for during integration (see
            ``exotides/events.py``). Within each accepted adaptive step, every
            event's scalar function is evaluated at the step's two
            endpoints using the same dense Taylor-polynomial (``horner``)
            already computed for that step -- exactly the mechanism the
            original C TIDES library's ``dp_tides_find_zeros`` uses -- and
            any sign change is bisected (cheap: more ``horner``
            evaluations, no re-integration) to locate the crossing time
            precisely. Triggered crossings are recorded in
            ``self.last_events`` (cleared at the start of every call). If
            the earliest crossing within a step belongs to a ``terminal``
            event, integration stops exactly there instead of continuing
            to *tend*.

        Returns
        -------
        t_hist : ndarray
            1-D array of recorded times.
        v_hist : ndarray
            2-D array of shape ``(len(t_hist), nvar)`` with states.

        Also sets ``self.last_run_stats`` (cleared/overwritten at the start
        of every call, same lifetime as ``self.last_events``):
        ``{"accepted_steps", "rejected_steps", "mean_order",
        "coefficient_evals"}`` -- ``coefficient_evals`` sums ``order + 1``
        over every accepted step, a proxy for total recurrence/RHS cost
        (cheap steps at low order cost less than expensive ones at high
        order, unlike a plain step count). Used to compare
        ``stepsize_controller`` variants on the same problem; see
        ``experiments/exp8_stepsize_controller.py``.
        """
        self.last_events = []
        self.last_run_stats = {}
        if self.is_mpfr is None:
            # Detecta si debe trabajar en multiprecisión. Basta con que una
            # variable o parámetro sea mpfr para conservar ese tipo durante
            # todo el cálculo.
            is_mpfr = HAS_GMPY2 and (
                any(isinstance(x, gmpy2.mpfr) for x in v_init)
                or any(isinstance(x, gmpy2.mpfr) for x in p_init)
            )
        else:
            # Elección explícita (constructor is_mpfr=True/False): se
            # respeta sin importar el dtype real de v_init/p_init.
            is_mpfr = self.is_mpfr

        dtype = object if is_mpfr else np.float64

        if is_mpfr:
            v = np.array([to_mpfr(x) for x in v_init], dtype=dtype)
            p = np.array([to_mpfr(x) for x in p_init], dtype=dtype)
        else:
            v = np.array(v_init, dtype=dtype)
            p = np.array(p_init, dtype=dtype)

        t0 = tini
        tzero = tini
        deltat = dt

        if is_mpfr:
            t0       = gmpy2.mpfr(t0)
            tzero    = gmpy2.mpfr(tzero)
            deltat   = gmpy2.mpfr(deltat)
            tolabs_v = gmpy2.mpfr(self.tolabs)
            tolrel_v = gmpy2.mpfr(self.tolrel)
            extra    = gmpy2.mpfr("0.0")
            fac1     = gmpy2.mpfr("0.95")
            fac2     = gmpy2.mpfr("10.0")
            fac3     = gmpy2.mpfr("0.8")
            rmaxstep = gmpy2.mpfr("100.0")
            rminstep = gmpy2.mpfr("0.01")
        else:
            tolabs_v = float(self.tolabs)
            tolrel_v = float(self.tolrel)
            extra    = 0.0
            fac1     = 0.95
            fac2     = 10.0
            fac3     = 0.8
            rmaxstep = 100.0
            rminstep = 0.01

        # Dirección de integración: el mismo solver sirve para avanzar o
        # retroceder en el tiempo.
        if t0 < tend:
            tflag = 0
            if deltat < 0:
                deltat = -deltat
        else:
            tflag = 1
            if deltat > 0:
                deltat = -deltat

        # Historial de salida. Siempre se guarda el punto inicial.
        t_hist = [t0]
        v_hist = [v.copy()]

        # XVAR contiene los coeficientes de Taylor de tiempo y variables.
        # XVAR2 se reserva para el control opcional de defecto.
        if is_mpfr:
            XVAR = np.empty((self.maxord + 1, self.nvar + 1), dtype=object)
            XVAR.fill(gmpy2.mpfr("0.0"))
            XVAR2 = np.empty((2, self.nvar + 1), dtype=object)
            XVAR2.fill(gmpy2.mpfr("0.0"))
        else:
            XVAR  = np.zeros((self.maxord + 1, self.nvar + 1), dtype=np.float64)
            XVAR2 = np.zeros((2, self.nvar + 1), dtype=np.float64)

        ipos          = 1
        accepted_steps = 0
        rejected_steps = 0
        order_sum = 0
        coefficient_evals = 0

        ynb     = gmpy2.mpfr("0.0") if is_mpfr else 0.0
        stepant = gmpy2.mpfr("0.0") if is_mpfr else 0.0
        nitermax = 5

        # ---- inner helpers ------------------------------------------------

        def norm_inf_vec(vec):
            return np.max(np.abs(vec))

        def norm_inf_mat(ord_idx):
            return np.max(np.abs(XVAR[ord_idx, 1:]))

        def get_tolerances(ynb_val):
            # Ajusta tolerancia y orden de Taylor según el tamaño de la
            # solución. Replica la heurística del integrador TIDES en C.
            yna  = norm_inf_vec(v)
            tol  = tolabs_v + max(yna, ynb_val) * tolrel_v
            miny = min(yna, ynb_val)
            if miny > 0.0:
                tolo = min(tolabs_v / miny, tolrel_v)
            else:
                tolo = min(tolabs_v, tolrel_v)
            order = min(
                self.maxord,
                math.floor(-math.log10(float(tolo)) / 2) + self.nordinc,
            )
            order = max(self.minord, order)
            return tol, tolo, order, yna

        def get_step(order, tol, stepant_val):
            # Estima el paso usando los últimos coeficientes no nulos: si los
            # términos altos son pequeños, se puede avanzar más sin perder
            # precisión.
            ord_val = order + 1
            ynu = gmpy2.mpfr("0.0") if is_mpfr else 0.0
            while ord_val > 0:
                ord_val -= 1
                ynu = norm_inf_mat(ord_val)
                if ynu != 0.0:
                    break
            if ord_val == 0:
                raise ValueError("TIDES step calculation error: all Taylor coefficients are zero")
            orda  = ord_val - 1
            ordp  = ord_val + 1
            dord  = 1.0 / ord_val
            dorda = 1.0 / orda if orda > 0 else 0.0
            dordp = 1.0 / ordp

            ynp = norm_inf_mat(orda) if orda >= 0 else (gmpy2.mpfr("0.0") if is_mpfr else 0.0)
            if self.stepsize_controller == "original":
                # Abad et al. (2012, Algorithm 924)'s own ĥ_i: each retained
                # coefficient at order k is paired with *its own* exponent
                # 1/k on both TOL and ‖y^[k]‖ -- h_k = (TOL/‖y^[k]‖)^(1/k).
                if ynp == 0.0:
                    step_val = (tol ** dord) * ((1.0 / ynu) ** dord)
                else:
                    tp       = (tol ** dorda) * ((1.0 / ynp) ** dorda)
                    tu       = (tol ** dord)  * ((1.0 / ynu) ** dord)
                    step_val = min(tp, tu)
            else:
                # tp/tu pair each retained coefficient (orda=q-1, ord_val=q)
                # with the TOL exponent of the *next* term (ord_val, ordp)
                # -- i.e. each coefficient is treated as an estimator of the
                # term that is actually being truncated, not of its own
                # order. This differs from the same-index pairing above
                # ("original") -- a deliberately documented variant (not an
                # unrecorded accidental divergence), with no algebraic
                # equivalence claimed with that formula.
                if ynp == 0.0:
                    step_val = (tol ** dordp) * ((1.0 / ynu) ** dord)
                else:
                    tp       = (tol ** dord)  * ((1.0 / ynp) ** dorda)
                    tu       = (tol ** dordp) * ((1.0 / ynu) ** dord)
                    step_val = min(tp, tu)

            if stepant_val != 0.0:
                rstep = step_val / stepant_val
                if rstep > rmaxstep:
                    step_val = rmaxstep * stepant_val
                elif rstep < rminstep:
                    step_val = rminstep * stepant_val

            step_val *= fac1
            if tflag == 1:
                step_val = -step_val
            return step_val

        def make_zero_vector():
            if is_mpfr:
                res = np.empty(self.nvar, dtype=object)
                res.fill(gmpy2.mpfr("0.0"))
                return res
            return np.zeros(self.nvar, dtype=np.float64)

        def horner(t_val, order):
            # Evalúa el polinomio de Taylor con Horner, evitando potencias
            # explícitas y reduciendo operaciones.
            res = make_zero_vector()
            for j in range(1, self.nvar + 1):
                temp = XVAR[order, j] * t_val
                for i in range(order - 1, 0, -1):
                    temp = t_val * (temp + XVAR[i, j])
                res[j - 1] = temp + XVAR[0, j]
            return res

        def hornerd(t_val, order):
            # Misma evaluación que horner(), pero aplicada a la derivada.
            res = make_zero_vector()
            for j in range(1, self.nvar + 1):
                temp = order * XVAR[order, j] * t_val
                for i in range(order - 1, 1, -1):
                    temp = t_val * (temp + i * XVAR[i, j])
                res[j - 1] = temp + XVAR[1, j]
            return res

        def steps_DEC(t_val, tol, order, step_val):
            # Control de defecto opcional: compara la derivada del polinomio
            # con la dinámica recalculada y reduce el paso si es necesario.
            nonlocal rejected_steps
            iter_count = 1
            norma = 1e99
            t_curr = step_val
            while norma > fac2 * tol and iter_count < nitermax:
                vh  = horner(t_curr, order)
                vdh = hornerd(t_curr, order)
                self.mincseries(t_val + t_curr, vh, p, XVAR2, 1, 1)
                norma = abs(XVAR2[1, 1] - vdh[0])
                for idx in range(2, self.nvar + 1):
                    norma = max(abs(XVAR2[1, idx] - vdh[idx - 1]), norma)
                if iter_count > 1:
                    t_curr = fac3 * t_curr
                    rejected_steps += 1
                iter_count += 1
            return t_curr

        def find_event_crossings(t_start, v_start, nstep, order):
            # Busca cruces por cero de cada evento dentro del paso aceptado
            # actual, evaluando su función en los dos extremos (usando el
            # mismo polinomio de Horner ya calculado para este paso) y
            # afinando por bisección los que cambian de signo. Devuelve
            # (offset_terminal_o_None, lista_de_(offset, estado, evento)
            # ordenada por offset ascendente).
            v_end = horner(nstep, order)
            found = []
            for evt in events:
                f0 = evt.function(t_start, v_start)
                f1 = evt.function(t_start + nstep, v_end)
                if f0 == 0.0:
                    continue
                rising = f0 < 0.0 and f1 > 0.0
                falling = f0 > 0.0 and f1 < 0.0
                if not (rising or falling):
                    continue
                if evt.direction > 0 and not rising:
                    continue
                if evt.direction < 0 and not falling:
                    continue

                lo, hi = 0.0, nstep
                flo = f0
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    fmid = evt.function(t_start + mid, horner(mid, order))
                    if (fmid < 0.0) == (flo < 0.0):
                        lo, flo = mid, fmid
                    else:
                        hi = mid
                    if abs(float(hi) - float(lo)) < 1e-13 * max(1.0, abs(float(nstep))):
                        break
                offset = 0.5 * (lo + hi)
                found.append((offset, horner(offset, order), evt))

            found.sort(key=lambda item: item[0] if nstep >= 0 else -item[0])

            terminal_offset = None
            kept = []
            for offset, state, evt in found:
                kept.append((offset, state, evt))
                if evt.terminal:
                    terminal_offset = offset
                    break
            return terminal_offset, kept

        # ---- main integration loop ----------------------------------------

        while ((t0 < tend) and (tflag == 0)) or ((t0 > tend) and (tflag == 1)):
            # Cada vuelta construye una serie alrededor de t0, selecciona un
            # paso adaptativo y registra los puntos de salida que caen dentro.
            tol, tolo, order, yna = get_tolerances(ynb)

            self.mincseries(t0, v, p, XVAR, order, self.maxord)

            step = get_step(order, tol, stepant)

            if self.defect_error_control:
                step = steps_DEC(t0, tol, order, step)

            temp  = t0
            nstep = step + extra
            t0    = temp + nstep
            extra = (temp - t0) + nstep

            # Truncate to tend if overshooting
            if ((t0 > tend) and (tflag == 0)) or ((t0 < tend) and (tflag == 1)):
                nstep = tend - temp
                t0    = tend

            accepted_steps += 1
            order_sum += order
            coefficient_evals += order + 1

            terminal_offset = None
            if events:
                terminal_offset, event_hits = find_event_crossings(temp, v, nstep, order)
                for offset, state, evt in event_hits:
                    self.last_events.append({
                        "time": temp + offset, "state": state.copy(),
                        "name": evt.name, "terminal": evt.terminal,
                    })

            # Si un evento terminal cae dentro de este paso, el paso efectivo
            # (para registrar puntos de salida y avanzar el estado) se recorta
            # justo hasta ese instante.
            step_limit = nstep if terminal_offset is None else terminal_offset

            # Registra puntos de salida dentro del paso adaptativo actual,
            # aunque el paso interno no coincida con dt.
            ti  = tzero + ipos * deltat
            tit = ti - temp
            while ((tit <= step_limit) and (tflag == 0)) or ((tit >= step_limit) and (tflag == 1)):
                vh = horner(tit, order)
                t_hist.append(ti)
                v_hist.append(vh)
                ipos += 1
                ti  = tzero + ipos * deltat
                tit = ti - temp

            if terminal_offset is not None:
                t0 = temp + terminal_offset
                v  = horner(terminal_offset, order)
                if len(t_hist) == 0 or t_hist[-1] != t0:
                    t_hist.append(t0)
                    v_hist.append(v.copy())
                ynb     = yna
                stepant = abs(terminal_offset)
                stopped_by_event = True
                break

            # Avanza el estado real hasta el final del paso aceptado.
            v       = horner(nstep, order)
            ynb     = yna
            stepant = abs(nstep)
        else:
            stopped_by_event = False

        # Garantiza que el punto final aparece en el historial aunque no caiga
        # exactamente sobre la rejilla de salida por redondeo (salvo que la
        # integración se haya detenido antes de tend por un evento terminal).
        if not stopped_by_event and (len(t_hist) == 0 or abs(float(t_hist[-1]) - float(tend)) > 1e-12):
            t_hist.append(t0)
            v_hist.append(v.copy())

        self.last_run_stats = {
            "accepted_steps": accepted_steps,
            "rejected_steps": rejected_steps,
            "mean_order": (order_sum / accepted_steps) if accepted_steps else 0.0,
            "coefficient_evals": coefficient_evals,
        }

        return np.array(t_hist, dtype=dtype), np.array(v_hist, dtype=dtype)
