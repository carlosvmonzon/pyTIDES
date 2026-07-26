"""
Python/tests/helpers/precision.py

Standard/multiple-precision helpers shared across the std/mpfr parametrized
tests (test_kepler.py, test_lorenz.py).

``exotides`` (Python/src/exotides/) is importable because the package is
installed in editable mode (``pip install -e .`` from Python/, see
README.md) -- no ``sys.path`` manipulation needed for it. This package
(Python/tests/helpers/) is test-only support code, never imported by
``exotides`` itself, so it isn't installed; each ``test_*.py`` script is run
directly (``python test_foo.py``), which puts its own directory
(Python/tests/) on ``sys.path`` automatically, making ``from helpers import
...`` resolve without any extra setup either.
"""

from pathlib import Path

import numpy as np

from exotides.core import HAS_GMPY2, gmpy2


PYTHON_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = PYTHON_DIR.parent


def assert_close(actual, expected, tol, label):
    # Comprobación común: calcula el error máximo absoluto, igual que hacen
    # los drivers C al comparar el estado final con una referencia.
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    error = float(np.max(np.abs(actual - expected)))
    assert error < tol, f"{label}: error={error:.17e}, tol={tol:.17e}"
    return error


def parse_precision_mode(argv, script_name):
    args = list(argv)[1:]
    if len(args) == 0:
        return "std"
    mode = args[0].lower()
    if mode not in {"std", "mpfr"}:
        raise SystemExit(f"usage: python {script_name} [std|mpfr]")
    return mode


def mpfr_available_or_skip(label):
    if HAS_GMPY2:
        return True
    print(f"{label} mpfr: SKIP (gmpy2 is not installed)")
    return False


def configure_precision(mode, precision=128):
    if mode == "mpfr":
        if not mpfr_available_or_skip("precision mode"):
            return False
        gmpy2.get_context().precision = precision
    return True


def to_precision(value, mode="std"):
    if mode == "mpfr":
        return gmpy2.mpfr(str(value))
    return float(value)


def vector_to_precision(values, mode):
    return [to_precision(value, mode) for value in values]


def solver_settings(mode, std_tol=1e-15, mpfr_tol="1e-30", std_maxord=26, mpfr_maxord=40):
    if mode == "mpfr":
        return {
            "tolrel": gmpy2.mpfr(mpfr_tol),
            "tolabs": gmpy2.mpfr(mpfr_tol),
            "maxord": mpfr_maxord,
            "minord": 8,
        }
    return {
        "tolrel": std_tol,
        "tolabs": std_tol,
        "maxord": std_maxord,
        "minord": 6,
    }


def kepler_mincseries(t, v, p, XVAR, ORDER, MO):
    from exotides.core import HAS_GMPY2, gmpy2, mul_mc, pow_mc_c

    # Generador de series equivalente al ejemplo C std_kepler/minc_kepler:
    # x' = vx, y' = vy, vx' = -mu*x/r^3, vy' = -mu*y/r^3.
    x, y, vx, vy = range(1, 5)
    is_mpfr = HAS_GMPY2 and any(isinstance(value, gmpy2.mpfr) for value in v)
    dtype = object if is_mpfr else np.float64
    zero = gmpy2.mpfr("0.0") if is_mpfr else 0.0
    one = gmpy2.mpfr("1.0") if is_mpfr else 1.0
    r2 = np.empty(MO + 1, dtype=dtype)
    r3inv = np.empty(MO + 1, dtype=dtype)
    r2.fill(zero)
    r3inv.fill(zero)

    XVAR[: ORDER + 1, :] = zero
    XVAR[0, 0] = t
    XVAR[1, 0] = one
    XVAR[0, x] = v[0]
    XVAR[0, y] = v[1]
    XVAR[0, vx] = v[2]
    XVAR[0, vy] = v[3]

    for i in range(ORDER):
        # En cada orden se actualiza r^-3 y se escribe el siguiente
        # coeficiente de Taylor de posición y velocidad.
        r2[i] = mul_mc(XVAR[:, x], XVAR[:, x], i) + mul_mc(XVAR[:, y], XVAR[:, y], i)
        r3inv[i] = pow_mc_c(r2, -1.5, r3inv, i)

        inext = i + 1
        XVAR[inext, x] = XVAR[i, vx] / inext
        XVAR[inext, y] = XVAR[i, vy] / inext
        XVAR[inext, vx] = -p[0] * mul_mc(XVAR[:, x], r3inv, i) / inext
        XVAR[inext, vy] = -p[0] * mul_mc(XVAR[:, y], r3inv, i) / inext


def lorenz_mincseries(t, v, p, XVAR, ORDER, MO):
    from exotides.core import HAS_GMPY2, gmpy2, mul_mc

    # Generador de series equivalente al ejemplo C std_lorenz/minc_lorenz:
    # sistema caótico de Lorenz con parámetros sigma, rho y beta.
    x, y, z = range(1, 4)
    sigma, rho, beta = p
    is_mpfr = HAS_GMPY2 and any(isinstance(value, gmpy2.mpfr) for value in v)
    zero = gmpy2.mpfr("0.0") if is_mpfr else 0.0
    one = gmpy2.mpfr("1.0") if is_mpfr else 1.0

    XVAR[: ORDER + 1, :] = zero
    XVAR[0, 0] = t
    XVAR[1, 0] = one
    XVAR[0, x] = v[0]
    XVAR[0, y] = v[1]
    XVAR[0, z] = v[2]

    for i in range(ORDER):
        # Los productos xy y xz se calculan como productos de series, no como
        # productos de valores escalares del estado.
        xy = mul_mc(XVAR[:, x], XVAR[:, y], i)
        xz = mul_mc(XVAR[:, x], XVAR[:, z], i)

        inext = i + 1
        XVAR[inext, x] = sigma * (XVAR[i, y] - XVAR[i, x]) / inext
        XVAR[inext, y] = (rho * XVAR[i, x] - XVAR[i, y] - xz) / inext
        XVAR[inext, z] = (xy - beta * XVAR[i, z]) / inext
