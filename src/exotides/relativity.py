"""
exotides/relativity.py

Post-Newtonian (1PN) pairwise relativistic correction for the TIDES N-body
package.

Simplified "dominant mass" 1PN acceleration (Anderson et al. 1975; the same
approximation used by, e.g., REBOUNDx's basic ``gr`` module): for each pair
of bodies, the lighter body's acceleration gets an extra correction treating
the heavier body as an (approximately fixed) point-mass source of parameter
GM = G*M:

    a_PN = (GM / c^2) * [ (4*GM/r - v^2) * r_vec / r^3 + 4*(v.r_vec)/r^3 * v ]

with ``r_vec``/``v`` the light body's position/velocity *relative to the
source*, and ``r = |r_vec|``. This is not a full cross-coupled
Einstein-Infeld-Hoffmann treatment (no genuine three-body PN cross terms) --
it is the standard reduced two-body-per-pair approximation used throughout
the exoplanet-dynamics literature for hierarchical systems, appropriate when
each pair's mass ratio is not close to 1 (a real hierarchical star/planet/
moon system, not a comparable-mass cluster).

The speed of light ``c`` is appended to the parameter vector after the usual
N-body parameters, as one extra trailing value:

    p = [G, m0, ..., mN-1, c]
"""

import math

import numpy as np


PN_PARAM_COUNT = 1


def append_pn_params(p_init, c):
    """Return a parameter vector extended with the speed of light ``c``."""
    p_arr = np.asarray(p_init)
    dtype = object if p_arr.dtype == object else np.float64
    return np.array(list(p_init) + [c], dtype=dtype)


def has_pn_params(p, n_bodies):
    """Whether *p* contains the trailing speed-of-light parameter."""
    return len(p) >= 1 + n_bodies + PN_PARAM_COUNT


def pn_speed_of_light(p, n_bodies):
    """Read back the speed-of-light parameter appended by ``append_pn_params``."""
    return p[1 + n_bodies]


def gr_precession_rate_per_orbit(G, M, a, e, c):
    """
    Textbook (leading-order) general-relativistic apsidal precession rate,
    in radians per orbit, for a test mass on an orbit of semi-major axis
    ``a`` and eccentricity ``e`` around a dominant mass ``M``:

        delta_omega_per_orbit = 6*pi*G*M / (c^2 * a * (1 - e^2))

    This is the analytic reference the pairwise 1PN correction above
    (``nbody_pn_mincseries``) is expected to reproduce numerically -- see
    ``tests/test_relativity.py``.
    """
    return 6.0 * math.pi * G * M / (c ** 2 * a * (1.0 - e ** 2))
