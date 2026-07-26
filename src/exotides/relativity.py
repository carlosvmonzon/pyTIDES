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
import warnings

import numpy as np


PN_PARAM_COUNT = 1

# Heuristic heavier-to-lighter mass-ratio thresholds used by
# pn_pair_quality/warn_pn_quality below -- not a rigorously derived error
# bound, just a practical rule of thumb for how far a pair is from the
# extreme-mass-ratio limit the "dominant mass" approximation assumes.
PN_QUALITY_RATIO_GOOD = 10.0
PN_QUALITY_RATIO_FAIR = 3.0


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


def pn_pair_quality(m_a, m_b):
    """
    Heuristic quality label -- ``"good"``, ``"fair"``, or ``"poor"`` -- for
    the pairwise "dominant mass" 1PN approximation (module docstring above)
    applied to one pair of masses.

    The approximation replaces a pair's *mutual* two-body 1PN dynamics with
    a test body orbiting a fixed point-mass source, which is only exact in
    the limit of an extreme mass ratio (Anderson et al. 1975). The heavier-
    to-lighter mass ratio is used here as a proxy for how far a pair is from
    that limit:

        ratio >= 10        -> "good"  (e.g. star-planet, planet-moon)
        3 <= ratio < 10     -> "fair"
        ratio < 3           -> "poor"  (comparable masses, e.g. an
                                         equal-mass binary star)
    """
    lo, hi = sorted((abs(m_a), abs(m_b)))
    if lo == 0.0:
        return "good"
    ratio = hi / lo
    if ratio >= PN_QUALITY_RATIO_GOOD:
        return "good"
    if ratio >= PN_QUALITY_RATIO_FAIR:
        return "fair"
    return "poor"


def warn_pn_quality(masses, names=None, stacklevel=2):
    """
    Warn about the expected quality of the pairwise "dominant mass" 1PN
    correction for a set of masses -- one check per *interacting* pair, not
    just parent-child edges in a hierarchy template, since
    ``nbody_pn_mincseries`` applies the correction between every pair of
    bodies.

    Silent when every pair is ``"good"`` (see ``pn_pair_quality``).
    Otherwise raises one ``UserWarning`` naming the worst-quality pair(s)
    found. ``names`` labels bodies by name instead of positional index when
    given (e.g. a hierarchy template's body names).
    """
    rank = {"good": 0, "fair": 1, "poor": 2}
    worst = "good"
    worst_pairs = []
    n = len(masses)
    for j in range(n):
        for k in range(j + 1, n):
            quality = pn_pair_quality(masses[j], masses[k])
            if rank[quality] > rank[worst]:
                worst = quality
                worst_pairs = []
            if quality == worst and quality != "good":
                label_j = names[j] if names is not None else f"body {j}"
                label_k = names[k] if names is not None else f"body {k}"
                worst_pairs.append(f"{label_j}-{label_k}")

    if worst == "good":
        return

    pairs_str = ", ".join(worst_pairs)
    warnings.warn(
        f"1PN 'dominant mass' approximation quality is '{worst}' for this system "
        f"(pair(s): {pairs_str}) -- it reduces each pair's mutual two-body 1PN "
        "dynamics to a test body orbiting a fixed point-mass source, which "
        "degrades as that pair's mass ratio approaches 1; see "
        "exotides/relativity.py and exotides.relativity.pn_pair_quality.",
        stacklevel=stacklevel,
    )
