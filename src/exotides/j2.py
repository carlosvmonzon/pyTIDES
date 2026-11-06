"""
exotides/j2.py

**WORK IN PROGRESS -- DO NOT USE YET.** The model/recurrence in this module
does not yet behave as intended and has not been validated against a known
reference (e.g. the analytic J2 nodal/apsidal precession rate, or an
independent reconstruction). That's why the only public entry point that
uses it, ``exotides.nbody.nbody_j2_mincseries``, deliberately raises
``NotImplementedError`` -- do not remove that guard until this has been
verified. This file's helpers (``append_j2_params``, ``has_j2_params``,
``read_j2_params``) are plain parameter-vector packing/unpacking, run no
physics, and are left operational.

Continuous J2 (Earth-oblateness-on-Moon) perturbation for the TIDES N-body
package, added as a genuine term inside the Taylor-series recurrence (not a
discrete "kick" applied between separate integration segments: an earlier
discrete-kick implementation, in a separate project, showed a real numerical
stability threshold around dt~1-2 days for century-long integrations).

Physical model: the standard J2 zonal-harmonic acceleration on a body due to
an oblate primary, primary's spin axis fixed along +Z of the integration
frame (no precession/nutation of the primary's own orientation -- the same
approximation the discrete-kick version already made):

    a_J2 = -(3/2) * J2 * GM_primary * Re^2 / r^5 *
           [ x*(1 - 5*(z/r)^2), y*(1 - 5*(z/r)^2), z*(3 - 5*(z/r)^2) ]

applied to the *lighter* body's relative position (x,y,z) = r_light - r_heavy,
r = |r_vec|. An equal-and-opposite reaction acceleration is applied to the
heavy body (scaled by the light/heavy mass ratio) so the pair's total
momentum stays conserved -- there is no compensating torque on the primary's
*orientation* since that orientation is held fixed by assumption; this is a
translational-momentum bookkeeping choice only, matching the discrete-kick
version's convention.

Only one designated (heavy, light) pair receives this correction -- unlike
the 1PN correction (exotides/relativity.py), which is pairwise-generic, J2
is a property of one specific body's shape, not of every pair.

Parameter layout (appended after the usual N-body parameters):

    p = [G, m0, ..., mN-1, heavy_idx, light_idx, J2, Re]

``heavy_idx`` must be < ``light_idx`` (the pair-slot lookup in
``_nbody_mincseries_core`` always keys off the canonical (min, max) index
order, so this keeps the direction of the stored relative vector -- from the
heavy body to the light body -- unambiguous without extra sign bookkeeping).

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

J2_PARAM_COUNT = 4


def append_j2_params(p_init, heavy_idx, light_idx, j2_coefficient, radius):
    """Return a parameter vector extended with (heavy_idx, light_idx, J2, Re).

    Raises ``ValueError`` if ``heavy_idx >= light_idx`` -- see the module
    docstring for why the canonical (min, max) ordering is required.
    """
    if heavy_idx >= light_idx:
        raise ValueError(
            f"heavy_idx ({heavy_idx}) must be < light_idx ({light_idx}): the pair-slot "
            "lookup keys off the canonical (min, max) index order, so this keeps the "
            "stored relative vector's direction (heavy -> light) unambiguous"
        )
    p_arr = np.asarray(p_init)
    dtype = object if p_arr.dtype == object else np.float64
    return np.array(list(p_init) + [heavy_idx, light_idx, j2_coefficient, radius], dtype=dtype)


def has_j2_params(p, n_bodies):
    """Whether *p* contains the trailing (heavy_idx, light_idx, J2, Re) block."""
    return len(p) >= 1 + n_bodies + J2_PARAM_COUNT


def read_j2_params(p, n_bodies):
    """Read back (heavy_idx, light_idx, j2_coefficient, radius) appended by
    ``append_j2_params``."""
    base = 1 + n_bodies
    return int(p[base]), int(p[base + 1]), p[base + 2], p[base + 3]
