"""
tests/test_orbital_elements.py

Regression test for exotides.orbital.cartesian_to_keplerian's handling of the
degenerate coplanar case (i = 0), where the ascending node is undefined.

Before this fix, cartesian_to_keplerian silently returned aop = 0.0 whenever
n == 0 (i = 0 or pi), even for e > 0 -- discarding the orbit's actual in-plane
periapsis orientation instead of folding it into aop under the lan = 0
convention. This broke any attempt to recover the argument of periapsis (or
its time evolution, e.g. to study apsidal precession) for a coplanar system,
even though building a system *from* a given (i=0, lan=0, aop=theta) via
keplerian_to_cartesian has always used aop correctly (it reduces to a single
2D rotation by aop when i=lan=0, so there is no construction-side ambiguity --
only the inverse, Cartesian-to-elements direction is degenerate).

This test builds Cartesian states at several planar, eccentric (aop, ta)
combinations directly via keplerian_to_cartesian, converts them back with
cartesian_to_keplerian, and asserts the recovered elements match the
originals -- exercising both the i=0/h_z>0 (prograde) and i=pi/h_z<0
(retrograde) branches of the fix.
"""

import numpy as np

from exotides.orbital import cartesian_to_keplerian, keplerian_to_cartesian


def _round_trip(a, e, i, lan, aop, ta, mu):
    pos, vel = keplerian_to_cartesian(a, e, i, lan, aop, ta, mu)
    return cartesian_to_keplerian(pos, vel, mu)


def test_coplanar_prograde_aop_recovered():
    mu = 1.0
    a, e = 1.3, 0.4
    for aop in [0.0, 0.7, 2.1, -1.5, 3.0]:
        for ta in [0.0, 1.0, -2.2]:
            elements = _round_trip(a, e, i=0.0, lan=0.0, aop=aop, ta=ta, mu=mu)
            assert abs(elements["a"] - a) < 1e-10
            assert abs(elements["e"] - e) < 1e-10
            assert elements["i"] < 1e-10
            # lan is conventionally 0 in the degenerate case; aop must absorb
            # the full periapsis angle, not silently collapse to 0.
            assert abs(elements["lan"]) < 1e-10
            assert abs(np.remainder(elements["aop"] - aop, 2 * np.pi)) < 1e-10 or \
                abs(np.remainder(elements["aop"] - aop, 2 * np.pi) - 2 * np.pi) < 1e-10
            assert abs(np.remainder(elements["ta"] - ta, 2 * np.pi)) < 1e-10 or \
                abs(np.remainder(elements["ta"] - ta, 2 * np.pi) - 2 * np.pi) < 1e-10


def test_coplanar_retrograde_aop_recovered():
    mu = 1.0
    a, e = 1.3, 0.4
    for aop in [0.0, 0.7, 2.1, -1.5]:
        elements = _round_trip(a, e, i=np.pi, lan=0.0, aop=aop, ta=0.5, mu=mu)
        assert abs(elements["a"] - a) < 1e-10
        assert abs(elements["e"] - e) < 1e-10
        assert abs(np.remainder(elements["aop"] - aop, 2 * np.pi)) < 1e-9 or \
            abs(np.remainder(elements["aop"] - aop, 2 * np.pi) - 2 * np.pi) < 1e-9


def test_coplanar_circular_orbit_aop_is_zero_by_convention():
    # e = 0 is the other, genuinely-unrecoverable degeneracy (no periapsis
    # direction exists at all); aop should stay the documented default 0.0,
    # not raise or return something spurious.
    mu = 1.0
    elements = _round_trip(a=1.0, e=0.0, i=0.0, lan=0.0, aop=0.0, ta=0.9, mu=mu)
    assert elements["aop"] == 0.0
