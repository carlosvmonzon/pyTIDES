"""
tests/test_kozai_nbody.py

N-body verification of Kozai-Lidov oscillations: in a hierarchical triple
where the outer companion's orbit is highly inclined relative to the inner
planet's, secular perturbation should drive a periodic exchange between the
inner orbit's eccentricity and inclination (rising eccentricity as
inclination falls, and back), even though this test integrates the full
Newtonian equations of motion directly rather than the quadrupole-averaged
secular equations the effect is classically derived from.

Realistic units throughout (AU, solar masses, G = 4*pi**2 so periods come
out in years -- same convention as test_events.py), and a realistic
hot-Jupiter-mass planet (``M_JUP``) orbiting a solar-mass star (``R_SUN``
radius) with a stellar companion inclined 85 degrees to it. That high a
mutual inclination is what makes this the "eccentric" extreme of
Kozai-Lidov cycling: the inner orbit's eccentricity is driven from 0.05 to
above 0.995 -- close enough to a purely radial orbit that its pericenter
distance collapses from 0.95 AU to a few thousandths of an AU, well inside
R_SUN + R_JUP. A terminal ``exotides.events.collision_event`` on the
Star A/Planet pair stops the integration exactly at that contact -- i.e.
the Kozai-Lidov cycle itself, not the starting configuration, is what
produces the collision (physically, this is the same instability tidal
friction is invoked to arrest before impact in the standard hot-Jupiter
migration story; no tidal forces are modeled here, so gravity alone carries
it all the way in).

``main()`` integrates ``build_kozai_system()`` for up to ``TEND`` years and
asserts: (1) the inner planet's eccentricity and inclination (measured
relative to Star A -- ``planet_eccentricity_span``) each swing through a
wide enough range to count as genuine Kozai-Lidov cycling, not just noise,
and (2) exactly one terminal collision fires, well after t=0 (so it is
attributable to the cycle) and well before ``TEND``.
"""

import math

import matplotlib
matplotlib.use("Agg")
import numpy as np

from helpers import PYTHON_DIR, build_system, assert_single_terminal_collision
from exotides.core import TidesSolver
from exotides.events import collision_event
from exotides.nbody import nbody_mincseries, unpack_state
from exotides.orbital import cartesian_to_keplerian
from exotides.plotting import animate_orbit, plot_orbital_elements

G_AU_MSUN_YR = 4.0 * math.pi ** 2
M_JUP = 9.543e-4      # Jupiter mass, in solar masses
R_SUN = 0.00465047    # 1 solar radius, in AU
R_JUP = 0.00046732    # 1 Jupiter radius, in AU

TEND = 1000.0  # years -- generous margin past the ~452 yr collision


def build_kozai_system():
    """
    Hierarchical triple:

    - body 0: primary star, 1 Msun, radius R_SUN
    - body 1: hot-Jupiter-mass inner planet, M_JUP, radius R_JUP
    - body 2: distant stellar companion, 0.8 Msun

    The outer orbit starts highly inclined (85 degrees) with respect to the
    planetary orbit -- the "eccentric" extreme of the classical quadrupole
    Kozai-Lidov setup, chosen (see module docstring) so the eccentricity
    excursion is large enough to bring the planet into contact with Star A.
    """
    return build_system([
        {"name": "Star A", "mass": 1.0},
        {
            "name": "Planet", "mass": M_JUP, "parent_name": "Star A",
            "elements": {"a": 1.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        {
            "name": "Star B", "mass": 0.8, "parent_name": "Star A",
            "elements": {
                "a": 10.0, "e": 0.01, "i": math.radians(85.0),
                "lan": 0.0, "aop": math.radians(90.0), "ta": 0.0,
            },
        },
    ], G=G_AU_MSUN_YR)


def planet_eccentricity_span(t_hist, states, p_init):
    """Planet eccentricity/inclination range relative to Star A, for the
    Kozai-Lidov oscillation-amplitude assertions in main()."""
    positions, velocities = unpack_state(states)
    mu_inner = p_init[0] * (p_init[1] + p_init[2])

    eccentricities = np.empty(len(t_hist), dtype=np.float64)
    inclinations = np.empty(len(t_hist), dtype=np.float64)
    for idx in range(len(t_hist)):
        rel_pos = positions[idx, 1] - positions[idx, 0]
        rel_vel = velocities[idx, 1] - velocities[idx, 0]
        elements = cartesian_to_keplerian(rel_pos, rel_vel, mu_inner)
        eccentricities[idx] = elements["e"]
        inclinations[idx] = math.degrees(elements["i"])
    return eccentricities, inclinations


def test_kozai_lidov(output_dir=None):
    """Pytest-discoverable entry point. ``output_dir=None`` (the pytest
    default) skips the elements/animation plots -- and the extra
    fine-grained re-solve that only exists to feed the animation -- while
    still running the main integration and every assertion; passing a real
    directory (as ``main()`` does for standalone script execution) also
    saves the plots."""
    v_init, p_init, nodes = build_kozai_system()

    solver = TidesSolver(
        mincseries_func=nbody_mincseries,
        nvar=len(v_init),
        npar=len(p_init),
        tolrel=1e-13,
        tolabs=1e-13,
        maxord=32,
        minord=8,
        is_mpfr=False
    )

    events = [collision_event(0, 1, R_SUN, R_JUP)]
    t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=TEND, dt=TEND / 200.0, events=events)
    eccentricities, inclinations = planet_eccentricity_span(t_hist, states, p_init)

    e_span = float(np.max(eccentricities) - np.min(eccentricities))
    i_span = float(np.max(inclinations) - np.min(inclinations))
    assert e_span > 0.05, f"eccentricity oscillation too small: {e_span:.6e}"
    assert i_span > 1.0, f"inclination oscillation too small: {i_span:.6e} deg"

    # The collision must come from the Kozai-Lidov cycle pumping up the
    # eccentricity, not from the starting configuration (whose pericenter,
    # a*(1-e) = 0.95 AU, is nowhere near R_SUN + R_JUP = 0.0051 AU). Checked
    # here, right after the solve that set it, since ``solver.last_events``
    # reflects only the most recent ``solve()`` call and the fine-grained
    # re-solve below would otherwise clobber it.
    assert float(t_hist[-1]) > 50.0, "collision fired implausibly early for a cycle-driven contact"
    hit = assert_single_terminal_collision(
        solver, t_hist, states, [R_SUN, R_JUP], TEND, label="kozai", nodes=nodes,
    )

    print("Kozai-Lidov N-body setup:")
    print(f"  bodies: {[node.name for node in nodes]}")
    print(f"  time span: {float(t_hist[0]):.1f} -> {float(t_hist[-1]):.1f} yr")
    print(f"  eccentricity range: {np.min(eccentricities):.6f} -> {np.max(eccentricities):.6f}")
    print(f"  inclination range: {np.min(inclinations):.6f} -> {np.max(inclinations):.6f} deg")
    print(f"  collision at t={float(hit['time']):.1f} yr (R_SUN + R_JUP = {R_SUN + R_JUP:.6f} AU)")

    if output_dir is None:
        return

    elements_plot_path = plot_orbital_elements(
        "kozai", "Kozai-Lidov-like evolution in an inclined hierarchical triple",
        t_hist, states, p_init, body_idx=1, parent_idx=0, output_dir=output_dir, angles=False
    )

    # A separate, much finer-grained solve just for this plot: dt=1/60 yr
    # gives ~60 samples per inner orbit (period ~1 yr near a=1 AU) instead
    # of the ~1 every 5 orbits that dt=TEND/200 above would, so the
    # trajectory is actually traced instead of landing on effectively
    # random phases of it. Reusing the same ``events`` makes this solve
    # stop at the same physical collision -- TidesSolver's event
    # root-finding operates on the dense Taylor polynomial of each accepted
    # step regardless of ``dt`` (which only controls dense-output sampling,
    # see TidesSolver.solve), so this costs extra Horner evaluations, not
    # extra integration work or a different stopping point.
    t_hist_fine, states_fine = solver.solve(
        v_init, p_init, tini=0.0, tend=TEND, dt=1.0 / 60.0, events=events,
    )
    inner_orbit_animation_path = animate_orbit(
        "kozai_inner", "Kozai-Lidov inner orbit (Planet relative to Star A)",
        states_fine, nodes, output_dir=output_dir, ref_idx=0, body_indices=[1],
        fps=30, trail_frames=5,
    )

    print(f"  elements plot saved to: {elements_plot_path}")
    print(f"  inner orbit animation saved to: {inner_orbit_animation_path}")


def main():
    test_kozai_lidov(PYTHON_DIR)


if __name__ == "__main__":
    main()
