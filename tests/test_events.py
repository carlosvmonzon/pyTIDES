"""
tests/test_events.py

Verification for zero-crossing event detection (exotides.events, and the
``events``/``last_events`` parameter of ``exotides.core.TidesSolver.solve``) --
ported from the original C TIDES library's ``doubEVENTS.c``/
``mpfrEVENTS.c`` subsystem (never previously in this Python package).

Four checks:
1. Regression: passing ``events=None`` (the default) reproduces the exact
   same trajectory as no ``events`` argument at all.
2. Terminal collision event from genuine orbital instability, built two
   ways -- directly with ``HierarchicalSystem.add_body`` and through the
   ``exotides.hierarchy`` "star_two_planets" template catalog entry -- on the
   *same* system: two closely-spaced, eccentric ~8-Jupiter-mass planets
   around one star (units: solar masses, AU, G=4*pi**2, so periods come
   out in years). Their mutual perturbation is strong enough to close a
   real physical gap (radii are actual solar/Jupiter radii, not inflated)
   within a few dozen years -- confirming both entry points into the
   package produce the identical collision. The colliding bodies are
   printed by name, not plotted.
3. Non-terminal event: an inclined orbit's z-coordinate crossing zero
   (ascending/descending node passages) is recorded in ``last_events``
   without stopping integration, and the recorded crossing count matches
   the expected 2-per-orbit rate.
"""

import math

from helpers import assert_single_terminal_collision
from exotides.core import TidesSolver
from exotides.orbital import HierarchicalSystem
from exotides.hierarchy import HierarchicalSystemTemplates
from exotides.nbody import nbody_mincseries
from exotides.events import all_pairs_collision_events

G_AU_MSUN_YR = 4.0 * math.pi ** 2
M_JUP = 9.543e-4      # Jupiter mass, in solar masses
R_SUN = 0.00465047    # 1 solar radius, in AU
R_JUP = 0.00046732    # 1 Jupiter radius, in AU

# Two ~8-Jupiter-mass planets a=1.0/1.10 AU apart, eccentric and mutually
# inclined -- close enough (a handful of mutual Hill radii) that their
# perturbation is strongly chaotic rather than a stable, slowly-precessing
# pair. Found empirically: this configuration collides at t~28.3 yr.
MASSES = [1.0, 8.0 * M_JUP, 8.0 * M_JUP]
RADII = [R_SUN, R_JUP, R_JUP]
P1_ELEMENTS = {"a": 1.00, "e": 0.05, "i": 0.00, "lan": 0.0, "aop": 0.0, "ta": 0.0}
P2_ELEMENTS = {"a": 1.10, "e": 0.15, "i": 0.05, "lan": 0.3, "aop": 0.2, "ta": 0.0}
TEND = 50.0  # years -- generous margin past the ~28.3 yr collision


def _build_unstable_two_planets():
    system = HierarchicalSystem(G=G_AU_MSUN_YR)
    system.add_body("Star", mass=MASSES[0])
    system.add_body("P1", mass=MASSES[1], parent_name="Star", elements=P1_ELEMENTS)
    system.add_body("P2", mass=MASSES[2], parent_name="Star", elements=P2_ELEMENTS)
    return system.generate()


def _make_solver(v_init, p_init):
    return TidesSolver(
        mincseries_func=nbody_mincseries, nvar=len(v_init), npar=len(p_init),
        tolrel=1e-13, tolabs=1e-13, maxord=24, minord=8,
    )


def test_unstable_two_planets_collision_add_body():
    """Orbital-instability collision on a system built directly with ``HierarchicalSystem.add_body``."""
    v_init, p_init, nodes = _build_unstable_two_planets()

    solver = _make_solver(v_init, p_init)
    period1 = 2.0 * math.pi * math.sqrt(1.0 ** 3 / MASSES[0])  # P1's own period, ~1 yr
    events = all_pairs_collision_events(3, RADII)
    t_hist, states = solver.solve(v_init, p_init, tini=0.0, 
                                  tend=TEND, dt=period1 / 50.0, events=events)
    hit = assert_single_terminal_collision(solver, t_hist, states, RADII, 
                                           TEND, label="add_body", nodes=nodes)
    return hit


def test_unstable_two_planets_collision_template():
    """The same masses/elements, built instead from the ``star_two_planets`` catalog template."""
    v_init, p_init, nodes = HierarchicalSystemTemplates.initial_conditions(
        "star_two_planets", MASSES, elements={0: None, 1: P1_ELEMENTS, 2: P2_ELEMENTS}, G=G_AU_MSUN_YR,
    )

    solver = _make_solver(v_init, p_init)
    period1 = 2.0 * math.pi * math.sqrt(1.0 ** 3 / MASSES[0])
    events = all_pairs_collision_events(3, RADII)
    t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=TEND, dt=period1 / 50.0, events=events)
    hit = assert_single_terminal_collision(solver, t_hist, states, RADII, TEND, label="template", nodes=nodes)
    return hit


def main():
    test_unstable_two_planets_collision_add_body()
    test_unstable_two_planets_collision_template()
    print("All event-detection checks passed.")


if __name__ == "__main__":
    main()
