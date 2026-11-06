"""
tests/test_conservation_integrals.py

Conservation of the classical integrals (energy, angular momentum, linear
momentum, barycentre motion) in a hierarchical system with more than three
bodies: a 2+2 quadruple with comparable masses on inclined, eccentric
orbits, integrated for 200 periods of its inner binaries at tol = 1e-14.
Short version of the conservation study in the pyTIDES paper, which runs the
same check over 10^4 inner periods for several systems and tolerances.
"""

import math

import numpy as np

from exotides.core import TidesSolver
from exotides.nbody import compute_energy, nbody_mincseries, unpack_state
from exotides.orbital import HierarchicalSystem


def _el(a, e, i=0.0, lan=0.0, aop=0.0, ta=0.0):
    return {"a": a, "e": e, "i": i, "lan": lan, "aop": aop, "ta": ta}


def test_quadruple_2plus2_conserves_classical_integrals():
    system = HierarchicalSystem(G=1.0)
    system.add_body("A", mass=1.0)
    system.add_body("B", mass=0.9, parent_name="A", elements=_el(1.0, 0.1))
    system.add_body("C", mass=0.8, parent_name="A", elements=_el(20.0, 0.3, i=0.6, lan=0.2, aop=0.4, ta=1.0))
    system.add_body("D", mass=0.7, parent_name="C", elements=_el(1.2, 0.2, i=0.3, lan=0.5, aop=0.1, ta=2.0))
    v_init, p_init, nodes = system.generate()
    masses = np.array([node.mass for node in nodes])

    period = 2.0 * math.pi * math.sqrt(1.0 / 1.9)
    solver = TidesSolver(
        mincseries_func=nbody_mincseries, nvar=len(v_init), npar=len(p_init),
        tolrel=1e-14, tolabs=1e-14, maxord=32, minord=8, is_mpfr=False,
    )
    t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=200 * period, dt=period)
    t_hist = np.asarray(t_hist, dtype=float)

    energy = compute_energy(states, masses, 1.0)
    positions, velocities = unpack_state(states)
    m = masses[None, :, None]
    L = (m * np.cross(positions, velocities)).sum(axis=1)
    P = (m * velocities).sum(axis=1)
    R = (m * positions).sum(axis=1)
    p_scale = np.sum(masses * np.linalg.norm(velocities[0], axis=1))
    r_scale = np.sum(masses * np.linalg.norm(positions[0], axis=1))

    assert np.max(np.abs(energy - energy[0])) / abs(energy[0]) < 1e-12
    assert np.max(np.linalg.norm(L - L[0], axis=1)) / np.linalg.norm(L[0]) < 1e-12
    assert np.max(np.linalg.norm(P - P[0], axis=1)) / p_scale < 1e-13
    assert np.max(np.linalg.norm(R - R[0] - np.outer(t_hist, P[0]), axis=1)) / r_scale < 1e-11
