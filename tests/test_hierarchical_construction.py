"""
tests/test_hierarchical_construction.py

Regression test for HierarchicalSystem's construction of Cartesian initial
conditions from nested Keplerian elements.

Before this fix, each body was placed at its *own* position on its orbit,
with mu = G * (M_interior + m_body). That is only right when everything
bound to the body (its moons, a companion star) has negligible mass: in a
2+2 quadruple with equal masses, C was put on the outer orbit and D was then
added around C, giving the C-D barycenter half of the inner orbital velocity
-- the requested circular outer orbit (a = 100) came out hyperbolic
(a = -5.75, e = 18.5). The construction now builds a binary tree of nested
subsystems and places each subsystem's barycenter on its orbit, with
mu = G * (M_inner + M_outer).

This test recovers every orbit's elements from the generated Cartesian state,
via cartesian_to_keplerian applied to subsystem barycenters, for hierarchies
where both sides of an orbit carry comparable mass, and checks that every
catalog template comes out barycentric.
"""

import numpy as np
import pytest

from exotides.hierarchy import HierarchicalSystemTemplates
from exotides.orbital import HierarchicalSystem, cartesian_to_keplerian


def _barycenter(nodes, names):
    selected = [node for node in nodes if node.name in names]
    mass = sum(node.mass for node in selected)
    pos = sum(node.mass * node.pos for node in selected) / mass
    vel = sum(node.mass * node.vel for node in selected) / mass
    return mass, pos, vel


def _orbit_elements(nodes, inner_names, outer_names, G=1.0):
    m_in, r_in, v_in = _barycenter(nodes, inner_names)
    m_out, r_out, v_out = _barycenter(nodes, outer_names)
    return cartesian_to_keplerian(r_out - r_in, v_out - v_in, G * (m_in + m_out))


def _assert_shape(recovered, requested):
    for key in ("a", "e", "i"):
        assert abs(recovered[key] - requested[key]) < 1e-10 * max(1.0, requested[key]), key


def _elements(a, e, i, lan=0.3, aop=0.7, ta=1.1):
    return {"a": a, "e": e, "i": i, "lan": lan, "aop": aop, "ta": ta}


def test_quadruple_2plus2_recovers_all_three_orbits():
    inner_ab = _elements(1.0, 0.1, 0.2)
    inner_cd = _elements(1.5, 0.3, 0.6)
    outer = _elements(100.0, 0.4, 0.9)

    system = HierarchicalSystem(G=1.0)
    system.add_body("A", 1.0)
    system.add_body("B", 0.8, "A", inner_ab)
    system.add_body("C", 0.9, "A", outer)
    system.add_body("D", 0.7, "C", inner_cd)
    _, _, nodes = system.generate()

    _assert_shape(_orbit_elements(nodes, {"A"}, {"B"}), inner_ab)
    _assert_shape(_orbit_elements(nodes, {"C"}, {"D"}), inner_cd)
    _assert_shape(_orbit_elements(nodes, {"A", "B"}, {"C", "D"}), outer)


def test_triple_chained_under_inner_component_orbits_inner_barycenter():
    # Star C is attached to Star B but orbits wider than A-B: it must orbit
    # the A-B barycenter, and must not drag B's own orbit with it.
    inner = _elements(10.0, 0.3, 0.1)
    outer = _elements(60.0, 0.2, 0.5)

    system = HierarchicalSystem(G=1.0)
    system.add_body("A", 1.1)
    system.add_body("B", 0.9, "A", inner)
    system.add_body("C", 1.0, "B", outer)
    _, _, nodes = system.generate()

    _assert_shape(_orbit_elements(nodes, {"A"}, {"B"}), inner)
    _assert_shape(_orbit_elements(nodes, {"A", "B"}, {"C"}), outer)


def test_massive_moon_orbit_describes_planet_moon_barycenter():
    planet = _elements(5.0, 0.05, 0.1)
    moon = _elements(0.02, 0.01, 0.3)

    system = HierarchicalSystem(G=1.0)
    system.add_body("Star", 1.0)
    system.add_body("Planet", 1e-3, "Star", planet)
    system.add_body("Moon", 5e-4, "Planet", moon)
    _, _, nodes = system.generate()

    _assert_shape(_orbit_elements(nodes, {"Planet"}, {"Moon"}), moon)
    _assert_shape(_orbit_elements(nodes, {"Star"}, {"Planet", "Moon"}), planet)


@pytest.mark.parametrize("key", HierarchicalSystemTemplates.keys())
def test_every_template_is_barycentric(key):
    template = HierarchicalSystemTemplates.get(key)
    masses = [1.0 if body_type == "star" else 1e-4 for body_type in template.body_types]
    elements = HierarchicalSystemTemplates.default_elements(key, masses)
    v_init, p_init, _ = HierarchicalSystemTemplates.initial_conditions(key, masses, elements)

    state = np.asarray(v_init, dtype=float).reshape(-1, 6)
    m = np.asarray(p_init[1:], dtype=float)[:, None]
    scale_r = np.max(np.linalg.norm(state[:, :3], axis=1)) or 1.0
    scale_v = np.max(np.linalg.norm(state[:, 3:], axis=1)) or 1.0
    assert np.linalg.norm((m * state[:, :3]).sum(axis=0)) < 1e-13 * scale_r * m.sum()
    assert np.linalg.norm((m * state[:, 3:]).sum(axis=0)) < 1e-13 * scale_v * m.sum()
