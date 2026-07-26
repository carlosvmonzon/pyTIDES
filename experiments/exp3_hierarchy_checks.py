"""
experiments/exp3_hierarchy_checks.py

Reproduces the "Hierarchy construction checks" experiment in docs/paper.tex
(Section 8.3): for four representative configurations (star-planet-moon, an
S-type planet in a binary, a P-type circumbinary planet, and a nested
stellar triple), verify (1) the barycentric constraints
sum(m_i r_i) = 0, sum(m_i v_i) = 0, and (2) that the states produced by
HierarchicalSystem match an *independently* coded reference construction
(plain sequential two-body embedding via keplerian_to_cartesian, written
from scratch in this script -- NOT calling HierarchicalSystem._resolve()/
_interior_reference() at all).

The interior-reference set used by the independent reconstruction for each
body is determined by hand from the documented Jacobi-ordering rule
(exotides/orbital.py's HierarchicalSystem._interior_reference: a body's
reference is its parent plus any *already-resolved* sibling/ancestor whose
own semi-major axis is smaller than the body's), not by calling the
package's own resolution code -- this is what makes the comparison a
genuine independent check rather than a self-consistency tautology.
"""

import numpy as np

from exotides.core import as_float64
from exotides.orbital import HierarchicalSystem, keplerian_to_cartesian

G = 1.0


def embed(mu, elements, ref_pos, ref_vel):
    r_rel, v_rel = keplerian_to_cartesian(
        elements["a"], elements["e"], elements["i"],
        elements["lan"], elements["aop"], elements["ta"], mu,
    )
    return ref_pos + r_rel, ref_vel + v_rel


def barycenter(positions, velocities, masses, indices):
    m = np.array([masses[i] for i in indices])
    P = np.array([positions[i] for i in indices])
    V = np.array([velocities[i] for i in indices])
    M = m.sum()
    return (m[:, None] * P).sum(axis=0) / M, (m[:, None] * V).sum(axis=0) / M


def shift_to_barycentric(positions, velocities, masses):
    m = np.asarray(masses)
    R = (m[:, None] * positions).sum(axis=0) / m.sum()
    V = (m[:, None] * velocities).sum(axis=0) / m.sum()
    return positions - R, velocities - V


def independent_star_planet_moon(masses, elements):
    # Order: Star (root) -> Planet (ref = Star only) -> Moon (ref = Planet
    # only -- Planet's own orbit, a=1.0, is *not* narrower than Moon's,
    # a=0.01, so Star is not folded into Moon's reference; see the module
    # docstring). Crucially, mu for embedding a body uses its *subtree*
    # mass (HierarchicalSystem._resolve's `child.sys_mass`), not its bare
    # mass: Planet has a descendant (Moon), so mu_planet must include
    # m_moon too, exactly as if the (Planet, Moon) pair were one point
    # mass from the Star's perspective. This was the one subtlety this
    # independent script initially got wrong (see git history) -- caught
    # precisely because the comparison below did not match to machine
    # precision until it was fixed, which is the entire point of an
    # independent check.
    m_star, m_planet, m_moon = masses
    pos = [np.zeros(3), None, None]
    vel = [np.zeros(3), None, None]

    mu_planet = G * (m_star + (m_planet + m_moon))
    pos[1], vel[1] = embed(mu_planet, elements[1], pos[0], vel[0])

    mu_moon = G * (m_planet + m_moon)
    pos[2], vel[2] = embed(mu_moon, elements[2], pos[1], vel[1])

    positions = np.array(pos)
    velocities = np.array(vel)
    return shift_to_barycentric(positions, velocities, masses)


def independent_binary_s_planet_primary(masses, elements):
    # Bodies: 0=Star A (root), 1=Star B (parent A, a=2.0), 2=Planet (parent
    # A, a=0.3, S-type). Resolution order is ascending semi-major axis, so
    # Planet (a=0.3) is resolved *before* Star B (a=2.0):
    #   Planet: mu = G(mA + mPlanet), ref = Star A only (Star B not yet
    #     resolved, and even if it were, its a=2.0 is not < Planet's a=0.3).
    #   Star B: mu = G(mA + mPlanet + mB), ref = barycenter(Star A, Planet)
    #     -- Planet's a=0.3 *is* < Star B's a=2.0, so it is folded in.
    m_A, m_B, m_planet = masses
    pos = [np.zeros(3), None, None]
    vel = [np.zeros(3), None, None]

    mu_planet = G * (m_A + m_planet)
    pos[2], vel[2] = embed(mu_planet, elements[2], pos[0], vel[0])

    ref_pos, ref_vel = barycenter(pos, vel, masses, [0, 2])
    mu_B = G * (m_A + m_planet + m_B)
    pos[1], vel[1] = embed(mu_B, elements[1], ref_pos, ref_vel)

    positions = np.array(pos)
    velocities = np.array(vel)
    return shift_to_barycentric(positions, velocities, masses)


def independent_binary_p_planet(masses, elements):
    # Bodies: 0=Star A (root), 1=Star B (parent A, a=2.0), 2=Planet (parent
    # A in the tree, a=25.0, P-type). Star B (a=2.0) resolved first:
    #   Star B: mu = G(mA + mB), ref = Star A only.
    #   Planet: mu = G(mA + mB + mPlanet), ref = barycenter(Star A, Star B)
    #     -- Star B's a=2.0 *is* < Planet's a=25.0, so it is folded in,
    #     reproducing the P-type (circumbinary) reference despite the tree
    #     literally recording Planet's parent as Star A alone.
    m_A, m_B, m_planet = masses
    pos = [np.zeros(3), None, None]
    vel = [np.zeros(3), None, None]

    mu_B = G * (m_A + m_B)
    pos[1], vel[1] = embed(mu_B, elements[1], pos[0], vel[0])

    ref_pos, ref_vel = barycenter(pos, vel, masses, [0, 1])
    mu_planet = G * (m_A + m_B + m_planet)
    pos[2], vel[2] = embed(mu_planet, elements[2], ref_pos, ref_vel)

    positions = np.array(pos)
    velocities = np.array(vel)
    return shift_to_barycentric(positions, velocities, masses)


def independent_triple_star_chain(masses, elements):
    # Bodies: 0=Star A (root), 1=Star B (parent A, a=2.0), 2=Star C (parent
    # *B*, a=15.0 -- the actual "chain" topology: A<-B<-C, matching
    # src/exotides/hierarchy.py's "triple_star_chain" template, NOT a flat
    # tree with both B and C as direct children of A):
    #   Star B: mu = G(mA + B.sys_mass) = G(mA + mB + mC) -- B's own
    #     subtree includes C (its child), so embedding B treats "B+C" as
    #     one point of combined mass orbiting A, exactly as a moon's mass
    #     is folded into its planet's mu in the star-planet-moon case
    #     above. ref = Star A only.
    #   Star C: mu = G(mA + mB + mC), ref = barycenter(Star A, Star B) --
    #     the textbook nested-triple Jacobi case: the outer star feels the
    #     combined mass/barycenter of the inner pair, not just its nominal
    #     tree parent (which is literally Star B alone in this topology).
    m_A, m_B, m_C = masses
    pos = [np.zeros(3), None, None]
    vel = [np.zeros(3), None, None]

    mu_B = G * (m_A + (m_B + m_C))
    pos[1], vel[1] = embed(mu_B, elements[1], pos[0], vel[0])

    ref_pos, ref_vel = barycenter(pos, vel, masses, [0, 1])
    mu_C = G * (m_A + m_B + m_C)
    pos[2], vel[2] = embed(mu_C, elements[2], ref_pos, ref_vel)

    positions = np.array(pos)
    velocities = np.array(vel)
    return shift_to_barycentric(positions, velocities, masses)


CASES = {
    "star_planet_moon": dict(
        masses=[1.0, 1.0e-4, 1.0e-6],
        elements={
            0: None,
            1: {"a": 1.0, "e": 0.1, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 0.01, "e": 0.05, "i": 0.1, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        tree=[("Star", None, None), ("Planet", "Star", 1), ("Moon", "Planet", 2)],
        independent=independent_star_planet_moon,
    ),
    "binary_s_planet_primary": dict(
        masses=[1.0, 1.0, 1.0e-4],
        elements={
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 0.3, "e": 0.02, "i": 0.1, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        tree=[("StarA", None, None), ("StarB", "StarA", 1), ("Planet", "StarA", 2)],
        independent=independent_binary_s_planet_primary,
    ),
    "binary_p_planet": dict(
        masses=[1.0, 1.0, 1.0e-4],
        elements={
            0: None,
            1: {"a": 2.0, "e": 0.0, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 25.0, "e": 0.0, "i": 0.2, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        tree=[("StarA", None, None), ("StarB", "StarA", 1), ("Planet", "StarA", 2)],
        independent=independent_binary_p_planet,
    ),
    "triple_star_chain": dict(
        masses=[1.0, 1.0, 1.0e-3],
        elements={
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 15.0, "e": 0.05, "i": 0.3, "lan": 0.2, "aop": 0.1, "ta": 0.3},
        },
        tree=[("StarA", None, None), ("StarB", "StarA", 1), ("StarC", "StarB", 2)],
        independent=independent_triple_star_chain,
    ),
}


def run_case(key, spec):
    masses = spec["masses"]
    elements = spec["elements"]

    system = HierarchicalSystem(G=G)
    name_by_idx = {}
    for idx, (name, parent, elem_idx) in enumerate(spec["tree"]):
        parent_name = None if parent is None else parent
        el = None if elem_idx is None else elements[elem_idx]
        system.add_body(name, mass=masses[idx], parent_name=parent_name, elements=el)
        name_by_idx[idx] = name
    v_init, p_init, nodes = system.generate()

    from exotides.nbody import unpack_state
    pos_pkg, vel_pkg = unpack_state(v_init[np.newaxis, :])
    pos_pkg = as_float64(pos_pkg)[0]
    vel_pkg = as_float64(vel_pkg)[0]

    pos_ref, vel_ref = spec["independent"](masses, elements)

    max_pos_diff = float(np.max(np.abs(pos_pkg - pos_ref)))
    max_vel_diff = float(np.max(np.abs(vel_pkg - vel_ref)))

    m = np.array(masses)
    bary_pos_residual = float(np.max(np.abs((m[:, None] * pos_pkg).sum(axis=0))))
    bary_vel_residual = float(np.max(np.abs((m[:, None] * vel_pkg).sum(axis=0))))

    return {
        "max_pos_diff": max_pos_diff,
        "max_vel_diff": max_vel_diff,
        "bary_pos_residual": bary_pos_residual,
        "bary_vel_residual": bary_vel_residual,
    }


def main():
    print("=== Exp 3: Hierarchy construction checks ===")
    print(f"{'case':>28}  {'max|dpos|':>12}  {'max|dvel|':>12}  {'|sum m r|':>12}  {'|sum m v|':>12}")
    for key, spec in CASES.items():
        r = run_case(key, spec)
        print(
            f"{key:>28}  {r['max_pos_diff']:>12.3e}  {r['max_vel_diff']:>12.3e}  "
            f"{r['bary_pos_residual']:>12.3e}  {r['bary_vel_residual']:>12.3e}"
        )


if __name__ == "__main__":
    main()
