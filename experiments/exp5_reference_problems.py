"""
experiments/exp5_reference_problems.py

Reproduces the "Dynamical reference problems" experiment in docs/paper.tex
(Section 8.5): one nontrivial hierarchical N-body reference problem
(Kozai-Lidov eccentricity/inclination oscillation in an inclined
hierarchical triple, exercising the high-level hierarchy interface) and one
non-hierarchical problem integrated directly through the generic Newtonian
engine (the Pythagorean three-body problem, bypassing HierarchicalSystem
entirely -- there is no meaningful hierarchy to assign).
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from _common import savefig
from exotides.core import TidesSolver
from exotides.nbody import compute_energy, nbody_mincseries, pack_state, unpack_state, vector_norm
from exotides.orbital import HierarchicalSystem, cartesian_to_keplerian

G_AU_MSUN_YR = 4.0 * math.pi ** 2
M_JUP = 9.543e-4
R_SUN = 0.00465047
R_JUP = 0.00046732


def build_kozai_system():
    system = HierarchicalSystem(G=G_AU_MSUN_YR)
    system.add_body("StarA", mass=1.0)
    system.add_body(
        "Planet", mass=M_JUP, parent_name="StarA",
        elements={"a": 1.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
    )
    system.add_body(
        "StarB", mass=0.8, parent_name="StarA",
        elements={
            "a": 10.0, "e": 0.01, "i": math.radians(85.0),
            "lan": 0.0, "aop": math.radians(90.0), "ta": 0.0,
        },
    )
    return system.generate()


def run_kozai():
    v_init, p_init, nodes = build_kozai_system()
    tend = 400.0  # yr -- well inside the ~452 yr collision found separately, plenty for several cycles

    solver = TidesSolver(
        mincseries_func=nbody_mincseries, nvar=len(v_init), npar=len(p_init),
        tolrel=1e-13, tolabs=1e-13, maxord=32, minord=8, is_mpfr=False,
    )
    t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=tend, dt=tend / 2000.0)
    positions, velocities = unpack_state(states)
    mu_inner = p_init[0] * (p_init[1] + p_init[2])

    ecc = np.empty(len(t_hist))
    inc = np.empty(len(t_hist))
    for idx in range(len(t_hist)):
        rel_pos = positions[idx, 1] - positions[idx, 0]
        rel_vel = velocities[idx, 1] - velocities[idx, 0]
        elements = cartesian_to_keplerian(rel_pos, rel_vel, mu_inner)
        ecc[idx] = elements["e"]
        inc[idx] = math.degrees(elements["i"])

    print("Kozai-Lidov reference problem:")
    print(f"  eccentricity range over {tend:g} yr: {ecc.min():.4f} -> {ecc.max():.4f}")
    print(f"  inclination range: {inc.min():.2f} -> {inc.max():.2f} deg")
    return t_hist, ecc, inc


def build_pythagorean_system():
    positions = np.array([[1.0, 3.0, 0.0], [-2.0, -1.0, 0.0], [1.0, -1.0, 0.0]])
    velocities = np.zeros((3, 3))
    masses = np.array([3.0, 4.0, 5.0])
    v_init = pack_state(positions, velocities)
    p_init = np.concatenate(([1.0], masses))
    return v_init, p_init, masses


def run_pythagorean():
    v_init, p_init, masses = build_pythagorean_system()
    pairs = [(0, 1), (0, 2), (1, 2)]

    solver = TidesSolver(
        mincseries_func=nbody_mincseries, nvar=len(v_init), npar=len(p_init),
        tolrel=1e-12, tolabs=1e-12, maxord=32, minord=8, is_mpfr=False,
    )
    t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=50.0, dt=0.02)
    positions, _ = unpack_state(states)

    energy = compute_energy(states, masses, 1.0)
    energy_drift = float(np.max(np.abs(energy - energy[0])))
    min_sep = min(
        float(np.min(vector_norm(positions[:, i] - positions[:, j], axis=1)))
        for i, j in pairs
    )
    print("Pythagorean three-body reference problem (non-hierarchical):")
    print(f"  energy drift over 50 time units: {energy_drift:.3e}")
    print(f"  min pairwise separation reached: {min_sep:.4f} (initial separations were 3, 4, 5)")
    return t_hist, positions, energy_drift, min_sep


def main():
    print("=== Exp 5: Dynamical reference problems ===\n")
    t_kozai, ecc, inc = run_kozai()
    print()
    t_pyth, pos_pyth, energy_drift, min_sep = run_pythagorean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(t_kozai, ecc, color="tab:red", lw=1.0)
    axes[0].set_xlabel("time [yr]")
    axes[0].set_ylabel("inner-orbit eccentricity")
    axes[0].set_title("(a) Kozai-Lidov: eccentricity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_kozai, inc, color="tab:blue", lw=1.0)
    axes[1].set_xlabel("time [yr]")
    axes[1].set_ylabel("inner-orbit inclination [deg]")
    axes[1].set_title("(b) Kozai-Lidov: inclination")
    axes[1].grid(True, alpha=0.3)

    colors = ["tab:orange", "tab:green", "tab:purple"]
    labels = ["m=3", "m=4", "m=5"]
    for i in range(3):
        axes[2].plot(pos_pyth[:, i, 0], pos_pyth[:, i, 1], lw=0.8, color=colors[i], label=labels[i])
        axes[2].scatter(pos_pyth[0, i, 0], pos_pyth[0, i, 1], color=colors[i], marker="o", s=40)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].axis("equal")
    axes[2].set_title("(c) Pythagorean 3-body (non-hierarchical engine)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "reference_problems.png")

    print("\n--- Summary ---")
    print(f"  Kozai-Lidov: eccentricity {ecc.min():.3f}->{ecc.max():.3f}, "
          f"inclination {inc.min():.1f}->{inc.max():.1f} deg over {t_kozai[-1]:.0f} yr "
          f"(hierarchy-template interface, exercised via HierarchicalSystem directly).")
    print(f"  Pythagorean: energy drift {energy_drift:.3e}, min separation {min_sep:.4f} "
          f"(built directly in Cartesian coordinates, bypassing HierarchicalSystem entirely).")


if __name__ == "__main__":
    main()
