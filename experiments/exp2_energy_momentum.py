"""
experiments/exp2_energy_momentum.py

Reproduces the "Conservation of energy and angular momentum" experiment in
docs/paper.tex (Section 8.2). A hierarchical star-planet-moon triple is
integrated for many outer-planet orbits at several requested tolerances,
tracking

    eps_E(t) = |E(t) - E(0)| / |E(0)|
    eps_L(t) = ||L(t) - L(0)|| / ||L(0)||

to show the transition between the truncation-dominated regime (loose
tolerance: eps grows secularly with time) and the round-off-dominated
regime (tight tolerance: eps fluctuates near machine precision with no
secular trend).
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from _common import savefig
from exotides.core import TidesSolver
from exotides.nbody import compute_energy, nbody_mincseries, unpack_state
from exotides.orbital import HierarchicalSystem

G = 1.0
MASSES = [1.0, 1.0e-4, 1.0e-6]  # Star, Planet, Moon
A_PLANET = 1.0
N_ORBITS = 30
TOLERANCES = [1e-6, 1e-9, 1e-12, 1e-14]


def build_system():
    system = HierarchicalSystem(G=G)
    system.add_body("Star", mass=MASSES[0])
    system.add_body(
        "Planet", mass=MASSES[1], parent_name="Star",
        elements={"a": A_PLANET, "e": 0.2, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
    )
    system.add_body(
        "Moon", mass=MASSES[2], parent_name="Planet",
        elements={"a": 0.01, "e": 0.1, "i": 0.2, "lan": 0.0, "aop": 0.0, "ta": 0.0},
    )
    return system.generate()


def angular_momentum(states, masses):
    positions, velocities = unpack_state(states)
    L = np.zeros((states.shape[0], 3))
    for i, m in enumerate(masses):
        L += m * np.cross(positions[:, i, :], velocities[:, i, :])
    return L


def main():
    print("=== Exp 2: Energy / angular-momentum conservation ===")

    v_init, p_init, nodes = build_system()
    masses = np.array([node.mass for node in nodes])
    period_outer = 2.0 * math.pi * math.sqrt(A_PLANET ** 3 / (G * (MASSES[0] + MASSES[1])))
    tend = N_ORBITS * period_outer

    fig, (ax_e, ax_l) = plt.subplots(1, 2, figsize=(11, 4.5))
    summary = []

    for tol in TOLERANCES:
        solver = TidesSolver(
            mincseries_func=nbody_mincseries,
            nvar=len(v_init), npar=len(p_init),
            tolrel=tol, tolabs=tol, maxord=32, minord=8, is_mpfr=False,
        )
        t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=tend, dt=period_outer / 20.0)

        energy = compute_energy(states, masses, G)
        eps_E = np.abs(energy - energy[0]) / np.abs(energy[0])

        L = angular_momentum(states, masses)
        L0_norm = np.linalg.norm(L[0])
        eps_L = np.linalg.norm(L - L[0], axis=1) / L0_norm

        t_periods = np.asarray(t_hist) / period_outer
        ax_e.semilogy(t_periods, np.maximum(eps_E, 1e-20), label=f"tol={tol:.0e}")
        ax_l.semilogy(t_periods, np.maximum(eps_L, 1e-20), label=f"tol={tol:.0e}")

        max_eps_E = float(np.max(eps_E))
        max_eps_L = float(np.max(eps_L))
        summary.append((tol, max_eps_E, max_eps_L))
        print(f"  tol={tol:.0e}  max(eps_E)={max_eps_E:.3e}  max(eps_L)={max_eps_L:.3e}")

    ax_e.set_xlabel("time [outer orbital periods]")
    ax_e.set_ylabel(r"$\epsilon_E(t)$")
    ax_e.set_title("Energy conservation")
    ax_e.grid(True, which="both", alpha=0.3)
    ax_e.legend(fontsize=8)

    ax_l.set_xlabel("time [outer orbital periods]")
    ax_l.set_ylabel(r"$\epsilon_L(t)$")
    ax_l.set_title("Angular-momentum conservation")
    ax_l.grid(True, which="both", alpha=0.3)
    ax_l.legend(fontsize=8)

    fig.suptitle("Star-planet-moon triple: conservation vs. tolerance")
    fig.tight_layout()
    savefig(fig, "energy_momentum_conservation.png")

    print("\n--- Summary (max error over the full integration) ---")
    print(f"{'tol':>8}  {'max eps_E':>12}  {'max eps_L':>12}")
    for tol, eE, eL in summary:
        print(f"{tol:>8.0e}  {eE:>12.3e}  {eL:>12.3e}")


if __name__ == "__main__":
    main()
