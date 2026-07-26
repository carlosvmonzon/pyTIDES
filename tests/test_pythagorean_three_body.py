"""
tests/test_pythagorean_three_body.py

Non-hierarchical N-body verification: the "Pythagorean three-body problem"
(Burrau 1913) -- three masses (3, 4, 5) released from rest at the vertices
of a 3-4-5 right triangle, G=1. Unlike every other N-body test in this
suite, the bodies here aren't built through
``exotides.orbital.HierarchicalSystem`` -- no body dominates the other two,
so there's no natural parent/child pair to hang orbital elements off of.
The initial state is built directly in Cartesian coordinates via
``exotides.nbody.pack_state`` instead.

This configuration is a textbook example of gravitational instability: with
no hierarchy to keep any pair bound, the three bodies fall together and
undergo a strong close encounter well before t=20, scattering to
separations far outside their initial 3-4-5 triangle. ``main()`` integrates
it and asserts both that energy stays conserved (the one thing that
*should* stay well-behaved) and that some pairwise separation collapses far
below anything in the initial triangle -- checked early, before the
encounter has had time to amplify tiny numerical differences into a
different chaotic outcome, so the assertion stays robust across platforms
even though the system itself is chaotic.
"""

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np

from helpers import PYTHON_DIR
from exotides.core import TidesSolver, vector_norm
from exotides.nbody import nbody_mincseries, unpack_state, pack_state, compute_energy
from exotides.plotting import animate_orbit, plot_orbit


PAIRS = [(0, 1), (0, 2), (1, 2)]


def build_pythagorean_system():
    """
    Burrau's classic Pythagorean three-body problem: masses 3, 4, 5 at rest
    at the vertices of a 3-4-5 right triangle (G=1) -- already centered on
    the center of mass by construction, since the triangle was chosen that
    way.

    ``nodes`` are plain ``SimpleNamespace`` stand-ins exposing just the
    ``.name``/``.mass`` that ``exotides.plotting`` needs -- there's no
    hierarchy here, so ``HierarchicalSystem.Node`` objects (with unused
    ``parent_name``/``children`` fields) would be misleading.
    """
    positions = np.array([
        [1.0, 3.0, 0.0],
        [-2.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
    ])
    velocities = np.zeros((3, 3))
    masses = np.array([3.0, 4.0, 5.0])
    G = 1.0

    v_init = pack_state(positions, velocities)
    p_init = np.concatenate(([G], masses))
    nodes = [SimpleNamespace(name=f"m={mass:g}", mass=mass) for mass in masses]
    return v_init, p_init, nodes


def test_pythagorean_three_body(output_dir=None):
    """Pytest-discoverable entry point. ``output_dir=None`` (the pytest
    default) skips plotting/animation -- only the energy-conservation and
    close-encounter assertions run; passing a real directory (as ``main()``
    does for standalone script execution) also saves the plot/animation."""
    v_init, p_init, nodes = build_pythagorean_system()
    masses = np.array([node.mass for node in nodes])
    G = float(p_init[0])

    initial_positions, _ = unpack_state(v_init)
    initial_separations = {
        (i, j): float(vector_norm(initial_positions[i] - initial_positions[j]))
        for i, j in PAIRS
    }

    solver = TidesSolver(
        mincseries_func=nbody_mincseries,
        nvar=len(v_init),
        npar=len(p_init),
        tolrel=1e-12,
        tolabs=1e-12,
        maxord=32,
        minord=8,
        is_mpfr=False
    )

    t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=50.0, dt=0.005)
    positions, _ = unpack_state(states)

    energy = compute_energy(states, masses, G)
    energy_drift = float(np.max(np.abs(energy - energy[0])))

    min_separation = min(
        float(np.min(vector_norm(positions[:, i] - positions[:, j], axis=1)))
        for i, j in PAIRS
    )
    smallest_initial_separation = min(initial_separations.values())

    assert energy_drift < 1e-6, f"energy not conserved: drift={energy_drift:.3e}"
    assert min_separation < smallest_initial_separation / 5, (
        f"expected a close encounter well inside the initial triangle "
        f"(smallest initial separation {smallest_initial_separation:.3f}), got "
        f"min separation {min_separation:.3f} -- system stayed too stable for "
        f"a Pythagorean-problem sanity check"
    )

    print("Pythagorean three-body setup (non-hierarchical):")
    print(f"  bodies: {[node.name for node in nodes]}")
    print(f"  initial separations: {initial_separations}")
    print(f"  time span: {float(t_hist[0]):.1f} -> {float(t_hist[-1]):.1f}")
    print(f"  energy drift: {energy_drift:.3e}")
    print(f"  min pairwise separation reached: {min_separation:.6f}")

    if output_dir is not None:
        output_path = plot_orbit(
            "pythagorean", "Pythagorean three-body problem (non-hierarchical)",
            states, nodes, output_dir=output_dir,
        )
        animation_path = animate_orbit(
            "pythagorean", "Pythagorean three-body problem (non-hierarchical)",
            states, nodes, output_dir=output_dir,
        )
        print(f"  plot saved to: {output_path}")
        print(f"  animation saved to: {animation_path}")


def main():
    test_pythagorean_three_body(PYTHON_DIR)


if __name__ == "__main__":
    main()
