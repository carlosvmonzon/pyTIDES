"""
tests/test_lagrange_three_body.py

Non-hierarchical N-body verification: Lagrange's equilateral-triangle
homographic solution of the three-body problem (Lagrange 1772). Three
bodies at the vertices of an equilateral triangle, given the right
tangential velocity, keep that triangle's shape and size exactly constant
for all time -- the whole configuration just rotates rigidly about the
common center of mass. "Homographic" means exactly this: at every instant
the configuration is similar (same shape, only scaled/rotated) to the
initial one, as opposed to a generic three-body configuration whose shape
constantly changes.

Unlike ``test_pythagorean_three_body.py`` (same non-hierarchical
construction via ``exotides.nbody.pack_state``, no natural reference body),
this configuration is exactly the opposite of chaotic in the sense that
it's an *exact* solution for any masses at the vertices of an equilateral
triangle. It is not, however, a *stable* one for equal masses: the equal-mass
case fails the Routh criterion (stability requires
``27*(m1*m2+m2*m3+m3*m1) < (m1+m2+m3)**2``, which for ``m1=m2=m3`` reduces
to ``81 > 9``), so it is linearly unstable -- any perturbation, however
small, grows exponentially rather than staying bounded. In this test that
shows up empirically as roughly an 80x growth in the triangle-side deviation
per orbital period (e-folding time ~0.23 periods), so the "exact" homographic
shape only survives a handful of periods in practice before visibly
deforming, no matter how precisely it's integrated.

That last point is why ``mode="mpfr"`` matters here, and why it goes further
than just passing ``is_mpfr=True`` to ``TidesSolver``: with float64
initial conditions (ordinary ``math.cos``/``math.sin``/``math.sqrt``,
~1e-16 relative error baked in before the solver ever sees them), the
perturbation that the instability amplifies is already there at machine
epsilon regardless of the integration precision, so an ``is_mpfr=True`` run
starting from those same float64 numbers diverges at essentially the same
period as an ordinary float64 run -- extra integration precision only
avoids adding *more* error, it can't remove the part already present in the
starting condition. ``build_lagrange_system(mode="mpfr")`` instead builds
every number (including the trigonometry) with ``gmpy2`` at the active
``gmpy2.get_context().precision``, so the seed perturbation itself shrinks
to roughly that precision's rounding level -- which, since the growth is
exponential, delays the visible instability by a number of periods
proportional to the *extra digits* of precision (``ln(precision ratio) /
ln(80.7)`` periods), not by a proportionally huge amount. Verified
empirically: going from float64 (~1e-16) to 200-bit gmpy2 (~1e-60)
initial-condition rounding delays the deviation from float64-noise-level
(``<1e-15``, period 0-23) to a visible (~5%) deformation at period ~31,
vs. ~period 7-8 for the float64-seeded case -- a real, substantial delay,
just a logarithmic one rather than "mpfr fixes it."

``main()`` integrates 12 periods (``python test_lagrange_three_body.py
[std|mpfr]``, default ``std``) and asserts energy is conserved -- the
homographic/self-similar property itself is reported (``max triangle-side
deviation``) but not asserted on, since -- per the above -- 12 periods is
already past the point where either mode's triangle still looks
equilateral; it is left as a printed, honest data point rather than a pass
condition.
"""

import math

import matplotlib
matplotlib.use("Agg")
import numpy as np
from types import SimpleNamespace

from helpers import PYTHON_DIR, configure_precision, solver_settings
from exotides.core import TidesSolver
from exotides.nbody import nbody_mincseries, unpack_state, pack_state, compute_energy, vector_norm
from exotides.plotting import animate_orbit, plot_orbit


PAIRS = [(0, 1), (0, 2), (1, 2)]


def build_lagrange_system(mass=1.0, side=1.0, G=1.0, mode="std"):
    """
    Lagrange's equilateral-triangle homographic solution: three bodies of
    equal mass at the vertices of an equilateral triangle of side ``side``,
    each given the tangential velocity for rigid rotation at
    ``Omega = sqrt(G * (3*mass) / side**3)`` about the (stationary, by
    symmetry) center of mass -- the exact solution that keeps the triangle
    shape/size constant forever (for perfect arithmetic -- see the module
    docstring for why this configuration is nonetheless unstable in
    practice, and why ``mode`` matters).

    ``mode="mpfr"`` builds every quantity below (including the
    trigonometry) with ``gmpy2`` at ``gmpy2.get_context().precision``
    instead of Python's float64 ``math`` module, so the initial condition's
    own rounding error is at that precision rather than always stuck at
    ~1e-16. Requires ``configure_precision("mpfr")`` (or an equivalent
    ``gmpy2.get_context().precision`` assignment) to have been called
    first.
    """
    if mode == "mpfr":
        from exotides.core import gmpy2
        sqrt, cos, sin, pi = gmpy2.sqrt, gmpy2.cos, gmpy2.sin, gmpy2.const_pi()
        mass, side, G = gmpy2.mpfr(mass), gmpy2.mpfr(side), gmpy2.mpfr(G)
        zero, dtype = gmpy2.mpfr(0.0), object
    else:
        sqrt, cos, sin, pi = math.sqrt, math.cos, math.sin, math.pi
        zero, dtype = 0.0, np.float64

    masses = np.array([mass, mass, mass], dtype=dtype)
    circumradius = side / sqrt(3.0)
    omega = sqrt(G * masses.sum() / side**3)
    speed = omega * circumradius
    period = 2.0 * pi / omega

    angles = [zero, 2.0 * pi / 3.0, 4.0 * pi / 3.0]
    positions = np.array([
        [circumradius * cos(a), circumradius * sin(a), zero] for a in angles
    ], dtype=dtype)
    velocities = np.array([
        [-speed * sin(a), speed * cos(a), zero] for a in angles
    ], dtype=dtype)

    v_init = pack_state(positions, velocities)
    p_init = np.empty(4, dtype=dtype)
    p_init[0] = G
    p_init[1:] = masses
    nodes = [SimpleNamespace(name=f"Body {i + 1}", mass=mass) for i in range(3)]
    return v_init, p_init, nodes, period


def test_lagrange_homographic(output_dir=None):
    """Pytest-discoverable entry point. ``output_dir=None`` (the pytest
    default) skips plotting/animation -- only the energy-conservation
    assertion runs; passing a real directory (as ``main()`` does for
    standalone script execution) also saves the plot/animation. Skips
    entirely (no assertion) if ``gmpy2`` isn't installed, same as before
    this was made pytest-compatible -- this test always integrates in
    ``mode="mpfr"`` (see the module docstring for why)."""
    mode = "mpfr"
    if not configure_precision(mode, precision=200):
        return

    v_init, p_init, nodes, period = build_lagrange_system(mode=mode)
    masses = np.array([node.mass for node in nodes])
    G = float(p_init[0])
    side = 1.0

    settings = solver_settings(mode, std_tol=1e-13, mpfr_tol="1e-40", std_maxord=32, mpfr_maxord=40)
    solver = TidesSolver(
        mincseries_func=nbody_mincseries,
        nvar=len(v_init),
        npar=len(p_init),
        is_mpfr= True,
        **settings,
    )

    t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=12.0 * period, dt=period / 150.0)
    positions, _ = unpack_state(states)

    energy = compute_energy(states, masses, G)
    energy_drift = float(np.max(np.abs(energy - energy[0])))

    separations = {
        (i, j): vector_norm(positions[:, i] - positions[:, j], axis=1)
        for i, j in PAIRS
    }
    max_side_deviation = max(
        float(np.max(np.abs(sep - side))) for sep in separations.values()
    )

    assert energy_drift < 1e-6, f"energy not conserved: drift={energy_drift:.3e}"

    print(f"Lagrange equilateral three-body setup (non-hierarchical, mode={mode}):")
    print(f"  bodies: {[node.name for node in nodes]}")
    print(f"  period: {period:.6f}")
    print(f"  time span: {float(t_hist[0]):.1f} -> {float(t_hist[-1]):.1f}")
    print(f"  energy drift: {energy_drift:.3e}")
    print(f"  max triangle-side deviation: {max_side_deviation:.3e}")

    if output_dir is not None:
        output_path = plot_orbit(
            "lagrange", "Lagrange equilateral-triangle homographic solution (equal masses)",
            states, nodes, output_dir=output_dir,
        )
        animation_path = animate_orbit(
            "lagrange", "Lagrange equilateral-triangle homographic solution (equal masses)",
            states, nodes, output_dir=output_dir, trail_frames=12,
        )
        print(f"  plot saved to: {output_path}")
        print(f"  animation saved to: {animation_path}")


def main():
    test_lagrange_homographic(PYTHON_DIR)


if __name__ == "__main__":
    main()
