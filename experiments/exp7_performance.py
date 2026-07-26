"""
experiments/exp7_performance.py

Reproduces the "Performance and comparison with reference integrators"
experiment in docs/paper.tex (Section 9). Two parts:

1. Matched-accuracy benchmark: wall-clock time to integrate the same
   two-body Kepler problem (a=1, e=0.4) over exactly one orbital period,
   plotted against the state-closure error (same metric as
   exp1_kepler_convergence.py), for pyTIDES (TidesSolver, tolerance swept,
   both with its Numba-accelerated force evaluation and with a pure-Python
   force evaluation) and REBOUND (WHFast, timestep swept; IAS15 at its
   default settings -- this installed REBOUND version (5.1.1) does not
   expose IAS15's adaptive epsilon parameter at the Python level, so IAS15
   contributes one reference point rather than a full sweep). Timed with
   time.perf_counter around only the integration call itself (JIT warm-up
   for the Numba path is excluded -- see part 2's cold-vs-warm note).
   IMPORTANT CAVEAT, stated explicitly rather than left implicit: REBOUND
   is a fully compiled C library end to end, while pyTIDES here is a
   Python-orchestrated driver (order selection, Horner dense-output
   evaluation, step accept/reject -- none of that is JIT-compiled, since it
   is generic control flow, not a per-problem numerical kernel) with only
   its innermost force evaluation optionally JIT-compiled via Numba. The
   pure-Python force curve is plotted specifically to show how much of the
   gap is attributable to the (small, for N=2) force-evaluation cost that
   Numba addresses versus the driver-loop overhead that no amount of
   JIT-compiling the force law can remove. This benchmark should be read as
   comparing two packages as actually used, not as isolating the
   underlying numerical methods from their implementation language -- see
   the discussion following the matched-accuracy figure in the paper.

2. Numba vs. pure-Python scaling: wall-clock time for the Newtonian
   coefficient generator as a function of body count N (reproducing
   tests/test_fast_nbody.py's benchmark for the paper).

heyoka was attempted but could not be built in this environment (it
requires a C++ toolchain/CMake with no prebuilt wheel for this Python/OS
combination) -- omitted rather than fabricated.
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np

from _common import build_two_body_system, savefig
from exotides.core import TidesSolver
from exotides.nbody import nbody_mincseries, _nbody_mincseries_core

try:
    import rebound
    HAS_REBOUND = True
except ImportError:
    HAS_REBOUND = False

try:
    from exotides import _fast_nbody
    HAS_NUMBA = _fast_nbody.HAS_NUMBA
except ImportError:
    HAS_NUMBA = False


G = 1.0
MASS_STAR = 1.0
MASS_PLANET = 1.0e-6
MU = G * (MASS_STAR + MASS_PLANET)
A = 1.0
E = 0.4
PERIOD = 2.0 * math.pi * math.sqrt(A ** 3 / MU)


# ---------------------------------------------------------------------------
# Part 1: matched-accuracy comparison
# ---------------------------------------------------------------------------

def pytides_point(tol, force_func=nbody_mincseries):
    v_init, p_init, _ = build_two_body_system(mass_star=MASS_STAR, mass_planet=MASS_PLANET, a=A, e=E, G=G)
    solver = TidesSolver(
        mincseries_func=force_func, nvar=len(v_init), npar=len(p_init),
        tolrel=tol, tolabs=tol, maxord=32, minord=6,
    )
    t0 = time.perf_counter()
    _, states = solver.solve(v_init, p_init, tini=0.0, tend=PERIOD, dt=PERIOD / 4.0)
    wall = time.perf_counter() - t0
    y0 = np.asarray(states[0], dtype=np.float64)
    yP = np.asarray(states[-1], dtype=np.float64)
    err = float(np.linalg.norm(yP - y0) / np.linalg.norm(y0))
    return wall, err


def _pure_python_force(t, v, p, XVAR, ORDER, MO):
    return _nbody_mincseries_core(t, v, p, XVAR, ORDER, MO)


def rebound_whfast_point(dt_fraction):
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.dt = dt_fraction * PERIOD
    sim.add(m=MASS_STAR)
    sim.add(m=MASS_PLANET, a=A, e=E)
    sim.move_to_com()
    y0 = np.array([p.x for p in sim.particles] + [p.y for p in sim.particles] +
                  [p.vx for p in sim.particles] + [p.vy for p in sim.particles])
    t0 = time.perf_counter()
    sim.integrate(PERIOD)
    wall = time.perf_counter() - t0
    y1 = np.array([p.x for p in sim.particles] + [p.y for p in sim.particles] +
                  [p.vx for p in sim.particles] + [p.vy for p in sim.particles])
    err = float(np.linalg.norm(y1 - y0) / np.linalg.norm(y0))
    return wall, err


def rebound_ias15_point():
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.add(m=MASS_STAR)
    sim.add(m=MASS_PLANET, a=A, e=E)
    sim.move_to_com()
    y0 = np.array([p.x for p in sim.particles] + [p.y for p in sim.particles] +
                  [p.vx for p in sim.particles] + [p.vy for p in sim.particles])
    t0 = time.perf_counter()
    sim.integrate(PERIOD)
    wall = time.perf_counter() - t0
    y1 = np.array([p.x for p in sim.particles] + [p.y for p in sim.particles] +
                  [p.vx for p in sim.particles] + [p.vy for p in sim.particles])
    err = float(np.linalg.norm(y1 - y0) / np.linalg.norm(y0))
    return wall, err


def run_matched_accuracy():
    print("=== Part 1: matched-accuracy comparison (Kepler two-body, one period) ===\n")

    if HAS_NUMBA:
        cold_wall, _ = pytides_point(1e-6)
        print(f"pyTIDES cold start (first call, includes Numba JIT compilation): {cold_wall*1e3:.4f} ms")

    pytides_tols = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    pytides_pts = [pytides_point(tol) for tol in pytides_tols]
    print("\npyTIDES (TidesSolver, Numba-warmed force evaluation):")
    for tol, (wall, err) in zip(pytides_tols, pytides_pts):
        print(f"  tol={tol:.0e}  time={wall*1e3:.4f} ms  err={err:.3e}")

    # Pure-Python force evaluation (no Numba at all), same tolerances -- isolates
    # how much of the pyTIDES cost is the JIT-compiled force kernel vs. the
    # (never JIT-compiled, for a generic problem) TidesSolver driver loop itself
    # (order selection, Horner dense-output evaluation, step accept/reject).
    if HAS_NUMBA:
        pytides_pure_pts = [pytides_point(tol, force_func=_pure_python_force) for tol in pytides_tols]
        print("\npyTIDES (TidesSolver, pure-Python force evaluation, no Numba):")
        for tol, (wall, err) in zip(pytides_tols, pytides_pure_pts):
            print(f"  tol={tol:.0e}  time={wall*1e3:.4f} ms  err={err:.3e}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    w, e_ = zip(*pytides_pts)
    ax.loglog(e_, np.array(w) * 1e3, "o-", label="pyTIDES (Numba force path)", color="tab:red")
    if HAS_NUMBA:
        w, e_ = zip(*pytides_pure_pts)
        ax.loglog(e_, np.array(w) * 1e3, "o--", label="pyTIDES (pure-Python force)", color="tab:orange")

    if HAS_REBOUND:
        whfast_fracs = [1e-2, 1e-3, 1e-4, 1e-5]
        whfast_pts = [rebound_whfast_point(f) for f in whfast_fracs]
        print("\nREBOUND WHFast (dt swept as a fraction of the period):")
        for f, (wall, err) in zip(whfast_fracs, whfast_pts):
            print(f"  dt={f:.0e}*P  time={wall*1e3:.4f} ms  err={err:.3e}")
        w, e_ = zip(*whfast_pts)
        ax.loglog(e_, np.array(w) * 1e3, "s-", label="REBOUND WHFast (dt swept)", color="tab:blue")

        ias15_wall, ias15_err = rebound_ias15_point()
        print(f"\nREBOUND IAS15 (default settings): time={ias15_wall*1e3:.4f} ms  err={ias15_err:.3e}")
        ax.loglog([ias15_err], [ias15_wall * 1e3], "^", markersize=10,
                  label="REBOUND IAS15 (default)", color="tab:green")
    else:
        print("\nREBOUND not installed in this environment -- pyTIDES-only comparison.")

    ax.set_xlabel(r"state-closure error $\epsilon_{\mathrm{state}}$")
    ax.set_ylabel("wall-clock time [ms]")
    ax.set_title("Matched-accuracy comparison: Kepler two-body, one period")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "performance_matched_accuracy.png")


# ---------------------------------------------------------------------------
# Part 2: Numba vs. pure-Python scaling
# ---------------------------------------------------------------------------

def build_ring_system(n_planets):
    from exotides.orbital import HierarchicalSystem
    system = HierarchicalSystem(G=1.0)
    system.add_body("Star", mass=1.0)
    for k in range(n_planets):
        system.add_body(
            f"Planet{k}", mass=1.0e-3, parent_name="Star",
            elements={"a": 1.0 + 0.3 * k, "e": 0.05, "i": 0.02 * (k % 4),
                      "lan": 0.1 * k, "aop": 0.0, "ta": 0.0},
        )
    return system.generate()


def _time_solver(mincseries_func, v_init, p_init, tend, dt):
    solver = TidesSolver(
        mincseries_func=mincseries_func, nvar=len(v_init), npar=len(p_init),
        tolrel=1e-13, tolabs=1e-13, maxord=24, minord=8,
    )
    t0 = time.perf_counter()
    solver.solve(v_init, p_init, tini=0.0, tend=tend, dt=dt)
    return time.perf_counter() - t0


def run_scaling():
    print("\n=== Part 2: Numba vs. pure-Python Newtonian-core scaling ===\n")
    if not HAS_NUMBA:
        print("numba not installed -- skipping (nbody_mincseries falls back to pure Python).")
        return

    def pure_python_core(t, v, p, XVAR, ORDER, MO):
        return _nbody_mincseries_core(t, v, p, XVAR, ORDER, MO)

    body_counts = (2, 3, 4, 6, 8, 10, 12, 16)
    v_init, p_init, _ = build_ring_system(1)
    _time_solver(nbody_mincseries, v_init, p_init, 0.5, 0.5)  # JIT warm-up, excluded from timing

    rows = []
    for n_bodies in body_counts:
        v_init, p_init, _ = build_ring_system(n_bodies - 1)
        fast_time = _time_solver(nbody_mincseries, v_init, p_init, 3.0, 0.5)
        pure_time = _time_solver(pure_python_core, v_init, p_init, 3.0, 0.5)
        speedup = pure_time / fast_time if fast_time > 0 else float("inf")
        pairs = n_bodies * (n_bodies - 1) // 2
        rows.append((n_bodies, pairs, pure_time, fast_time, speedup))
        print(f"  N={n_bodies:3d}  pairs={pairs:4d}  pure={pure_time:.4f}s  numba={fast_time:.4f}s  speedup={speedup:.2f}x")

    fig, ax = plt.subplots(figsize=(6, 5))
    ns = [r[0] for r in rows]
    ax.plot(ns, [r[2] for r in rows], "o-", label="pure Python", color="tab:red")
    ax.plot(ns, [r[3] for r in rows], "s-", label="Numba JIT", color="tab:blue")
    ax.set_xlabel("number of bodies N")
    ax.set_ylabel("wall-clock time [s]")
    ax.set_title("Newtonian coefficient generator: pure-Python vs. Numba")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "performance_numba_scaling.png")
    return rows


def main():
    run_matched_accuracy()
    run_scaling()


if __name__ == "__main__":
    main()
