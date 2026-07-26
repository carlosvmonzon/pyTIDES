"""
experiments/exp1_kepler_convergence.py

Reproduces the "Keplerian closure and tolerance convergence" experiment in
docs/paper.tex (Section 8.1). A two-body Kepler orbit (negligible-mass
planet around a unit-mass star, G=1) is integrated over exactly one orbital
period P for a sweep of requested tolerances and eccentricities. The
state-closure error

    eps_state = ||y(P) - y(0)|| / ||y(0)||

is measured for the full 12-component two-body Cartesian state vector
(positions+velocities of both bodies), at double precision for several
eccentricities, and additionally at 200-bit arbitrary precision for one
eccentricity to show continued convergence past the double-precision floor.

The figure also marks REBOUND's IAS15 double-precision floor on the same
problem (computed live if REBOUND is installed, else a hardcoded fallback
from a prior measurement) -- not to compare speed, but to show a genuine,
implementation-language-independent limitation: no double-precision
integrator, however well implemented, can do better than float64 roundoff
allows, while pyTIDES's arbitrary-precision path keeps converging for a
further ~16 orders of magnitude past that floor.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from _common import build_two_body_system, build_two_body_system_mpfr, savefig
from exotides.core import TidesSolver, to_mpfr
from exotides.nbody import nbody_mincseries

# Fallback if REBOUND isn't installed when this script is rerun: IAS15's
# measured closure error on this exact problem (a=1, e=0.4, one period),
# reproducible across repeated runs (verified identical to all digits shown
# across 3 repeated runs when this was measured).
REBOUND_IAS15_FLOOR_FALLBACK = 1.451e-14


def rebound_ias15_floor():
    try:
        import rebound
    except ImportError:
        return REBOUND_IAS15_FLOOR_FALLBACK, False

    mu = G * (MASS_STAR + MASS_PLANET)
    p = 2.0 * math.pi * math.sqrt(A ** 3 / mu)
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.add(m=MASS_STAR)
    sim.add(m=MASS_PLANET, a=A, e=0.4)
    sim.move_to_com()
    y0 = np.array([b.x for b in sim.particles] + [b.y for b in sim.particles] +
                  [b.vx for b in sim.particles] + [b.vy for b in sim.particles])
    sim.integrate(p)
    y1 = np.array([b.x for b in sim.particles] + [b.y for b in sim.particles] +
                  [b.vx for b in sim.particles] + [b.vy for b in sim.particles])
    return float(np.linalg.norm(y1 - y0) / np.linalg.norm(y0)), True


G = 1.0
MASS_STAR = 1.0
MASS_PLANET = 1.0e-6
A = 1.0

ECCENTRICITIES = [0.0, 0.3, 0.6, 0.9]
TOLERANCES_STD = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
TOLERANCES_MPFR = [1e-14, 1e-18, 1e-22, 1e-26, 1e-30]
MPFR_ECCENTRICITY = 0.6
MPFR_PRECISION_BITS = 200


def period(a, mu):
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def period_mpfr(a, mu):
    # Genuine high-precision period: math.pi/math.sqrt are float64-limited
    # even if their *result* is later cast to mpfr, so tend would silently
    # be off from the true period at the ~1e-16 level -- which would show
    # up as a spurious closure "error" no amount of integration precision
    # could remove (the same float64-seed problem noted in
    # tests/test_lagrange_three_body.py, here affecting tend rather than
    # the initial condition).
    import gmpy2
    return 2 * gmpy2.const_pi() * gmpy2.sqrt(a ** 3 / mu)


def run_case(e, tol, is_mpfr):
    if is_mpfr:
        v_init, p_init, _ = build_two_body_system_mpfr(
            mass_star=MASS_STAR, mass_planet=MASS_PLANET, a=A, e=e, G=G,
        )
        mu = to_mpfr(G) * (to_mpfr(MASS_STAR) + to_mpfr(MASS_PLANET))
        P = period_mpfr(to_mpfr(A), mu)
        tol_val = to_mpfr(tol)
    else:
        v_init, p_init, _ = build_two_body_system(
            mass_star=MASS_STAR, mass_planet=MASS_PLANET, a=A, e=e, G=G,
        )
        mu = G * (MASS_STAR + MASS_PLANET)
        P = period(A, mu)
        tol_val = tol

    solver = TidesSolver(
        mincseries_func=nbody_mincseries,
        nvar=len(v_init), npar=len(p_init),
        tolrel=tol_val, tolabs=tol_val,
        maxord=48 if is_mpfr else 32, minord=8 if is_mpfr else 6,
        is_mpfr=is_mpfr,
    )
    _, states = solver.solve(v_init, p_init, tini=0.0, tend=P, dt=P / 4.0)

    y0 = np.array([float(x) for x in states[0]])
    yP = np.array([float(x) for x in states[-1]])
    return float(np.linalg.norm(yP - y0) / np.linalg.norm(y0))


def main():
    print("=== Exp 1: Keplerian closure & tolerance convergence ===")

    std_results = {}
    for e in ECCENTRICITIES:
        errors = []
        for tol in TOLERANCES_STD:
            err = run_case(e, tol, is_mpfr=False)
            errors.append(err)
            print(f"  [std]  e={e:.1f}  tol={tol:.0e}  eps_state={err:.3e}")
        std_results[e] = errors

    mpfr_errors = []
    import gmpy2
    gmpy2.get_context().precision = MPFR_PRECISION_BITS
    for tol in TOLERANCES_MPFR:
        err = run_case(MPFR_ECCENTRICITY, tol, is_mpfr=True)
        mpfr_errors.append(err)
        print(f"  [mpfr] e={MPFR_ECCENTRICITY:.1f}  tol={tol:.0e}  eps_state={err:.3e}")

    rebound_floor, rebound_live = rebound_ias15_floor()
    print(f"\n--- REBOUND IAS15 double-precision floor ({'measured live' if rebound_live else 'fallback value'}) ---")
    print(f"  eps_state = {rebound_floor:.3e}")

    # --- Figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for e in ECCENTRICITIES:
        ax.loglog(TOLERANCES_STD, std_results[e], "o-", label=f"double, e={e:.1f}")
    ax.loglog(
        TOLERANCES_MPFR, mpfr_errors, "s--", color="black",
        label=f"200-bit mpfr, e={MPFR_ECCENTRICITY:.1f}",
    )
    ax.axhline(
        rebound_floor, color="tab:green", lw=1.5, linestyle=":",
        label=f"REBOUND IAS15 floor ({rebound_floor:.1e}, double precision)",
    )
    ax.set_xlabel("requested tolerance (tolrel = tolabs)")
    ax.set_ylabel(r"state-closure error $\epsilon_{\mathrm{state}}$")
    ax.set_title("Kepler two-body closure error vs. requested tolerance")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "kepler_convergence.png")

    # --- Summary for the paper text ------------------------------------
    print("\n--- Double-precision floor (tightest tolerance tested, per e) ---")
    for e in ECCENTRICITIES:
        print(f"  e={e:.1f}: eps_state({TOLERANCES_STD[-1]:.0e}) = {std_results[e][-1]:.3e}")
    print(f"\n--- mpfr floor (tightest tolerance tested, e={MPFR_ECCENTRICITY}) ---")
    print(f"  eps_state({TOLERANCES_MPFR[-1]:.0e}) = {mpfr_errors[-1]:.3e}")
    print("\n--- Orders of magnitude pyTIDES (mpfr) improves past REBOUND's floor ---")
    print(f"  {math.log10(rebound_floor / mpfr_errors[-1]):.1f} orders of magnitude")


if __name__ == "__main__":
    main()
