"""
experiments/exp6_event_localization.py

Reproduces the "Event localization" experiment in docs/paper.tex
(Section 8.6): terminal and non-terminal events tested against
analytically/independently known crossing times, with event-time error
reported as a function of tolerance.

Both reference times below are computed in closed form from the Keplerian
two-body relations (true anomaly -> eccentric anomaly -> mean anomaly ->
time), independent of exotides.core.TidesSolver's own dense-output
root-finding -- not by re-running the integrator at finer resolution.

1. Non-terminal: a circular, inclined orbit's ascending/descending node
   crossings (z=0). Starting 90 degrees past the ascending node, crossings
   occur at exactly every half period thereafter (t = P/4, 3P/4, 5P/4, ...)
   -- closed-form because e=0 makes true anomaly, eccentric anomaly, and
   mean anomaly identical.
2. Terminal: an eccentric (e=0.5) two-body orbit with a collision radius
   equal to the semi-latus rectum p = a(1-e^2), so the (first, approaching)
   crossing occurs at true anomaly 270 degrees starting from apoapsis (true
   anomaly 180 degrees) -- computed via the standard nu->E->M relations.
"""

import math

import matplotlib.pyplot as plt

from _common import build_two_body_system, savefig
from exotides.core import TidesSolver
from exotides.events import Event, collision_event
from exotides.nbody import nbody_mincseries

G = 1.0
MASS_STAR = 1.0
MASS_PLANET = 1.0e-6
MU = G * (MASS_STAR + MASS_PLANET)


# ---------------------------------------------------------------------------
# 1. Non-terminal: node crossings on a circular inclined orbit
# ---------------------------------------------------------------------------

def node_crossing_errors(tol):
    a = 1.0
    inc = 0.3
    v_init, p_init, _ = build_two_body_system(
        mass_star=MASS_STAR, mass_planet=MASS_PLANET, a=a, e=0.0, i=inc,
        lan=0.0, aop=0.0, ta=math.pi / 2.0, G=G,
    )
    period = 2.0 * math.pi * math.sqrt(a ** 3 / MU)

    def z_of_planet(t, v):
        return v[6 * 1 + 2]

    node_event = Event(z_of_planet, terminal=False, direction=0, name="node_crossing")

    solver = TidesSolver(
        mincseries_func=nbody_mincseries, nvar=len(v_init), npar=len(p_init),
        tolrel=tol, tolabs=tol, maxord=28, minord=8,
    )
    n_periods = 5
    solver.solve(v_init, p_init, tini=0.0, tend=n_periods * period, dt=period / 20.0, events=[node_event])

    detected = sorted(float(ev["time"]) for ev in solver.last_events)
    analytic = [period / 4.0 + k * period / 2.0 for k in range(len(detected))]
    errors = [abs(d - a_) for d, a_ in zip(detected, analytic)]
    return max(errors) if errors else float("nan"), len(detected)


# ---------------------------------------------------------------------------
# 2. Terminal: collision at an analytically known true anomaly
# ---------------------------------------------------------------------------

def true_anomaly_to_time(nu, e, n):
    """Closed-form time since periapsis for true anomaly `nu` (radians), mean motion n."""
    E = math.atan2(math.sqrt(1.0 - e ** 2) * math.sin(nu), e + math.cos(nu))
    if E < 0:
        E += 2.0 * math.pi
    M = E - e * math.sin(E)
    return M / n


def collision_time_error(tol):
    a = 1.0
    e = 0.5
    v_init, p_init, _ = build_two_body_system(
        mass_star=MASS_STAR, mass_planet=MASS_PLANET, a=a, e=e, i=0.0,
        lan=0.0, aop=0.0, ta=math.pi, G=G,  # start at apoapsis (nu=180 deg)
    )
    period = 2.0 * math.pi * math.sqrt(a ** 3 / MU)
    n = 2.0 * math.pi / period

    p_semi_latus = a * (1.0 - e ** 2)
    r0, r1 = 0.5 * p_semi_latus, 0.5 * p_semi_latus  # sum of "radii" = p_semi_latus
    event = collision_event(0, 1, r0, r1)

    t_start = true_anomaly_to_time(math.pi, e, n)          # apoapsis, nu=180 deg
    t_target = true_anomaly_to_time(3.0 * math.pi / 2.0, e, n)  # nu=270 deg
    analytic_elapsed = (t_target - t_start) % period

    solver = TidesSolver(
        mincseries_func=nbody_mincseries, nvar=len(v_init), npar=len(p_init),
        tolrel=tol, tolabs=tol, maxord=28, minord=8,
    )
    t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=period, dt=period / 20.0, events=[event])

    if not solver.last_events:
        return float("nan"), None
    detected = float(solver.last_events[0]["time"])
    return abs(detected - analytic_elapsed), detected


def main():
    print("=== Exp 6: Event localization ===\n")

    tolerances = [1e-8, 1e-10, 1e-12, 1e-13]

    print("--- Non-terminal: node-crossing errors ---")
    node_errors = []
    for tol in tolerances:
        err, n_detected = node_crossing_errors(tol)
        node_errors.append(err)
        print(f"  tol={tol:.0e}  max|error|={err:.3e}  ({n_detected} crossings detected)")

    print("\n--- Terminal: collision-time errors ---")
    collision_errors = []
    for tol in tolerances:
        err, t_detected = collision_time_error(tol)
        collision_errors.append(err)
        print(f"  tol={tol:.0e}  |error|={err:.3e}  (detected t={t_detected})")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(tolerances, node_errors, "o-", label="non-terminal (node crossing)", color="tab:blue")
    ax.loglog(tolerances, collision_errors, "s-", label="terminal (collision)", color="tab:red")
    ax.set_xlabel("requested tolerance (tolrel = tolabs)")
    ax.set_ylabel("event-time error vs. analytic crossing time")
    ax.set_title("Event localization accuracy vs. tolerance")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "event_localization.png")

    print("\n--- Summary ---")
    print(f"  Node crossings: error shrinks from {node_errors[0]:.2e} to {node_errors[-1]:.2e} "
          f"over tol={tolerances[0]:.0e}..{tolerances[-1]:.0e}.")
    print(f"  Collision: error shrinks from {collision_errors[0]:.2e} to {collision_errors[-1]:.2e} "
          f"over the same range.")


if __name__ == "__main__":
    main()
