"""
tests/test_lorenz.py

Regression test for the Lorenz attractor (``helpers.lorenz_mincseries``,
nvar=3, classic sigma=10/rho=28/beta=8/3 parameters).

The initial state and ``tend`` are a known unstable periodic orbit (UPO) of
the Lorenz system, so -- despite the system being chaotic -- integrating
for exactly one period must return to the starting point. This is a
stringent regression check: because nearby trajectories diverge
exponentially, any integrator error is amplified rather than damped, unlike
in a non-chaotic system. Asserts the final state matches the initial state
to within 1e-10. Runs in both ``std`` (float64) and ``mpfr`` (arbitrary
precision) modes -- see ``parse_precision_mode``.
"""

import sys

from helpers import (
    assert_close,
    configure_precision,
    lorenz_mincseries,
    parse_precision_mode,
    solver_settings,
    to_precision,
    vector_to_precision,
)
from exotides.core import TidesSolver


def make_values(mode):
    if not configure_precision(mode):
        return None
    return (
        vector_to_precision([-13.7636106821342, -19.5787519424518, 27.0], mode),
        vector_to_precision([10.0, 28.0, 2.666666666666667], mode),
        to_precision(0.0, mode),
        to_precision(1.558652210716175, mode),
        to_precision(1.558652210716175, mode),
    )


def test_lorenz_periodic_orbit(mode="std"):
    """Pytest-discoverable entry point -- ``mode`` has a default, so pytest
    calls this with no fixture involved, always exercising ``std``; the
    ``mpfr`` mode stays a manual CLI-only check (``python test_lorenz.py
    mpfr``), same as before this was made pytest-compatible."""
    values = make_values(mode)
    if values is None:
        return

    initial_state, parameters, tini, tend, dt = values
    settings = solver_settings(mode, std_tol=1e-16)
    solver = TidesSolver(
        mincseries_func=lorenz_mincseries,
        nvar=3,
        npar=3,
        **settings,
    )

    _, states = solver.solve(initial_state, parameters, tini, tend, dt)
    error = assert_close(states[-1], initial_state, 1e-10, f"test_lorenz {mode}")
    print(f"test_lorenz {mode} max error: {error:.17e}")
    return error


def main():
    mode = parse_precision_mode(sys.argv, "Python/tests/test_lorenz.py")
    test_lorenz_periodic_orbit(mode)


if __name__ == "__main__":
    main()
