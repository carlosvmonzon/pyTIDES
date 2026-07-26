"""
tests/test_kepler.py

Regression test for the classic planar Kepler two-body problem
(``helpers.kepler_mincseries``, nvar=4: x, y, vx, vy).

Integrates exactly one orbital period (``tend`` = the period itself) and
asserts the final state matches the initial state to within 1e-10 -- a
closed elliptical orbit must return to its starting point after one full
period, so any integrator error shows up directly as drift. Runs in both
``std`` (float64) and ``mpfr`` (arbitrary precision) modes -- see
``parse_precision_mode``.
"""

import sys

from helpers import (
    assert_close,
    configure_precision,
    kepler_mincseries,
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
        vector_to_precision([0.30000000000000004, 0.0, 0.0, 2.3804761428476167], mode),
        vector_to_precision([1.0], mode),
        to_precision(0.0, mode),
        to_precision(62.83185307179586, mode),
        to_precision(62.83185307179586, mode),
    )


def test_kepler_period_closure(mode="std"):
    """Pytest-discoverable entry point -- ``mode`` has a default, so pytest
    calls this with no fixture involved, always exercising ``std``; the
    ``mpfr`` mode stays a manual CLI-only check (``python test_kepler.py
    mpfr``), same as before this was made pytest-compatible."""
    values = make_values(mode)
    if values is None:
        return

    initial_state, parameters, tini, tend, dt = values
    settings = solver_settings(mode, std_tol=1e-16)
    solver = TidesSolver(
        mincseries_func=kepler_mincseries,
        nvar=4,
        npar=1,
        **settings,
    )

    _, states = solver.solve(initial_state, parameters, tini, tend, dt)
    error = assert_close(states[-1], initial_state, 1e-10, f"test_kepler {mode}")
    print(f"test_kepler {mode} max error: {error:.17e}")
    return error


def main():
    mode = parse_precision_mode(sys.argv, "Python/tests/test_kepler.py")
    test_kepler_period_closure(mode)


if __name__ == "__main__":
    main()
