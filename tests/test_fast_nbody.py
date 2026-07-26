"""
tests/test_fast_nbody.py

Verification for the optional Numba-accelerated Newtonian core
(src/exotides/_fast_nbody.py, dispatched from exotides.nbody.nbody_mincseries).

If Numba isn't installed, this is a no-op (HAS_NUMBA is False and
nbody_mincseries already falls back to the pure-Python core transparently
-- nothing to verify here in that case).

Three checks:
1. Correctness: the Numba path and the pure-Python core produce identical
   trajectories for the same hierarchical system.
2. Execution-time study: sequential (pure-Python) vs. Numba wall-clock time
   across a range of body counts N. The per-order force loop in
   src/exotides/nbody.py is O(N^2) (every pair, every Taylor order), so the
   speedup factor isn't guaranteed to be constant with N -- this measures
   it directly at several sizes rather than assuming one number from a
   single small system generalizes.
3. A sanity assertion that the Numba path stays faster than pure Python at
   every size tested (not just reported).
"""

import time

import numpy as np
import pytest

from helpers import build_system
from exotides.core import TidesSolver
from exotides.nbody import nbody_mincseries, _nbody_mincseries_core
from exotides import _fast_nbody


def build_triple_system():
    return build_system([
        {"name": "Star", "mass": 1.0},
        {
            "name": "Planet", "mass": 1.0e-3, "parent_name": "Star",
            "elements": {"a": 1.0, "e": 0.4, "i": 0.1, "lan": 0.2, "aop": 0.3, "ta": 0.0},
        },
        {
            "name": "Moon", "mass": 1.0e-5, "parent_name": "Planet",
            "elements": {"a": 0.01, "e": 0.1, "i": 0.05, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
    ])


def build_ring_system(n_planets, G=1.0):
    """
    A central star with ``n_planets`` S-type planets at increasing
    semi-major axes (total bodies = ``n_planets + 1``), built directly via
    ``HierarchicalSystem`` (``helpers.build_system``) rather than a named
    ``exotides.hierarchy`` catalog template. The catalog caps out at 5 bodies
    (3 stars/2 planets/1 moon, see ``src/exotides/hierarchy.py``'s
    ``MAX_STARS``/``MAX_PLANETS``/``MAX_MOONS``); the underlying N-body
    engine this module accelerates has no such limit (``nbody_mincseries``
    loops over a fully generic ``N``, see ``docs/design-notes.md`` section
    4) -- this builder exists purely to scale N past what the catalog
    documents, for this execution-time study.
    """
    bodies = [{"name": "Star", "mass": 1.0}]
    for k in range(n_planets):
        bodies.append({
            "name": f"Planet{k}", "mass": 1.0e-3, "parent_name": "Star",
            "elements": {
                "a": 1.0 + 0.3 * k, "e": 0.05, "i": 0.02 * (k % 4),
                "lan": 0.1 * k, "aop": 0.0, "ta": 0.0,
            },
        })
    return build_system(bodies, G=G)


def _pure_python_mincseries(t, v, p, XVAR, ORDER, MO):
    return _nbody_mincseries_core(t, v, p, XVAR, ORDER, MO)


def test_numba_matches_pure_python():
    v_init, p_init, _ = build_triple_system()
    solver_settings = dict(tolrel=1e-13, tolabs=1e-13, maxord=24, minord=8)

    solver_fast = TidesSolver(mincseries_func=nbody_mincseries, nvar=len(v_init), npar=len(p_init), **solver_settings)
    solver_pure = TidesSolver(mincseries_func=_pure_python_mincseries, nvar=len(v_init), npar=len(p_init), **solver_settings)

    t_fast, states_fast = solver_fast.solve(v_init, p_init, tini=0.0, tend=5.0, dt=0.1)
    t_pure, states_pure = solver_pure.solve(v_init, p_init, tini=0.0, tend=5.0, dt=0.1)

    assert np.allclose(t_fast, t_pure)
    max_diff = float(np.max(np.abs(states_fast - states_pure)))
    assert max_diff < 1e-10, f"Numba and pure-Python paths diverged: max abs diff = {max_diff:.3e}"
    return max_diff


# Total body counts to probe in the scaling study (a central star plus 1..15
# planets). Pair count P = N*(N-1)/2 grows quadratically, which is what the
# per-order force loop in src/exotides/nbody.py actually scales with.
BODY_COUNTS = (2, 3, 4, 6, 8, 10, 12, 16)


def _time_solver(mincseries_func, v_init, p_init, solver_settings, tend, dt):
    solver = TidesSolver(mincseries_func=mincseries_func, nvar=len(v_init), npar=len(p_init), **solver_settings)
    t0 = time.perf_counter()
    solver.solve(v_init, p_init, tini=0.0, tend=tend, dt=dt)
    return time.perf_counter() - t0


def run_scaling_study():
    """
    Times the Numba and pure-Python cores across ``BODY_COUNTS``, returning
    one ``(n_bodies, pair_count, pure_time, fast_time, speedup)`` tuple per
    size.
    """
    solver_settings = dict(tolrel=1e-13, tolabs=1e-13, maxord=24, minord=8)
    tend, dt = 3.0, 0.5

    # Warm up the JIT once before timing anything (first call pays
    # compilation cost; every body count after this reuses the same
    # compiled float64 core, since Numba specializes on argument dtypes,
    # not array shapes).
    v_init, p_init, _ = build_ring_system(1)
    _time_solver(nbody_mincseries, v_init, p_init, solver_settings, tend=0.5, dt=0.5)

    rows = []
    for n_bodies in BODY_COUNTS:
        n_planets = n_bodies - 1
        pair_count = n_bodies * (n_bodies - 1) // 2
        v_init, p_init, _ = build_ring_system(n_planets)

        fast_time = _time_solver(nbody_mincseries, v_init, p_init, solver_settings, tend, dt)
        pure_time = _time_solver(_pure_python_mincseries, v_init, p_init, solver_settings, tend, dt)
        speedup = pure_time / fast_time if fast_time > 0 else float("inf")
        rows.append((n_bodies, pair_count, pure_time, fast_time, speedup))
    return rows


def test_numba_speedup_scaling():
    """Pytest-discoverable entry point for the execution-time study. Skips
    (rather than silently passing) if numba isn't installed -- there is
    nothing to benchmark in that case, see the module docstring."""
    if not _fast_nbody.HAS_NUMBA:
        pytest.skip("numba is not installed -- nothing to benchmark")

    rows = run_scaling_study()
    min_speedup = min(row[4] for row in rows)
    assert min_speedup > 1.0, (
        f"Numba was not faster than pure Python at every body count tested "
        f"(worst case: {min_speedup:.2f}x)"
    )
    return rows


def main():
    if not _fast_nbody.HAS_NUMBA:
        print("Numba is not installed -- nbody_mincseries falls back to the pure-Python "
              "core transparently. Nothing to verify.")
        return

    max_diff = test_numba_matches_pure_python()
    print(f"Numba vs pure-Python max state diff: {max_diff:.3e} (correctness OK)\n")

    rows = test_numba_speedup_scaling()
    print(f"{'N bodies':>9}  {'pairs':>6}  {'pure[s]':>9}  {'numba[s]':>9}  {'speedup':>8}")
    for n_bodies, pair_count, pure_time, fast_time, speedup in rows:
        print(f"{n_bodies:>9}  {pair_count:>6}  {pure_time:>9.3f}  {fast_time:>9.3f}  {speedup:>7.2f}x")

    min_speedup = min(row[4] for row in rows)
    print(f"\nOK: Numba is faster than pure Python at every body count tested "
          f"(N={BODY_COUNTS[0]}..{BODY_COUNTS[-1]}, worst-case speedup {min_speedup:.2f}x).")


if __name__ == "__main__":
    main()
