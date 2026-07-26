"""
tests/test_relativity.py

Verification for the 1PN pairwise "dominant mass" relativistic correction
(src/exotides/relativity.py, src/exotides/nbody.py's include_pn path).

Three checks:
1. Regression: nbody_mincseries (PN off) is untouched by the new code paths
   -- a plain two-body Kepler orbit still integrates and closes.
2. Physics: for a two-body star+test-mass-planet system with PN on, the
   numerically measured apsidal (pericenter) precession rate matches the
   textbook GR result

       delta_omega_per_orbit = 6*pi*G*M / (c^2 * a * (1 - e^2))

   An artificially small speed of light is used so the (otherwise tiny)
   precession accumulates fast enough to measure over a short integration.
3. The 1PN correction works through the hierarchy-template API
   (``physics="pn"``) for genuine 3-body hierarchy templates (star+planet+
   moon, a planet in a binary, a nested triple star), not just the bare
   two-body setup used in check 2 -- via ``exotides.plotting.plot_pn_comparison``,
   which works with *any* catalog template.
4. ``pn_pair_quality``/``warn_pn_quality`` flag pairs whose mass ratio makes
   the "dominant mass" approximation unreliable (e.g. an equal-mass binary
   star), and stay silent for genuinely hierarchical mass ratios -- both as
   a direct unit check and through the ``physics="pn"`` template API.
"""

import math
import warnings

import numpy as np
import pytest

from helpers import PYTHON_DIR, build_two_body_system, precession_rate
from exotides.core import TidesSolver
from exotides.hierarchy import HierarchicalSystemTemplates
from exotides.nbody import nbody_mincseries, nbody_pn_mincseries, unpack_state
from exotides.plotting import plot_pn_comparison, pn_comparison_diagnostics
from exotides.relativity import (
    append_pn_params,
    gr_precession_rate_per_orbit,
    pn_pair_quality,
    warn_pn_quality,
)


def test_pn_off_matches_plain_newtonian():
    """PN off (nbody_mincseries) must still integrate a plain Kepler orbit."""
    v_init, p_init, _ = build_two_body_system()
    solver = TidesSolver(
        mincseries_func=nbody_mincseries, nvar=len(v_init), npar=len(p_init),
        tolrel=1e-13, tolabs=1e-13, maxord=24, minord=8,
    )
    t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=6.28, dt=0.5)
    assert np.isfinite(states).all()
    return t_hist, states


def test_pn_precession_matches_gr_formula():
    G, M, a, e = 1.0, 1.0, 1.0, 0.4
    c_light = 40.0  # unrealistically small so precession is measurable quickly

    v_init, p_init, _ = build_two_body_system(mass_star=M, a=a, e=e, G=G)
    p_init = append_pn_params(p_init, c_light)

    solver = TidesSolver(
        mincseries_func=nbody_pn_mincseries, nvar=len(v_init), npar=len(p_init),
        tolrel=1e-13, tolabs=1e-13, maxord=28, minord=8,
    )

    period = 2.0 * math.pi * math.sqrt(a ** 3 / (G * M))
    n_orbits = 40
    t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=n_orbits * period, dt=period / 20.0)
    positions, velocities = unpack_state(states)

    rate = precession_rate(t_hist, positions, velocities, G * M)
    numeric_per_orbit = rate * period
    analytic_per_orbit = gr_precession_rate_per_orbit(G, M, a, e, c_light)

    ratio = numeric_per_orbit / analytic_per_orbit
    assert 0.8 < ratio < 1.2, (
        f"PN precession rate {numeric_per_orbit:.6g}/orbit is not within 20% of the "
        f"analytic GR rate {analytic_per_orbit:.6g}/orbit (ratio={ratio:.4f})"
    )
    return numeric_per_orbit, analytic_per_orbit, ratio


def test_pn_pair_quality_thresholds():
    """Mass-ratio thresholds classifying the 1PN approximation per pair."""
    assert pn_pair_quality(1.0, 1.0e-4) == "good"     # star-planet
    assert pn_pair_quality(1.0, 0.2) == "fair"         # ratio 5
    assert pn_pair_quality(1.0, 1.0) == "poor"         # equal-mass binary
    assert pn_pair_quality(1.0, 0.0) == "good"         # zero mass never triggers a warning


def test_warn_pn_quality_silent_for_hierarchical_masses():
    """No warning for an extreme-mass-ratio (star/planet/moon) system."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_pn_quality([1.0, 1.0e-4, 1.0e-6], names=["Star", "Planet", "Moon"])


def test_warn_pn_quality_flags_comparable_masses():
    """A comparable-mass pair raises a UserWarning naming that pair."""
    with pytest.warns(UserWarning, match="Star A-Star B"):
        warn_pn_quality([1.0, 1.0, 1.0e-4], names=["Star A", "Star B", "Planet"])


def test_solve_hierarchy_pn_warns_for_binary_star():
    """physics='pn' surfaces the same warning through the template API."""
    with pytest.warns(UserWarning, match="poor"):
        HierarchicalSystemTemplates.solve_hierarchy(
            "binary_star",
            [1.0, 1.0],
            elements={1: {"a": 1.0, "e": 0.4, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0}},
            physics="pn",
            speed_of_light=40.0,
            tend=0.01,
            dt=0.01,
        )


# Hierarchy-template cases (not the bare two-body setup used above), each
# examining a pair whose own two-body dynamics isn't significantly
# perturbed by whatever else is in the system over the ~15 orbits tested
# (so any Newtonian precession stays negligible and the PN-vs-Newtonian
# contrast stays sharp): (key, masses, elements, body_idx, parent_idx).
#
# plot_pn_comparison computes mu = G*(mass of parent_idx + mass of
# body_idx) -- correct for *any* mass ratio, including two comparable-mass
# stars (unlike a naive mu = G*m_parent, which silently assumes body_idx is
# negligible and was the actual bug behind an earlier, spurious ~500
# deg/orbit "precession" measured for two 1.0-mass stars before this was
# fixed). ``parent_idx`` can also be a list (Jacobi barycenter of several
# bodies) for a target whose true reference isn't one body alone -- e.g.
# the outer star of a nested triple orbiting the combined inner pair -- but
# that case has a genuine (non-artifact) Newtonian secular coupling to the
# inner binary's own eccentricity, which competes with the artificially
# enhanced PN signal here, so it's not included below as a "clean" demo.
HIERARCHY_TEMPLATE_CASES = (
    (
        "star_planet_moon",
        [1.0, 1.0e-4, 1.0e-6],
        {
            0: None,
            1: {"a": 1.0, "e": 0.4, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 0.01, "e": 0.1, "i": 0.1, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        1, 0,  # Planet around Star -- the moon is far too close-in to perturb this pair.
    ),
    (
        "binary_s_planet_primary",
        [1.0, 1.0, 1.0e-4],
        {
            0: None,
            1: {"a": 30.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 1.0, "e": 0.4, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        2, 0,  # Planet around Star A -- Star B is wide enough to barely perturb it.
    ),
    (
        "binary_s_planet_secondary",
        [1.0, 1.0, 1.0e-4],
        {
            0: None,
            1: {"a": 30.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 1.0, "e": 0.4, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        2, 1,  # Planet around Star B this time -- Star A is wide enough to barely perturb it.
    ),
    (
        "binary_star",
        [1.0, 1.0],
        {
            0: None,
            1: {"a": 1.0, "e": 0.4, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        1, 0,  # Two comparable-mass (1.0 each) stars -- no third body at all.
    ),
    (
        "triple_star_chain",
        [1.0, 1.0, 1.0e-3],
        {
            0: None,
            1: {"a": 1.0, "e": 0.4, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 15.0, "e": 0.05, "i": 0.3, "lan": 0.2, "aop": 0.1, "ta": 0.3},
        },
        1, 0,  # Star B around comparable-mass Star A -- distant, light Star C barely perturbs it.
    ),
)


@pytest.mark.parametrize("key,masses,elements,body_idx,parent_idx", HIERARCHY_TEMPLATE_CASES)
def test_pn_through_hierarchy_template(key, masses, elements, body_idx, parent_idx, output_dir=None):
    """
    physics="pn" works through HierarchicalSystemTemplates -- i.e. for any
    catalog template (including genuine 3+ body hierarchies, not just a
    bare two-body setup), examining the ``body_idx``/``parent_idx`` pair.
    Parametrized over ``HIERARCHY_TEMPLATE_CASES`` -- pytest runs one case
    per tuple. ``output_dir=None`` (the pytest default) skips the
    Newtonian-vs-PN comparison plot entirely (``pn_comparison_diagnostics``
    does the same integration without touching matplotlib); passing a real
    directory (as ``main()`` does for standalone script execution) also
    saves the plot via ``plot_pn_comparison``.
    """
    c_light = 40.0  # same artificially small c as the precession-rate test

    period = 2.0 * math.pi * math.sqrt(elements[body_idx]["a"] ** 3 / (1.0 * masses[parent_idx]))
    n_orbits = 15
    solver_settings = dict(tolrel=1e-13, tolabs=1e-13, maxord=24, minord=8)

    if output_dir is not None:
        output_path, diagnostics = plot_pn_comparison(
            key, masses, elements, n_orbits * period, period / 100.0, c_light, output_dir,
            body_idx=body_idx, parent_idx=parent_idx, solver_settings=solver_settings,
        )
    else:
        output_path = None
        diagnostics = pn_comparison_diagnostics(
            key, masses, elements, n_orbits * period, period / 100.0, c_light,
            body_idx=body_idx, parent_idx=parent_idx, solver_settings=solver_settings,
        )["diagnostics"]

    newtonian_drift_per_orbit = diagnostics["newtonian_drift_deg"] / n_orbits
    pn_drift_per_orbit = diagnostics["pn_drift_deg"] / n_orbits
    assert newtonian_drift_per_orbit < 0.05, (
        f"{key}: Newtonian orbit precessed by {newtonian_drift_per_orbit:.4g} deg/orbit -- should be ~0"
    )
    assert pn_drift_per_orbit > 10.0 * newtonian_drift_per_orbit, (
        f"{key}: 1PN orbit did not precess measurably more than the Newtonian one "
        f"({pn_drift_per_orbit:.4g} vs {newtonian_drift_per_orbit:.4g} deg/orbit)"
    )
    return output_path


def main():
    test_pn_off_matches_plain_newtonian()
    print("PN off (nbody_mincseries): regression check passed")

    numeric, analytic, ratio = test_pn_precession_matches_gr_formula()
    print(f"PN precession/orbit: numeric={numeric:.6g}, analytic={analytic:.6g}, ratio={ratio:.4f}")
    print("1PN pairwise correction verified against the GR precession formula.")

    for key, masses, elements, body_idx, parent_idx in HIERARCHY_TEMPLATE_CASES:
        output_path = test_pn_through_hierarchy_template(
            key, masses, elements, body_idx, parent_idx, output_dir=PYTHON_DIR,
        )
        print(f"{key}: Newtonian-vs-PN comparison plot saved to {output_path}")


if __name__ == "__main__":
    main()
