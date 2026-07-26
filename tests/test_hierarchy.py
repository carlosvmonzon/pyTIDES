"""
tests/test_hierarchy.py

Verification for every semantic template in the hierarchy catalog
(src/exotides/hierarchy.py's ``HierarchicalSystemTemplates``).

Two checks:
1. Structural, for *every* catalog template: keys are unique, the expected
   entries are present/absent, ``validate()`` accepts it, the star/planet/
   moon parent-type rules hold, and ``build()``/``generate()`` produce a
   state vector and parameter vector of the right shape from
   ``default_elements()``'s smoke-test values.
2. Dynamical, for one example per template (``EXAMPLES``): hand-picked
   masses/elements (not the generic smoke-test defaults) chosen so the
   configuration is actually dynamically bound over its ``tend`` -- see the
   comment above ``EXAMPLES`` for the stability/Hill-radius conventions
   used to pick them. ``plot_examples`` integrates each and saves a 3D
   trajectory plot (+ optional animation) plus, for 3+ body systems, an
   orbital-elements-vs-time plot of the smallest body.

Runs in both ``std`` (float64) and ``mpfr`` (arbitrary precision) modes --
see ``parse_precision_mode`` (``python test_hierarchy.py [std|mpfr]``).
Only the *integration* runs at that precision (``TidesSolver.is_mpfr``,
set directly on the solver by ``solve_and_plot_hierarchy``); hierarchy
construction (``HierarchicalSystemTemplates``) always builds plain
``float64`` regardless of mode.
"""

import sys

from helpers import (
    PYTHON_DIR,
    assert_close,
    configure_precision,
    parse_precision_mode,
    reference_template_indices,
    select_smaller_body,
    solve_and_plot_hierarchy,
    state_index_by_name,
)
from helpers import solver_settings as precision_solver_settings
from exotides.hierarchy import (
    MAX_MOONS,
    MAX_PLANETS,
    MAX_STARS,
    HierarchicalSystemTemplates,
)
from exotides.plotting import plot_orbital_elements


def print_solution(key, t_hist, positions, velocities, nodes):
    print(f"\n{key}")
    print(f"  interval: {float(t_hist[0]):.6f} -> {float(t_hist[-1]):.6f}")
    print("  final state:")
    for body_idx, node in enumerate(nodes):
        pos = positions[-1, body_idx]
        vel = velocities[-1, body_idx]
        print(
            f"    {node.name}: "
            f"r=({pos[0]: .10e}, {pos[1]: .10e}, {pos[2]: .10e}), "
            f"v=({vel[0]: .10e}, {vel[1]: .10e}, {vel[2]: .10e})"
        )


# One example per template in the catalog. ``masses`` and ``elements`` are
# hand-picked (rather than the smoke-test ``default_elements``) so that each
# configuration is dynamically bound over its ``tend``:
#
# - Multi-star hierarchies use a wide enough separation ratio to satisfy the
#   Holman-Wiegert / Mardling-Aarseth stability limits.
# - In the "chain" triples (a third star orbiting one binary component, not
#   the pair's barycenter), that component's own gravity is all the code
#   accounts for -- so the inner pair is given a lopsided mass ratio (one
#   star near 1e-3) to keep that approximation accurate. See the note in
#   README.md.
# - Moons (and, in the lopsided chain triples, the S-type planet orbiting the
#   light star) are kept at ~0.3x the Hill radius of their immediate parent,
#   well under the ``MOON_HILL_FRACTION`` limit enforced by ``build()``.
#
# ``triple_star_s_planet_moon`` uses a shorter ``tend`` and a looser
# tolerance (see ``SOLVER_OVERRIDES``): its moon's period is so much faster
# than the outer star's that integrating even one full outer orbit would
# take several minutes at the default tolerance.
EXAMPLES = (
    (
        "single_star",
        [1.0],
        {0: None},
        1.0,
        0.5,
    ),
    (
        "star_planet",
        [1.0, 1.0e-4],
        {
            0: None,
            1: {"a": 1.0, "e": 0.01, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        13.0,
        0.02,
    ),
    (
        "star_planet_moon",
        [1.0, 1.0e-4, 1.0e-6],
        {
            0: None,
            1: {"a": 1.0, "e": 0.01, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 0.01, "e": 0.01, "i": 0.2, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        13.0,
        0.02,
    ),
    (
        "star_two_planets",
        [1.0, 1.0e-4, 1.0e-4],
        {
            0: None,
            1: {"a": 1.0, "e": 0.01, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 2.0, "e": 0.07, "i": 0.0, "lan": 0.3, "aop": 0.0, "ta": 0.0},
        },
        18.0,
        0.05,
    ),
    (
        "star_two_planets_inner_moon",
        [1.0, 1.0e-4, 1.0e-4, 1.0e-6],
        {
            0: None,
            1: {"a": 1.0, "e": 0.01, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 2.0, "e": 0.01, "i": 0.15, "lan": 0.3, "aop": 0.0, "ta": 0.0},
            3: {"a": 0.01, "e": 0.01, "i": 0.1, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        18.0,
        0.05,
    ),
    (
        "star_two_planets_outer_moon",
        [1.0, 1.0e-4, 1.0e-4, 1.0e-6],
        {
            0: None,
            1: {"a": 1.0, "e": 0.01, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 2.0, "e": 0.01, "i": 0.15, "lan": 0.3, "aop": 0.0, "ta": 0.0},
            3: {"a": 0.02, "e": 0.01, "i": 0.1, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        18.0,
        0.05,
    ),
    (
        "binary_star",
        [1.0, 1.0],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        13.0,
        0.03,
    ),
    (
        "binary_s_planet_primary",
        [1.0, 1.0, 1.0e-4],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 0.3, "e": 0.01, "i": 0.2, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        13.0,
        0.02,
    ),
    (
        "binary_s_planet_secondary",
        [1.0, 1.0, 1.0e-4],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 0.3, "e": 0.01, "i": 0.2, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        13.0,
        0.02,
    ),
    (
        "binary_p_planet",
        [1.0, 1.0, 1.0e-4],
        {
            0: None,
            1: {"a": 2.0, "e": 0.0, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 25.0, "e": 0.0, "i": 0.2, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        580.0,
        1.5,
    ),
    (
        "binary_s_planet_moon",
        [1.0, 1.0, 1.0e-4, 1.0e-6],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 0.3, "e": 0.01, "i": 0.2, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            3: {"a": 0.003, "e": 0.01, "i": 0.1, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        13.0,
        0.02,
    ),
    (
        "binary_p_planet_moon",
        [1.0, 1.0, 1.0e-4, 1.0e-6],
        {
            0: None,
            1: {"a": 2.0, "e": 0.0, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 30.0, "e": 0.0, "i": 0.2, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            3: {"a": 0.23, "e": 0.0, "i": 0.1, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        803.0,
        2.0,
    ),
    (
        "binary_two_s_planets",
        [1.0, 1.0, 1.0e-4, 1.0e-4],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 0.3, "e": 0.01, "i": 0.2, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            3: {"a": 0.3, "e": 0.01, "i": 0.2, "lan": 0.3, "aop": 0.0, "ta": 0.0},
        },
        13.0,
        0.02,
    ),
    (
        "binary_two_s_planets_one_moon",
        [1.0, 1.0, 1.0e-4, 1.0e-4, 1.0e-6],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 0.3, "e": 0.01, "i": 0.2, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            3: {"a": 0.3, "e": 0.01, "i": 0.2, "lan": 0.3, "aop": 0.0, "ta": 0.0},
            4: {"a": 0.003, "e": 0.01, "i": 0.1, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        13.0,
        0.02,
    ),
    (
        "binary_two_s_planets_same_star",
        [1.0, 1.0, 1.0e-4, 1.0e-4],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 0.2, "e": 0.01, "i": 0.1, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            3: {"a": 0.4, "e": 0.01, "i": 0.15, "lan": 0.3, "aop": 0.0, "ta": 0.0},
        },
        13.0,
        0.02,
    ),
    (
        "binary_two_p_planets",
        [1.0, 1.0, 1.0e-4, 1.0e-4],
        {
            0: None,
            1: {"a": 2.0, "e": 0.0, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 25.0, "e": 0.0, "i": 0.2, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            3: {"a": 45.0, "e": 0.0, "i": 0.25, "lan": 0.3, "aop": 0.0, "ta": 0.0},
        },
        1350.0,
        3.0,
    ),
    (
        "triple_star_flat",
        [1.0, 1.0e-3, 1.0e-3],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 15.0, "e": 0.05, "i": 0.3, "lan": 0.2, "aop": 0.1, "ta": 0.3},
        },
        383.0,
        1.0,
    ),
    (
        "triple_star_chain",
        [1.0, 1.0, 1.0e-3],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 15.0, "e": 0.05, "i": 0.3, "lan": 0.2, "aop": 0.1, "ta": 0.3},
        },
        400.0,
        1.0,
    ),
    (
        "triple_star_s_planet",
        [1.0, 1.0, 1.0e-3, 1.0e-4],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 15.0, "e": 0.05, "i": 0.3, "lan": 0.2, "aop": 0.1, "ta": 0.3},
            3: {"a": 0.3, "e": 0.01, "i": 0.1, "lan": 0.4, "aop": 0.0, "ta": 0.0},
        },
        400.0,
        1.0,
    ),
    (
        "triple_star_s_planet_c",
        [1.0, 1.0, 1.0e-3, 1.0e-4],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 15.0, "e": 0.05, "i": 0.3, "lan": 0.2, "aop": 0.1, "ta": 0.3},
            3: {"a": 0.05, "e": 0.01, "i": 0.1, "lan": 0.4, "aop": 0.0, "ta": 0.0},
        },
        400.0,
        1.0,
    ),
    (
        "triple_star_s_planet_moon",
        [1.0, 1.0, 1.0e-3, 1.0e-4, 1.0e-6],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 15.0, "e": 0.05, "i": 0.3, "lan": 0.2, "aop": 0.1, "ta": 0.3},
            3: {"a": 0.3, "e": 0.01, "i": 0.1, "lan": 0.4, "aop": 0.0, "ta": 0.0},
            4: {"a": 0.003, "e": 0.01, "i": 0.05, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        400.0,
        1.0,
    ),
    (
        "triple_star_two_s_planets",
        [1.0, 1.0, 1.0e-3, 1.0e-4, 1.0e-4],
        {
            0: None,
            1: {"a": 2.0, "e": 0.05, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 15.0, "e": 0.05, "i": 0.3, "lan": 0.2, "aop": 0.1, "ta": 0.3},
            3: {"a": 0.3, "e": 0.01, "i": 0.1, "lan": 0.4, "aop": 0.0, "ta": 0.0},
            4: {"a": 0.3, "e": 0.01, "i": 0.1, "lan": 0.5, "aop": 0.2, "ta": 0.3},
        },
        400.0,
        1.0,
    ),
)

# Per-example overrides for the default solver settings below. Only needed
# for ``triple_star_s_planet_moon``: its moon's period is thousands of times
# shorter than the outer star's, so the default tolrel=1e-12/maxord=24 would
# take several minutes even for the modest tend=60.0 used above.
SOLVER_OVERRIDES = {
    "triple_star_s_planet_moon": {"tolrel": 1e-9, "tolabs": 1e-9, "maxord": 16},
    # Same disparity as triple_star_s_planet_moon: the planet's own orbit
    # around the light (1e-3) Star C is ~100x faster than Star C's orbit
    # around the AB binary.
    "triple_star_s_planet_c": {"tolrel": 1e-9, "tolabs": 1e-9, "maxord": 16},
    # Same disparity again: the moon's period around either planet is tens
    # of times shorter than the outer planet's period around the star.
    "star_two_planets_inner_moon": {"tolrel": 1e-9, "tolabs": 1e-9, "maxord": 16},
    "star_two_planets_outer_moon": {"tolrel": 1e-9, "tolabs": 1e-9, "maxord": 16},
}


def plot_examples(output_dir, animate=False, mode="std"):
    """
    Integrate every ``EXAMPLES`` entry and save its 3D trajectory plot
    (plus a GIF animation if ``animate``), applying any per-key
    ``SOLVER_OVERRIDES``. For templates with 3+ bodies, also saves an
    orbital-elements-vs-time plot of the smallest body (a lone moon if
    present, otherwise the outermost tied-lightest planet/star -- see
    ``select_smaller_body``) against its correct Jacobi reference frame
    (``reference_template_indices``); skipped for exactly 2 bodies, where
    a bare Kepler orbit has no secular/periodic variation to show.

    ``mode`` ("std" or "mpfr", see ``parse_precision_mode``) picks the
    *integration's* precision via ``is_mpfr`` -- the initial conditions
    built by ``HierarchicalSystemTemplates``/``HierarchicalSystem`` are
    always plain ``float64`` regardless of ``mode`` (see ``TidesSolver``'s
    ``is_mpfr``, the sole place precision is chosen).

    Returns the list of every saved file path.
    """
    output_paths = []
    for key, masses, elements, tend, dt in EXAMPLES:
        template = HierarchicalSystemTemplates.get(key)

        if template.n_bodies == 1:
            # A single static body has no dynamics and nothing worth
            # plotting -- see solve_hierarchy/solve_nbody's docstrings.
            print(f"{key}: single-body template, nothing to solve or plot")
            continue

        settings = precision_solver_settings(
            mode, std_tol=1e-12, mpfr_tol="1e-20", std_maxord=24, mpfr_maxord=32,
        )
        settings["minord"] = 6 if mode == "std" else 8
        settings.update(SOLVER_OVERRIDES.get(key, {}))

        t_hist, states, p_init, nodes, paths = solve_and_plot_hierarchy(
            key, key, masses, elements, tend, dt, template.title, output_dir,
            animate=animate, solver_settings=settings, is_mpfr=False,
        )
        for idx, path in enumerate(paths):
            label = "3D plot" if idx == 0 else "3D animation"
            print(f"{key}: {label} saved to {path}")
        output_paths.extend(paths)

        if template.n_bodies > 2 and output_dir is not None:
            # Skip for exactly 2 bodies: a plain two-body Kepler orbit has
            # constant osculating elements by definition (no perturber to
            # drive any secular/periodic variation), so the plot would show
            # nothing but floating-point-level noise around the nominal
            # values -- not informative. Skip entirely when output_dir is
            # None (the pytest fast path): plot_orbital_elements always
            # saves a figure, never a no-op.
            #
            # The smaller body's orbit: a lone moon if present (it's always
            # the lightest, unique body); otherwise the outermost of the
            # tied-lightest planets/stars -- see select_smaller_body's
            # docstring (utils/hierarchy_reference.py). select_smaller_body
            # works in *template* body-index space (matching masses/
            # elements/template.parents), which is NOT always the same as
            # the state-vector/``nodes`` order -- see state_index_by_name's
            # docstring for why (e.g. binary_two_s_planets).
            target_idx = select_smaller_body(template, masses, elements)
            ref_template_indices = reference_template_indices(template, elements, target_idx)
            target_state_idx = state_index_by_name(nodes, template.default_names[target_idx])
            reference_state_indices = [
                state_index_by_name(nodes, template.default_names[idx])
                for idx in ref_template_indices
            ]

            title = f"{template.default_names[target_idx]} orbital elements ({template.title})"
            elements_path = plot_orbital_elements(
                key, title, t_hist, states, p_init, target_state_idx, reference_state_indices, output_dir,
            )
            print(f"{key}: orbital elements plot saved to {elements_path}")
            output_paths.append(elements_path)
    return output_paths


def test_catalog_structure():
    """Structural checks only -- fast, no integration/plotting, so this runs
    under pytest with no ``output_dir`` involved."""
    keys = HierarchicalSystemTemplates.keys()
    assert len(keys) == len(set(keys)), "hierarchy template keys must be unique"
    assert "binary_s_planet_primary" in keys
    assert "binary_p_planet" in keys
    assert "binary_two_s_planets" in keys
    assert "binary_two_s_planets_same_star" in keys
    assert "binary_two_p_planets" in keys
    assert "triple_star_two_s_planets" in keys
    assert "triple_star_s_planet_c" in keys
    assert "star_two_planets" in keys
    assert "star_two_planets_inner_moon" in keys
    assert "star_two_planets_outer_moon" in keys
    assert "star_planet_moon_submoon" not in keys
    assert "star_four_planets" not in keys

    for template in HierarchicalSystemTemplates.all():
        assert HierarchicalSystemTemplates.validate(template)
        assert template.count_type("star") <= MAX_STARS
        assert template.count_type("planet") <= MAX_PLANETS
        assert template.count_type("moon") <= MAX_MOONS

        for body_idx in range(1, template.n_bodies):
            parent_idx = template.parents[body_idx]
            body_type = template.body_types[body_idx]
            parent_type = template.body_types[parent_idx]
            orbit_class = template.orbit_classes[body_idx]
            if body_type == "star":
                assert parent_type == "star"
            elif body_type == "planet":
                assert parent_type == "star"
                assert orbit_class in {"S", "P"}
            elif body_type == "moon":
                assert parent_type == "planet"
            else:
                raise AssertionError(f"unknown body type: {body_type}")

        masses = [1.0 if body_type == "star" else 1.0e-4 for body_type in template.body_types]
        elements = HierarchicalSystemTemplates.default_elements(template.key, masses=masses)
        system = HierarchicalSystemTemplates.build(template.key, masses, elements)
        v_init, p_init, nodes = system.generate()

        assert len(v_init) == 6 * template.n_bodies
        assert len(p_init) == 1 + template.n_bodies
        assert len(nodes) == template.n_bodies
        assert nodes[0].name == template.default_names[0]
        assert_close(p_init[1:], masses, 1e-15, f"{template.key} masses")


def test_catalog_examples(output_dir=None):
    """Dynamical check: integrates every ``EXAMPLES`` entry (see
    ``plot_examples``). ``output_dir=None`` (the pytest default) still runs
    every integration -- so a crash in any template's dynamics still fails
    the test -- but skips the plot/animation writes; passing a real
    directory (as ``main()`` does for standalone script execution) also
    saves them."""
    plot_examples(output_dir, animate=output_dir is not None)


def main():
    mode = parse_precision_mode(sys.argv, "Python/tests/test_hierarchy.py")
    if not configure_precision(mode):
        return

    test_catalog_structure()
    test_catalog_examples(PYTHON_DIR)


if __name__ == "__main__":
    main()
