"""
Python/tests/helpers/

Shared helpers for the tests/*.py scripts -- test-only support code, never
imported by the ``exotides`` package itself (see docs/design-notes.md), which
is why it lives under tests/ rather than as a top-level package of its own.
Import from here (``from helpers import ...``) rather than reaching into
individual submodules directly.

- precision            - std/mpfr parametrization helpers + PYTHON_DIR/REPO_DIR.
- systems              - generic HierarchicalSystem builder.
- orbital_analysis     - eccentricity-vector-angle / precession-rate helpers.
- hierarchy_reference  - "smaller body" + Jacobi reference-frame selection
                         for src/exotides/hierarchy.py templates.
- catalog_runner       - solve-one-config-and-plot loop body.
- event_helpers        - shared assertions for terminal-collision-event
                         tests (test_events.py).
"""

from .precision import (
    PYTHON_DIR,
    REPO_DIR,
    assert_close,
    parse_precision_mode,
    mpfr_available_or_skip,
    configure_precision,
    to_precision,
    vector_to_precision,
    solver_settings,
    kepler_mincseries,
    lorenz_mincseries,
)

from .systems import build_system, build_two_body_system

from .orbital_analysis import eccentricity_vector_angles, precession_rate

from .hierarchy_reference import (
    select_smaller_body,
    reference_template_indices,
    state_index_by_name,
)

from .catalog_runner import solve_and_plot_hierarchy

from .event_helpers import assert_single_terminal_collision, print_collision

__all__ = [
    "PYTHON_DIR",
    "REPO_DIR",
    "assert_close",
    "parse_precision_mode",
    "mpfr_available_or_skip",
    "configure_precision",
    "to_precision",
    "vector_to_precision",
    "solver_settings",
    "kepler_mincseries",
    "lorenz_mincseries",
    "build_system",
    "build_two_body_system",
    "eccentricity_vector_angles",
    "precession_rate",
    "select_smaller_body",
    "reference_template_indices",
    "state_index_by_name",
    "solve_and_plot_hierarchy",
    "assert_single_terminal_collision",
    "print_collision",
]
