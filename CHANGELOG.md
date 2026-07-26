# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Everything below is the project's initial feature set -- nothing has been
tagged/published yet (see docs/design-notes.md and the production-readiness
notes). This section becomes `## [0.1.0] - YYYY-MM-DD` at the first release.

### Added
- Taylor-series (TIDES) N-body integrator (`TidesSolver`), variable-order/
  variable-step, with double precision and arbitrary precision (via
  `gmpy2`) support.
- Newtonian N-body right-hand-side generator (`nbody_mincseries`), with an
  optional Numba-accelerated fast path (`exotides._fast_nbody`).
- `HierarchicalSystem` builder and Jacobi-ordering/interior-reference
  convention for assembling barycentric initial conditions from a tree of
  Keplerian elements.
- `HierarchicalSystemTemplates` catalog of 22 named hierarchical
  configurations (single/binary/triple stars, S-type/P-type planets,
  moons).
- Simplified pairwise 1PN relativistic correction (`exotides.relativity`),
  verified against the analytic GR apsidal-precession formula.
- Zero-crossing/collision event detection (`exotides.events`), ported from
  the original C TIDES library's event subsystem.
- Trajectory/orbital-element/Newtonian-vs-PN comparison plotting helpers
  (`exotides.plotting`).
- pytest-based verification suite (19 tests) covering Kepler/Lorenz
  closure regressions, every hierarchy template, real multi-star exoplanet
  systems, Kozai-Lidov cycling, 1PN precession, Numba correctness/speedup,
  and event detection -- each test also remains runnable directly
  (`python tests/test_foo.py`) to generate plots/animations.
- MIT `LICENSE`, packaging metadata (`authors`, `classifiers`, `keywords`,
  `readme`), and minimum dependency version pins in `pyproject.toml`.
- GitHub Actions CI (lint job + test matrix across OS/Python versions) and
  a `ruff`-based `pre-commit` hook.
- Coverage measurement (`pytest-cov`) wired into the test suite and CI
  (currently 69% of `exotides`; `plotting.py` reads low by design since
  the automated suite skips figure generation -- see `pyproject.toml`'s
  `[tool.coverage.report]`).

### Fixed
- `HierarchicalSystem.add_body` now validates mass positivity, `elements`
  completeness, semi-major-axis positivity, and duplicate body names,
  instead of failing later with unclear errors (or, in the duplicate-name
  case, silently corrupting the hierarchy tree).
- A missing comma in `exotides/__init__.py`'s `__all__` was silently
  concatenating two entries (`vector_norm`, `nbody_mincseries`) into one
  invalid string, excluding both names from the declared public API.
