# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `HierarchicalSystem._resolve()` computed each body's placement using
  `mu = G * (central_mass + child.sys_mass)`, where `child.sys_mass` was that
  body's own mass *plus every descendant added under it*, unconditionally.
  That's the right mass to use when a child's descendants are genuinely
  nested far tighter than the child's own orbit (a moon under a planet, a
  planet under a star) -- their mass is negligible relative to what they're
  chained under, so the approximation is harmless. It's wrong whenever a
  later-added descendant's own orbit is *not* much tighter than the level
  being resolved and its mass isn't negligible either: `child.pos`/`child.vel`
  are assigned as that body's own individual position/velocity, not a
  subtree barycenter, so the two-body `mu` used to place them must match --
  only `child.mass`, not its whole subtree.
  This was silent for every template except the five "nested triple star"
  ones (`triple_star_chain`, `triple_star_s_planet`, `triple_star_s_planet_c`,
  `triple_star_s_planet_moon`, `triple_star_two_s_planets`), which all chain
  Star C's parent to Star B: for a real hierarchical triple, Star C's own
  orbit is *wider* than the inner pair's, and its mass is typically
  comparable to (not negligible next to) the inner pair's combined mass --
  exactly the conditions that make the bug severe rather than negligible.
  Concretely, for `m1=1.1`, `m2=0.9`, `m3=1.0 M_sun` (all in solar masses),
  `a_in=10`, `e_in=0.3` au, `a_out=60`, `e_out=0.2` au: Star B's *actual*
  osculating orbit around Star A came out as `a=140 au, e=0.95` instead of
  the requested `(10, 0.3)` -- not a marginal error, a completely different,
  far wider and more eccentric ellipse. Every quantity derived from that
  initial condition (apparent "disintegration" of the inner binary over a
  few thousand years, at any `a_out`, independent of the outer star's mass
  scaling behavior) was consequently spurious. `Star C`'s own orbit (anchored
  via `_interior_reference`, unaffected by this bug) was already correct in
  every case checked.
  Root cause fixed in `HierarchicalSystem._resolve()` (`child.mass` instead
  of `child.sys_mass`); the now-unused `Node.sys_mass` field and its
  propagation logic in `add_body()` were removed, along with the two other
  call sites that mirrored the same (buggy) convention for consistency:
  `HierarchicalSystemTemplates._check_moon_hill_stability()`'s Hill-radius
  estimate and `default_elements()`'s automatic moon-semi-major-axis pick
  (both now use a sibling's own `mass`, not its subtree total, matching the
  fix). No template's `parents` topology needed to change -- verified with a
  from-scratch audit of all 22 catalog templates, comparing each body's
  actual osculating elements (computed independently via
  `cartesian_to_keplerian`, not by trusting the generator) against the
  elements it was asked for.
  The lopsided (`~1e-3`) inner-pair mass ratio used for the `triple_star_*`
  dynamical examples in `tests/test_hierarchy.py` predates this fix and is
  no longer required for correctness; kept as-is for now since it doesn't
  affect what those examples verify.

## [0.1.1] - 2026-07-27

### Fixed
- `HierarchicalSystemTemplates.solve_hierarchy`/`solve_nbody` (`physics="pn"`)
  built relativistic runs without ever checking the pairwise mass-ratio
  quality, so the `UserWarning` defined in `exotides.relativity.warn_pn_quality`
  never surfaced through the template API -- e.g. a comparable-mass binary
  star (`binary_star`) integrated silently despite the "dominant mass" 1PN
  approximation being a poor fit for it. Now wired in, and the warning
  additionally names the offending template and the actual body names used
  by that system (matching plotting legends) instead of a generic
  "this system"/"body N" fallback.

### Changed
- README: removed a stale reference to `experiments/` (deleted from the
  repo), documented the `physics="pn"` mass-ratio quality warning above,
  and corrected the pytest test count (19 -> 23).

## [0.1.0] - 2026-07-26

Initial feature set.

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
- **`HierarchicalSystem._resolve()` violated the barycentric constraint
  (`sum(m_i r_i) = 0`, `sum(m_i v_i) = 0`) for any hierarchy 3+ levels deep
  (a body whose parent is itself non-root -- e.g. a moon around a planet,
  or a third star chained onto one component of an inner binary).** The
  previous incremental "shift the tree after every insertion" scheme only
  correctly re-centered the immediate subtree being adjusted; whenever a
  body's interior reference folded in an ancestor beyond its immediate
  parent (the documented nested-triple/circumbinary case), the required
  correction needed to reach further up the tree than the shift actually
  applied, silently leaving a residual net momentum/offset. Found via an
  independent, from-scratch reconstruction of four representative
  hierarchies (star-planet-moon, S-type and P-type binary planets, a
  nested stellar triple) while preparing `docs/paper.tex`'s validation
  section -- the residual was as large as ~5 (order-unity, not a rounding
  artifact) for the affected templates
  (`star_planet_moon`, `star_two_planets_inner_moon`,
  `star_two_planets_outer_moon`, `binary_s_planet_secondary`,
  `binary_s_planet_moon`, `binary_p_planet_moon`,
  `binary_two_s_planets_one_moon`, `triple_star_chain` and everything
  built on top of that chain). Replaced with a single barycentering pass
  after the whole tree is resolved (see `orbital.py`'s `_resolve`/
  `generate` docstrings for why this is unconditionally correct rather
  than a narrower patch): every one of the 22 catalog templates now
  satisfies both constraints to machine precision
  (`experiments/exp3_hierarchy_checks.py`).
- `HierarchicalSystem.add_body` now validates mass positivity, `elements`
  completeness, semi-major-axis positivity, and duplicate body names,
  instead of failing later with unclear errors (or, in the duplicate-name
  case, silently corrupting the hierarchy tree).
- A missing comma in `exotides/__init__.py`'s `__all__` was silently
  concatenating two entries (`vector_norm`, `nbody_mincseries`) into one
  invalid string, excluding both names from the declared public API.
- `tests/test_events.py`'s two-planet instability collision test used a
  configuration whose collision time is itself chaos-sensitive over dozens
  of orbits; retuned to a configuration that collides within the first few
  orbits (verified stable across tolerances and both construction paths),
  so the test isn't fragile to unrelated, legitimate floating-point-level
  changes elsewhere in the package.
