# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Package description in `pyproject.toml`: "for small-N gravitational systems, with
  ready-made hierarchical templates up to five bodies", matching the package's scope.
- Documentation brought up to date with v2.0.0: README (default `"original"` step-size
  controller and the current comparison with `"pytides"`, precision handling, the
  hierarchy convention and what the S/P labels do, 1PN accuracy for comparable masses),
  `docs/design-notes.md`, module docstrings and test docstrings; references to scripts and
  paths that are not part of this repository were removed.
- All code comments are now in English. Several comments in `core.py`'s series-algebra
  routines had drifted one function down and were moved back to the routines they
  describe.

### Removed
- Unused `REPO_DIR` test helper, which pointed outside the repository.

### Fixed
- CI: the ruff version specifier is quoted, so `pip install "ruff>=0.16"` no longer
  redirects its output to a file named `=0.16`.

## [2.0.0] - 2026-11-04

### Added
- `TidesSolver(stepsize_controller=...)`, choosing between `"original"` (Abad et al. 2012,
  Algorithm 924) and `"pytides"`, also exposed through
  `HierarchicalSystemTemplates.solve_nbody`/`solve_hierarchy`/`make_nbody_solver` (first
  tagged as v1.0.0).
- `exotides.j2`: a continuous J2 (oblateness) term, as work in progress.
  `nbody_j2_mincseries` raises `NotImplementedError` until it has been validated.
- Tests `test_hierarchical_construction.py` and `test_conservation_integrals.py` (see the
  construction fix below) and `test_orbital_elements.py`.

### Changed
- **Breaking:** the default `stepsize_controller` is now `"original"` instead of
  `"pytides"`. `"pytides"` is equivalent to `"original"` at a looser, order-dependent
  effective tolerance `TOL^(q/(q+1))`; it is kept as an alternative, but it stopped being
  the default after producing a spurious instability in long integrations of hierarchical
  systems with a short inner period.
- Relicensed from MIT to GPL-3.0-or-later, matching the original TIDES (first tagged as
  v1.1.0).
- N-body and hierarchy setup information is printed with plain `print` instead of
  `logging`.

### Fixed
- `HierarchicalSystem` placed each body at its *own* position on its orbit,
  with `mu = G * (M_interior + m_body)`, even when other bodies were bound to
  it. That only works when those bodies (a moon, an S-type planet) have
  negligible mass. For a 2+2 quadruple with equal masses -- `A`-`B` and
  `C`-`D` at `a = 1`, `(A,B)`-`(C,D)` at `a = 100`, all circular -- `D` added
  around `C` gave the `C`-`D` barycenter half the inner orbital velocity and
  the outer orbit came out hyperbolic (`a = -5.75`, `e = 18.5`). The
  `_resolve()`/`_interior_reference()`/`_local_interior()` scheme and its
  final barycentering pass are replaced by an explicit binary tree of nested
  subsystems (`_build_subsystems()`): orbits are merged in ascending `a`, each
  joining the body's subsystem to its parent's, and `_place()` puts each
  subsystem's barycenter on its orbit with `mu = G * (M_inner + M_outer)`.
  This is exact for any hierarchy and barycentric by construction.
  **Convention change:** a body's `elements` now describe the orbit of its
  subsystem's barycenter (the body plus any descendants on tighter orbits),
  not of the body itself. The `add_body` API and the 22 catalog templates
  are unchanged. 13 templates give identical initial conditions (to
  round-off). The 9 with a nested massive companion (moons, planets around a
  secondary star) shift by `~1e-4` (positions) to `~2e-3` (velocities),
  relative, for planet/moon masses `1e-3`/`1e-5`. See
  `tests/test_hierarchical_construction.py`.
- `orbital.cartesian_to_keplerian()` silently returned `aop = 0.0` whenever the
  orbital plane coincided with the reference (x, y) plane (`i = 0` or `i =
  pi`), even when `e > 0` -- discarding the orbit's real in-plane periapsis
  orientation instead of folding it into `aop` under the standard `lan = 0`
  convention for this degenerate case. The forward direction
  (`keplerian_to_cartesian`) was never affected: with `i = lan = 0` it
  reduces to a single 2D rotation by `aop`, so building a system from a given
  `aop` always placed bodies correctly. Only the inverse (Cartesian-to-elements)
  conversion was broken, which meant `aop`/argument-of-periapsis time series
  extracted from a coplanar integration (e.g. to study apsidal precession)
  always came back identically zero, regardless of the actual dynamics. Fixed
  by computing `aop` directly from the eccentricity vector's in-plane angle
  when the ascending node is undefined, with a sign flip for the retrograde
  (`h_z < 0`) case; see `tests/test_orbital_elements.py`.
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
