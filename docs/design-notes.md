# Design notes: lineage, comparisons, and honest limitations

This document collects the context that doesn't fit in the README: where pyTIDES sits
relative to the literature and to other N-body/Taylor-integration codes, what's genuinely
novel about it versus what's a straightforward reimplementation, and where the "not
general-purpose" framing in the README needs a more precise reading of the actual code.

---

## 1. Lineage: the TIDES method itself

The Taylor-series method implemented in `src/exotides/core.py` (`mul_mc`, `div_mc`, `pow_mc_c`,
`exp_mc`, `log_mc`, `sin_mc`, `cos_mc`, and the `TidesSolver` variable-order/variable-step
driver) is not original to this package. It reimplements, in pure Python, the algorithm
described in:

> A. Abad, R. Barrio, F. Blesa, M. Rodríguez, "Algorithm 924: TIDES, a Taylor Series
> Integrator for Differential EquationS", *ACM Transactions on Mathematical Software*,
> 39(1), 2012.

The original TIDES (GME group, Universidad de Zaragoza) ships as a **C/Fortran library**
(`libTIDES`) plus a **Mathematica** preprocessor
(`MathTIDES`) that generates the C/Fortran code. There is no official Python
implementation or binding — checked directly against the project's SourceForge page
(gme.unizar.es/software/tides), which lists only those two interfaces. The closest
adjacent project by lineage, **Omnisode/sode** (Ruby, generates C/C++/Ruby/Maxima/Maple
from `.ode` files, tracing back to Y. F. Chang and George Corliss's Taylor-series work of
the 1970s–80s, predating TIDES), attempted Python support around 2014 and abandoned it.

**So: `pyTIDES` is a genuine from-scratch Python port of the method, not a wrapper around
`libTIDES`.** That's worth stating plainly because it's easy to assume a "Python bindings for
an existing C library" story where none exists.

The installable package itself is named `exotides` (`import exotides`), not `pytides` --
`pytides` is already a real, unrelated PyPI package for ocean tidal-harmonic analysis and
prediction (sam-cox/pytides), so using it here would collide with an existing project despite
the different domain. `exotides` keeps the project's own name, **pyTIDES**, as the
human-facing name (this repo, the README title) while the Python import name specifically
signals what the package is *for* -- hierarchical **exo**planet systems -- rather than
reusing the already-taken generic name.

---

## 2. Comparison to heyoka (Biscani & Izzo)

[heyoka](https://github.com/bluescarni/heyoka) / `heyoka.py`, introduced in Biscani, F., &
Izzo, D. (2021), "Revisiting high-order Taylor methods for astrodynamics and celestial
mechanics," *Monthly Notices of the Royal Astronomical Society*, 504(2), 2614-2628, is the
closest thing to a modern equivalent, and the similarities go deeper than "also Taylor
series":

- **Event detection is conceptually the same mechanism.** heyoka's terminal/non-terminal
  event distinction and its root-finding on the Taylor coefficients already computed for
  the current step (via dense output) -- described in detail in Biscani, F., & Izzo, D.
  (2022), "Reliable event detection for Taylor methods in astrodynamics," *MNRAS*, 513(4),
  4833-4844 -- is the same idea as `src/exotides/events.py` + `TidesSolver.solve`'s
  `find_event_crossings` (`src/exotides/core.py:711`), which itself ports the original C
  library's `dp_tides_find_zeros`.
- **Arbitrary precision**: heyoka uses `mp++`'s `mppp::real`; pyTIDES uses `gmpy2.mpfr`.
  Same goal (runtime-configurable precision), different backend library.
- **Performance**: heyoka JIT-compiles a *symbolic, problem-specific* computational graph
  via LLVM — every product/quotient/power in the Taylor recurrence for *that exact system*
  gets fused and optimized as one specialized function, plus a SIMD "batch mode" that
  integrates many instances of the same system at once. pyTIDES's Numba path
  (`src/exotides/_fast_nbody.py`) JIT-compiles a *generic* N-body loop (works for any N, any
  masses) — real speedup over the pure-Python core, but not the same class of
  specialization. heyoka should be expected to win on raw throughput.
- **Orbital elements are *not* as convenient in the heyoka ecosystem.** heyoka itself only
  ships `model.nbody()` (symbolic Cartesian N-body equations) and `kepE()` (Kepler-equation
  inversion, a building block). Keplerian ↔ Cartesian conversion lives in a *separate*
  sibling package by the same authors, **pykep** (`par2ic`/`ic2par`), and neither package
  has anything resembling `exotides.orbital.HierarchicalSystem`: assembling a barycentric,
  Jacobi-ordered multi-body state from a tree of orbital elements would have to be
  hand-written on top of pykep's single-orbit conversion. In pyTIDES it's
  `sys.add_body(...)` + `sys.generate()`.
- **No semantic hierarchy catalog.** heyoka is a general-purpose ODE/astrodynamics
  integrator; it has no analogue of `HierarchicalSystemTemplates` (named S-type/P-type
  configurations, Jacobi-ordering baked in, hierarchy diagrams).

Net: heyoka is the closest relative *at the integrator/event-detection level*, and is very
likely faster and more mature there. What it doesn't have is the hierarchy-template layer,
which remains the actual differentiator of this package.

---

## 3. Comparison to REBOUND and `kozai`

### REBOUND

[REBOUND](https://github.com/hannorein/rebound) (Rein, H., & Liu, S.-F. (2012), "REBOUND: an
open-source multi-purpose N-body code for collisional dynamics," *Astronomy & Astrophysics*,
537, A128) is a mature, widely-used, **general-purpose** C-based N-body integrator: it handles
arbitrary particle counts and configurations (star clusters, planetesimal disks, arbitrary
hierarchies or none at all), with many integrators (WHFast, IAS15, ...) and an extensive
physics-extension ecosystem (REBOUNDx). It is faster and far more battle-tested than pyTIDES
for essentially everything it does, and it also installs on Windows via `pip install rebound`
(the pure-Python interface; using the C interface directly needs the MSVC compiler).

pyTIDES does not compete with that breadth -- it solves a much narrower problem (a hierarchical
star/planet/moon system, specifically) and leans entirely into that narrowness: a semantic
template catalog, a Jacobi-ordering convention baked into every computation, and a plotting API
that already knows what a moon's parent or a circumbinary planet's reference frame should be.
Within that narrower scope, a few tradeoffs can still matter:

- **Arbitrary precision.** REBOUND's integrators are fixed double precision. pyTIDES can
  integrate at 128-bit (or higher) precision via `gmpy2`, useful for chaos-sensitive or very
  long-term integrations where double-precision roundoff matters.
- **No compiled extension to build for the Python-only path.** The core solver, the
  hierarchy builder, the 1PN correction and even the Numba-accelerated fast path are all
  pure Python/JIT -- no C compiler needed at all to install or to read/modify the physics.
- **A semantic hierarchy-template catalog built in.** `HierarchicalSystemTemplates` ships
  22 named star/planet/moon configurations (S-type, P-type, nested triples) with correct
  Jacobi-ordering baked in, rather than requiring every configuration to be hand-assembled
  from individual particles and orbital elements.
- **A Taylor-series (automatic-differentiation-like) integration method**, rather than a
  symplectic or embedded Runge-Kutta scheme -- a genuinely different numerical approach that
  reaches high accuracy by increasing the series order rather than switching integrators.

On the other hand, REBOUND remains the better choice for large particle counts, long-term
integration campaigns, and the breadth of physics already implemented and peer-reviewed in
REBOUNDx (including more complete general-relativistic treatments than pyTIDES's simplified
pairwise 1PN correction). Collision *detection* exists in both (see README Features Guide §6),
but REBOUND additionally *resolves* collisions mid-simulation (merge, bounce) at scale via
tree-based algorithms; pyTIDES instead stops exactly at the contact event, which fits its fixed
hierarchy-template topology better than merging bodies into a different tree shape would.

### `kozai`

**`kozai`** (joe-antognini/kozai) is the closest *domain* match — hierarchical triples
with 1PN precession and 2.5PN gravitational-radiation terms (Blaes et al. 2002) — but it
integrates the **secularly-averaged** (orbit-averaged) equations, not full Cartesian
N-body dynamics. It answers a different question (long-term eccentricity/inclination
evolution cheaply) than pyTIDES (short-to-medium-term full trajectories).

---

## 4. What "not general-purpose" actually means in this codebase

The README states the package "doesn't aim to integrate arbitrary particle counts or
non-hierarchical configurations." Taken literally, that's stronger than what's actually
true of the code, and worth being precise about:

**The integrator itself has no such restriction.** `TidesSolver` (`src/exotides/core.py`) is a
generic variable-order/variable-step Taylor ODE solver with no notion of "N-body" or
"hierarchy" at all — the proof is that `tests/test_lorenz.py` hands it the Lorenz system
(not gravitational, not N-body) and it works unchanged. Likewise,
`_nbody_mincseries_core`/`nbody_mincseries` (`src/exotides/nbody.py`) computes pairwise Newtonian
gravity for a fully generic `N = len(v) // 6` with no upper bound — the double loop over
`j, k in range(N)` doesn't know or care whether the system is hierarchical.

**What *is* scoped to hierarchies, specifically:**

- `HierarchicalSystemTemplates`'s catalog validation (`src/exotides/hierarchy.py:35-37`,
  `MAX_STARS = 3`, `MAX_PLANETS = 2`, `MAX_MOONS = 1`) is a self-imposed limit on what the
  *named template catalog* documents and draws — not a limit on the solver. Nothing stops
  a caller from driving `TidesSolver` + `nbody_mincseries` directly with a hand-built state
  vector of, say, 50 bodies; it would integrate the pairwise-Newtonian dynamics correctly.
  `tests/test_fast_nbody.py`'s `build_ring_system` does exactly this — it builds systems up
  to 16 bodies via `HierarchicalSystem` directly (not a catalog template) purely to measure
  how the Numba speedup scales with `N`, which only works because nothing below the catalog
  layer objects to going past 5 bodies.
- `HierarchicalSystem._resolve()`/`_interior_reference()` (`src/exotides/orbital.py`) — the
  convenience layer that turns a tree of Keplerian elements into barycentric initial
  conditions — assumes a well-defined "what's interior to this body" partial order. A star
  cluster or planetesimal disk has no such natural order, so this layer specifically
  (not the integrator) doesn't generalize to those cases without being bypassed entirely.
- **Collisions are always terminal by design** (README, Features Guide §6): appropriate
  when the topology is a fixed hierarchy tree that a merge would invalidate, but exactly
  wrong for star-cluster-scale dynamics, where mergers are routine, not exceptional.
- **No algorithmic scaling for large N**: the pairwise loops are O(N²) per Taylor order per
  step, in Python/Numba, with no tree-code approximation (Barnes-Hut), no softening, and no
  close-encounter regularization. This is a real practical ceiling on N even though it's
  not a hard-coded one.

So the accurate statement is: **the Taylor-series N-body engine is generic; the template
catalog, the Jacobi-ordering/interior-reference convention, the always-terminal collision
model, and the absence of large-N performance techniques are what's specifically built for
—and scoped to— hierarchies.** The README has been reworded accordingly.

---

## 5. Future work: secular equations and tidal forces

Two omissions from the present physical model are natural directions for future work rather
than fundamental obstacles, precisely because `TidesSolver` (§4 above) has no built-in notion
of what equations of motion it advances -- `tests/test_lorenz.py` already proves this by
integrating the non-gravitational Lorenz system unchanged.

**Secularly-averaged (orbit-averaged) equations for hierarchical triples**, of the kind
`kozai` implements (§3), could in principle be supplied as an alternative right-hand side to
the same adaptive-order, arbitrary-precision propagator -- letting the fast-orbital-phase-free
regime benefit from high-order Taylor propagation and arbitrary precision the same way the
direct N-body regime already does. This is a substantial addition rather than a new force
term, however: the secular state (eccentricity/angular-momentum vectors or orbital elements
evolved directly) is not the Cartesian per-body layout `pack_state`/`unpack_state` and
`HierarchicalSystem` assume throughout the present package, so it would need its own
initial-condition builder and diagnostic tools rather than a drop-in coefficient generator.

**Dissipative tidal forces** (e.g. equilibrium-tide or constant-time-lag models) are absent
from the present physical model, even though they are the standard mechanism invoked to
arrest the same Kozai-Lidov eccentricity excursions that this package's own Kozai-Lidov test
problem (`tests/test_kozai_nbody.py`) drives all the way to a terminal collision -- gravity
alone, with no tidal dissipation, carries that configuration to contact. Adding such a term
would follow the same pattern already used for the 1PN correction
(`src/exotides/relativity.py`): an additional pairwise acceleration term in the same
Newtonian coefficient generator, and would not require the architectural change the
secular-equations extension above does.
