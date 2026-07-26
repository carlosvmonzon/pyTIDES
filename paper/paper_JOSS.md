---
title: 'pyTIDES: A Python Framework for High-Order Taylor Integration of Hierarchical Planetary and Stellar Systems'
tags:
  - Python
  - astronomy
  - celestial mechanics
  - N-body simulation
  - Taylor series methods
  - orbital dynamics
authors:
  - name: Carlos Vázquez Monzón
    affiliation: 1
affiliations:
  - name: Universidad Loyola Andalucía, Dos Hermanas, Sevilla, Spain
    index: 1
date: 26 July 2026
bibliography: paper.bib
---

# Summary

`pyTIDES`, distributed as the Python package `exotides`, is a Python-native
framework for high-order Taylor-series integration of ordinary differential
equations, following the recurrence-based approach of the original TIDES
software [@Abad2012], together with a construction layer purpose-built for
hierarchical planetary and stellar systems. Given a system described as a
tree of named bodies and Keplerian orbital elements, `exotides` determines
the correct interior dynamical reference for every orbit (a Jacobi-like
convention generalized to arbitrary nesting) and assembles a dynamically
consistent, barycentric Cartesian initial condition. The underlying Taylor
solver and Newtonian force generator are themselves generic and not
restricted to gravitational or hierarchical problems -- the same solver
integrates the non-gravitational Lorenz system unchanged in the package's
test suite -- while the hierarchy builder, a catalog of 22 named
architectures (single/binary/triple-star systems with circumstellar,
circumbinary, and moon configurations), dense-output event detection, an
optional pairwise first post-Newtonian correction, and an optional
Numba-accelerated force path are specialized for the astrophysical use case.
Both standard double-precision and arbitrary-precision (`gmpy2`) arithmetic
are supported throughout.

# Statement of need

Existing high-order Taylor-integration software for astrodynamics is
centered on compiled languages. The original TIDES ecosystem generates C or
Fortran recurrence code from a symbolic problem description [@Abad2012], and
the more recent `heyoka`/`heyoka.py` compiles a problem-specific LLVM
computational graph for substantially higher throughput
[@Biscani2021; @Biscani2022]. Both deliver excellent runtime performance but
require, respectively, a code-generation step or a C++/CMake toolchain and a
fresh compilation whenever the system changes, and neither ships tooling for
constructing a dynamically consistent multi-body initial condition from a
nested hierarchy: assembling a Jacobi-ordered state for a circumbinary
planet or a nested stellar triple must be written by hand from single-orbit
conversion routines. General-purpose $N$-body packages such as REBOUND
[@Rein2012] face the same gap in their orbital-element interface and are
fixed at double precision regardless of how the integration is tuned.
Secular, orbit-averaged codes such as `kozai` [@Blaes2002] solve a related
but different problem: they evolve only slowly-changing orbital elements,
which is efficient for long-term eccentricity/inclination cycling but
discards the fast orbital phase needed to localize a close encounter or
collision, and cannot address genuinely non-hierarchical or
comparable-mass configurations.

`pyTIDES` targets the resulting gap: pure-Python, hierarchy-aware,
optionally arbitrary-precision integration of small hierarchical systems
(a few to tens of bodies), with an explicit, validated interior-reference
construction rule and a catalog of pre-built named architectures. It is
aimed at researchers and students exploring the orbital dynamics of nested
planetary or stellar systems directly from Keplerian elements, in readable
Python, without a compiled toolchain, and with access to precision beyond
the double-precision floor when a problem demands it -- for example,
long-term or chaos-sensitive integrations where floating-point round-off,
not the integrator, is the limiting factor. `pyTIDES` is explicitly not
positioned as a replacement for REBOUND or `heyoka` in large-scale,
long-duration, or maximum-throughput settings; it is intended to
complement them for this narrower niche.

# Functionality

The package separates a generic numerical core from an astrophysical
convenience layer:

- A Taylor recurrence algebra (elementary series operations for
  multiplication, division, powers, and transcendental functions) and a
  generic `TidesSolver` with adaptive order/step selection and optional
  endpoint-defect control.
- A direct Newtonian $N$-body coefficient generator, with an optional
  Numba-accelerated evaluation path.
- A `HierarchicalSystem` builder that converts a tree of Keplerian elements
  into a barycentric Cartesian state via an interior-reference convention.
  This construction was independently re-derived from scratch and checked
  against the package's own output across all 22 catalog templates as part
  of preparing this software for release; the exercise uncovered, and this
  release fixes, a genuine defect in barycentric-constraint satisfaction
  for hierarchies nested three levels deep or more, now verified to
  machine precision across the full catalog.
- An optional simplified pairwise first post-Newtonian correction.
- Dense-output event detection, using Horner evaluation of the same Taylor
  polynomial already produced by each accepted step, supporting both
  terminal (e.g. collision) and non-terminal (e.g. node-crossing) events.
- Arbitrary-precision arithmetic via `gmpy2`, available as a drop-in
  alternative to double precision throughout the same recurrence code.

As one illustration of what arbitrary precision enables, \autoref{fig:kepler}
shows the state-closure error of an integrated Kepler orbit after one
orbital period, as a function of requested tolerance. In double precision
every eccentricity tested floors at the same value REBOUND's IAS15
integrator reaches on the identical problem ($\sim 10^{-14}$), a limit set
by float64 round-off rather than by either integrator's algorithm; building
the same initial condition natively in 200-bit `gmpy2` arithmetic and
integrating at matching tolerance continues converging linearly for a
further 15.5 orders of magnitude past that floor.

![State-closure error after one Kepler orbital period vs. requested
tolerance, in double precision (four eccentricities) and in 200-bit
arbitrary precision, with REBOUND's IAS15 double-precision floor marked
for reference.\label{fig:kepler}](../figures/paper_figures/kepler_convergence.png)

A companion methods paper describes the Taylor-series formulation, the
hierarchy-construction algorithm, and an extended numerical validation and
performance study -- including the defect described above, a detailed
comparison with REBOUND, `heyoka`, and `kozai`, and the reasoning behind
each validation experiment -- in more depth than is appropriate here. Every
number and figure quoted in either paper is produced by a versioned,
independently rerunnable script under `experiments/` in the source
repository, and the package's `pytest` suite is exercised by continuous
integration on every push.

# Acknowledgements

<!-- TODO: funding, institutional support, software acknowledgements. -->

# References
