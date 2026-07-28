"""
exotides/hierarchy.py

Reusable hierarchy templates for small astrophysical N-body systems.

The catalog is semantic: templates are named as systems a user may want to
simulate.  Planets in binary-star systems are marked as:

    S - circumstellar, orbiting one stellar component
    P - circumbinary, orbiting the binary barycenter

Copyright (C) 2026  Carlos Vazquez Monzon

This file is part of pyTIDES (the ``exotides`` package).

pyTIDES is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version.

pyTIDES is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along
with pyTIDES. If not, see <https://www.gnu.org/licenses/>.
"""

from dataclasses import dataclass
import functools
import warnings

import numpy as np

from .core import TidesSolver
from .nbody import nbody_mincseries, nbody_pn_mincseries
from .orbital import HierarchicalSystem
from .relativity import append_pn_params, warn_pn_quality

# Physical speed of light in the same units convention as G (see
# HierarchicalSystemTemplates.make_nbody_solver's ``c`` kwarg): with G=1 and
# lengths/masses in the caller's own units, ``c`` must be supplied in
# matching units -- there is no universal default, so callers of
# physics="pn" must pass one explicitly.
_PHYSICS_GENERATORS = {
    "newtonian": nbody_mincseries,
    "pn": nbody_pn_mincseries,
}


MAX_STARS = 3
MAX_PLANETS = 2
MAX_MOONS = 1

# A moon further out than this fraction of its parent planet's Hill radius is
# not reliably bound against the star's tidal perturbation (prograde orbits
# become unstable above ~0.4895 R_Hill; Domingos, Winter & Yokoyama 2006).
MOON_HILL_FRACTION = 0.4
# Fraction used by ``default_elements`` to leave comfortable margin under the
# hard limit above.
_DEFAULT_MOON_HILL_FRACTION = 0.2


@dataclass(frozen=True)
class HierarchyTemplate:
    """Semantic rooted hierarchy template."""

    key: str
    parents: tuple
    body_types: tuple
    orbit_classes: tuple
    title: str
    description: str
    default_names: tuple

    @property
    def n_bodies(self):
        """Total number of bodies in the template (root included)."""
        return len(self.parents)

    def count_type(self, body_type):
        """Number of bodies of ``body_type`` (``"star"``, ``"planet"``, or ``"moon"``)."""
        return self.body_types.count(body_type)

    def parent_name(self, body_idx, names):
        """
        ``names[parent_idx]`` for body ``body_idx`` (``None`` for the root),
        given a caller-supplied ``names`` sequence -- e.g. to look up a
        custom body name instead of ``default_names``.
        """
        parent_idx = self.parents[body_idx]
        return None if parent_idx is None else names[parent_idx]


def _tpl(key, parents, body_types, orbit_classes, title, description, names):
    return HierarchyTemplate(
        key,
        tuple(parents),
        tuple(body_types),
        tuple(orbit_classes),
        title,
        description,
        tuple(names),
    )


class HierarchicalSystemTemplates:
    """
    Catalog of user-facing hierarchy templates up to five bodies.

    Rules enforced by the catalog:

        - at most 3 stars
        - at most 2 planets, in any combination of S/P orbit classes and
          parents (two S-type planets sharing one star, two S-type planets
          on different stars, or two P-type circumbinary planets) --
          planet-planet interaction studies drive several catalog entries
        - at most 1 moon
        - stars orbit stars
        - S-type planets orbit one star
        - P-type planets orbit a binary-star subsystem
        - moons orbit planets
    """

    TEMPLATES = (
        _tpl(
            "single_star",
            [None],
            ["star"],
            [None],
            "Single star",
            "One isolated central star.",
            ["Star"],
        ),
        _tpl(
            "star_planet",
            [None, 0],
            ["star", "planet"],
            [None, "S"],
            "Star with one planet",
            "A planet orbiting a single star.",
            ["Star", "Planet"],
        ),
        _tpl(
            "star_planet_moon",
            [None, 0, 1],
            ["star", "planet", "moon"],
            [None, "S", None],
            "Star, planet and moon",
            "A planet orbits the star and one moon orbits the planet.",
            ["Star", "Planet", "Moon"],
        ),
        _tpl(
            "star_two_planets",
            [None, 0, 0],
            ["star", "planet", "planet"],
            [None, "S", "S"],
            "Star with two planets",
            "Two planets orbit a single star at different separations, for "
            "studying planet-planet gravitational interactions without a "
            "stellar companion.",
            ["Star", "Inner planet", "Outer planet"],
        ),
        _tpl(
            "star_two_planets_inner_moon",
            [None, 0, 0, 1],
            ["star", "planet", "planet", "moon"],
            [None, "S", "S", None],
            "Star with two planets, moon on the inner planet",
            "Two planets orbit a single star; a moon orbits the inner "
            "planet.",
            ["Star", "Inner planet", "Outer planet", "Moon"],
        ),
        _tpl(
            "star_two_planets_outer_moon",
            [None, 0, 0, 2],
            ["star", "planet", "planet", "moon"],
            [None, "S", "S", None],
            "Star with two planets, moon on the outer planet",
            "Two planets orbit a single star; a moon orbits the outer "
            "planet.",
            ["Star", "Inner planet", "Outer planet", "Moon"],
        ),
        _tpl(
            "binary_star",
            [None, 0],
            ["star", "star"],
            [None, None],
            "Binary star",
            "Two stars in a hierarchical binary.",
            ["Star A", "Star B"],
        ),
        _tpl(
            "binary_s_planet_primary",
            [None, 0, 0],
            ["star", "star", "planet"],
            [None, None, "S"],
            "Binary star with S-type planet around A",
            "A planet orbits one component of a binary star.",
            ["Star A", "Star B", "Planet A"],
        ),
        _tpl(
            "binary_s_planet_secondary",
            [None, 0, 1],
            ["star", "star", "planet"],
            [None, None, "S"],
            "Binary star with S-type planet around B",
            "A planet orbits the secondary component of a binary star.",
            ["Star A", "Star B", "Planet B"],
        ),
        _tpl(
            "binary_p_planet",
            [None, 0, 0],
            ["star", "star", "planet"],
            [None, None, "P"],
            "Binary star with P-type planet",
            "A circumbinary planet orbits the binary barycenter.",
            ["Star A", "Star B", "Circumbinary planet"],
        ),
        _tpl(
            "binary_s_planet_moon",
            [None, 0, 0, 2],
            ["star", "star", "planet", "moon"],
            [None, None, "S", None],
            "Binary star, S-type planet and moon",
            "An S-type planet in a binary star has one moon.",
            ["Star A", "Star B", "Planet", "Moon"],
        ),
        _tpl(
            "binary_p_planet_moon",
            [None, 0, 0, 2],
            ["star", "star", "planet", "moon"],
            [None, None, "P", None],
            "Binary star, P-type planet and moon",
            "A circumbinary planet has one moon.",
            ["Star A", "Star B", "Circumbinary planet", "Moon"],
        ),
        _tpl(
            "binary_two_s_planets",
            [None, 0, 0, 1],
            ["star", "star", "planet", "planet"],
            [None, None, "S", "S"],
            "Binary star with two S-type planets",
            "Each stellar component has one S-type planet.",
            ["Star A", "Star B", "Planet A", "Planet B"],
        ),
        _tpl(
            "binary_two_s_planets_one_moon",
            [None, 0, 0, 1, 2],
            ["star", "star", "planet", "planet", "moon"],
            [None, None, "S", "S", None],
            "Binary star, two S-type planets and one moon",
            "Each star has one S-type planet; one planet has a moon.",
            ["Star A", "Star B", "Planet A", "Planet B", "Moon"],
        ),
        _tpl(
            "binary_two_s_planets_same_star",
            [None, 0, 0, 0],
            ["star", "star", "planet", "planet"],
            [None, None, "S", "S"],
            "Binary star with two S-type planets around the same star",
            "Both planets orbit component A at different separations, for "
            "studying planet-planet gravitational interactions within one "
            "star's Hill sphere.",
            ["Star A", "Star B", "Inner planet", "Outer planet"],
        ),
        _tpl(
            "binary_two_p_planets",
            [None, 0, 0, 0],
            ["star", "star", "planet", "planet"],
            [None, None, "P", "P"],
            "Binary star with two P-type planets",
            "Two circumbinary planets orbit the binary barycenter at "
            "different distances, for studying planet-planet gravitational "
            "interactions between P-type orbits.",
            ["Star A", "Star B", "Inner circumbinary planet", "Outer circumbinary planet"],
        ),
        _tpl(
            "triple_star_flat",
            [None, 0, 0],
            ["star", "star", "star"],
            [None, None, None],
            "Triple star, wide companions",
            "Two stellar companions orbit the primary star.",
            ["Star A", "Star B", "Star C"],
        ),
        _tpl(
            "triple_star_chain",
            [None, 0, 1],
            ["star", "star", "star"],
            [None, None, None],
            "Nested triple star",
            "A close stellar binary is orbited hierarchically by a third star.",
            ["Star A", "Star B", "Star C"],
        ),
        _tpl(
            "triple_star_s_planet",
            [None, 0, 1, 0],
            ["star", "star", "star", "planet"],
            [None, None, None, "S"],
            "Triple star with S-type planet",
            "A nested triple-star system with one planet orbiting one component.",
            ["Star A", "Star B", "Star C", "Planet"],
        ),
        _tpl(
            "triple_star_s_planet_c",
            [None, 0, 1, 2],
            ["star", "star", "star", "planet"],
            [None, None, None, "S"],
            "Triple star with S-type planet around C",
            "A nested triple-star system with one planet orbiting the outer "
            "stellar component, Star C.",
            ["Star A", "Star B", "Star C", "Planet"],
        ),
        _tpl(
            "triple_star_s_planet_moon",
            [None, 0, 1, 0, 3],
            ["star", "star", "star", "planet", "moon"],
            [None, None, None, "S", None],
            "Triple star, S-type planet and moon",
            "A planet orbiting one stellar component has one moon.",
            ["Star A", "Star B", "Star C", "Planet", "Moon"],
        ),
        _tpl(
            "triple_star_two_s_planets",
            [None, 0, 1, 0, 1],
            ["star", "star", "star", "planet", "planet"],
            [None, None, None, "S", "S"],
            "Triple star with two S-type planets",
            "Two planets orbit two different stellar components.",
            ["Star A", "Star B", "Star C", "Planet A", "Planet B"],
        ),
    )

    _BY_KEY = {template.key: template for template in TEMPLATES}

    @classmethod
    def all(cls):
        """Return all semantic templates."""
        return cls.TEMPLATES

    @classmethod
    def by_size(cls, n_bodies):
        """Return templates with exactly ``n_bodies`` bodies."""
        return tuple(template for template in cls.TEMPLATES if template.n_bodies == n_bodies)

    @classmethod
    def keys(cls, n_bodies=None):
        """Return user-facing template keys, optionally filtered by body count."""
        templates = cls.TEMPLATES if n_bodies is None else cls.by_size(n_bodies)
        return tuple(template.key for template in templates)

    @classmethod
    def choices(cls, n_bodies=None):
        """Return scenario summaries for menus or parametrized tests."""
        templates = cls.TEMPLATES if n_bodies is None else cls.by_size(n_bodies)
        return tuple(
            {
                "key": template.key,
                "title": template.title,
                "description": template.description,
                "n_bodies": template.n_bodies,
                "body_types": template.body_types,
                "orbit_classes": template.orbit_classes,
            }
            for template in templates
        )

    @classmethod
    def get(cls, key):
        """Return one template by key."""
        try:
            return cls._BY_KEY[key]
        except KeyError as exc:
            raise ValueError(f"Unknown hierarchy template: {key}") from exc

    @classmethod
    def build(cls, key, masses, elements, G=1.0, names=None):
        """
        Build a ``HierarchicalSystem`` from a semantic template.

        ``elements`` may be a sequence with one entry per body, where
        ``elements[0]`` is ignored, or a dict keyed by body index.
        """
        template = cls.get(key)
        cls.validate(template)

        n_bodies = template.n_bodies
        if len(masses) != n_bodies:
            raise ValueError(f"{key} requires {n_bodies} masses, got {len(masses)}")

        if names is None:
            names = template.default_names
        elif len(names) != n_bodies:
            raise ValueError(f"{key} requires {n_bodies} names, got {len(names)}")

        system = HierarchicalSystem(G=G)
        for body_idx in range(n_bodies):
            body_elements = None if body_idx == 0 else cls._body_elements(elements, body_idx)
            if template.body_types[body_idx] == "moon":
                cls._check_moon_hill_stability(system, template, body_idx, names, body_elements)
            system.add_body(
                names[body_idx],
                mass=masses[body_idx],
                parent_name=template.parent_name(body_idx, names),
                elements=body_elements,
            )
        return system

    @classmethod
    def initial_conditions(cls, key, masses, elements=None, G=1.0, names=None):
        """
        Build the hierarchy and return the N-body initial condition vectors.

        Returns
        -------
        v_init : ndarray, shape (6*N,)
            Cartesian state packed as ``[x,y,z,vx,vy,vz]`` per body.
        p_init : ndarray, shape (1+N,)
            N-body parameter vector ``[G, m0, m1, ...]``.
        nodes : list
            Ordered hierarchy nodes matching the state-vector order.
        """
        if elements is None:
            elements = cls.default_elements(key, masses=masses)
        system = cls.build(key, masses, elements, G=G, names=names)
        return system.generate()

    @classmethod
    def make_nbody_solver(
        cls,
        key,
        *,
        physics="newtonian",
        use_numba=None,
        tolrel=1e-12,
        tolabs=1e-12,
        maxord=32,
        minord=8,
        nordinc=5,
        defect_error_control=False,
        stepsize_controller="pytides",
    ):
        """
        Create a ``TidesSolver`` configured for the selected hierarchy.

        ``physics`` selects the Taylor-series generator: ``"newtonian"``
        (default, ``nbody_mincseries``) or ``"pn"`` (``nbody_pn_mincseries``,
        the pairwise "dominant mass" 1PN correction -- see
        ``exotides/relativity.py``). ``"pn"`` requires one extra trailing
        parameter (the speed of light) appended to ``p_init``; see
        ``solve_nbody``'s ``speed_of_light`` argument for the convenience
        wrapper, or append it yourself via
        ``exotides.relativity.append_pn_params`` when calling this solver
        directly. The hierarchy only defines the initial barycentric state
        and the mass parameter vector -- it's the same for every physics
        choice.

        ``use_numba`` is passed straight through to ``nbody_mincseries`` (see
        its docstring): ``None`` auto-detects (the default), ``True``/``False``
        force the Numba or pure-Python core respectively. Only meaningful for
        ``physics="newtonian"``; the 1PN generator has no Numba path, so
        passing it with ``physics="pn"`` raises.

        This returns a plain ``TidesSolver`` -- there's no ``is_mpfr``
        parameter here (or on ``solve_nbody``): that's the one thing
        ``TidesSolver`` itself owns (see its docstring), so to run the
        result at arbitrary precision, just set the attribute on what this
        returns before calling ``.solve()``::

            solver = HierarchicalSystemTemplates.make_nbody_solver(key)
            solver.is_mpfr = True
            t_hist, states = solver.solve(v_init, p_init, tini=0.0, tend=tend, dt=dt)

        Numba can't JIT-compile ``gmpy2.mpfr`` state, so combining
        ``use_numba=True`` here with ``is_mpfr = True`` afterward fails --
        ``nbody_mincseries`` raises ``ValueError`` once ``solve()`` actually
        calls it with mpfr state.
        """
        if physics not in _PHYSICS_GENERATORS:
            raise ValueError(f"Unknown physics: {physics!r} (expected one of {sorted(_PHYSICS_GENERATORS)})")
        if use_numba is not None and physics != "newtonian":
            raise ValueError("use_numba is only meaningful for physics='newtonian' -- the 1PN generator has no Numba-accelerated path")

        template = cls.get(key)
        extra_params = 1 if physics == "pn" else 0
        mincseries_func = _PHYSICS_GENERATORS[physics]
        if use_numba is not None:
            mincseries_func = functools.partial(mincseries_func, use_numba=use_numba)
        return TidesSolver(
            mincseries_func=mincseries_func,
            nvar=6 * template.n_bodies,
            npar=1 + template.n_bodies + extra_params,
            tolrel=tolrel,
            tolabs=tolabs,
            maxord=maxord,
            minord=minord,
            nordinc=nordinc,
            defect_error_control=defect_error_control,
            stepsize_controller=stepsize_controller,
        )

    @classmethod
    def _build_nbody_run(
        cls,
        key,
        masses,
        *,
        elements=None,
        G=1.0,
        names=None,
        physics="newtonian",
        use_numba=None,
        speed_of_light=None,
        tolrel=1e-12,
        tolabs=1e-12,
        maxord=32,
        minord=8,
        nordinc=5,
        defect_error_control=False,
        stepsize_controller="pytides",
    ):
        """
        Shared prep for ``solve_nbody``/``solve_hierarchy``: initial
        conditions plus a configured (but not yet run) ``TidesSolver`` --
        everything both need except the choice of *whether*/*how* to call
        ``.solve()``, which differs only in the single-body edge case (see
        ``solve_nbody``) and ``is_mpfr`` (``TidesSolver``-only, see
        ``make_nbody_solver``).

        Returns
        -------
        v_init, p_init, nodes, solver
        """
        template = cls.get(key)
        print(f"Hierarchical system: {template.key} ({template.title})")

        if physics == "pn" and speed_of_light is None:
            raise ValueError("physics='pn' requires speed_of_light")
        if physics == "pn":
            # Same fallback as build() below -- so the pair labels in this
            # warning match the body names actually used in the system
            # (and in any figure legend), instead of generic "body N".
            warn_names = names if names is not None else template.default_names
            warn_pn_quality(masses, names=warn_names, system_label=key, stacklevel=4)

        v_init, p_init, nodes = cls.initial_conditions(
            key,
            masses,
            elements=elements,
            G=G,
            names=names,
        )
        if physics == "pn":
            p_init = append_pn_params(p_init, speed_of_light)

        solver = cls.make_nbody_solver(
            key,
            physics=physics,
            use_numba=use_numba,
            tolrel=tolrel,
            tolabs=tolabs,
            maxord=maxord,
            minord=minord,
            nordinc=nordinc,
            defect_error_control=defect_error_control,
            stepsize_controller=stepsize_controller,
        )
        return v_init, p_init, nodes, solver

    @classmethod
    def solve_nbody(
        cls,
        key,
        masses,
        *,
        elements=None,
        G=1.0,
        names=None,
        tini=0.0,
        tend=10.0,
        dt=0.1,
        physics="newtonian",
        use_numba=None,
        speed_of_light=None,
        tolrel=1e-12,
        tolabs=1e-12,
        maxord=32,
        minord=8,
        nordinc=5,
        defect_error_control=False,
        stepsize_controller="pytides",
    ):
        """
        Build and integrate one hierarchy as an N-body problem, in double
        precision.

        ``physics="pn"`` adds the pairwise "dominant mass" 1PN correction
        (``exotides/relativity.py``) and requires ``speed_of_light`` (in the
        same length/time units as everything else -- there's no universal
        default since ``G`` itself is a free unit choice here).

        ``use_numba`` picks whether the Numba-JIT core is used for
        ``physics="newtonian"`` (``None`` auto-detects; see
        ``nbody_mincseries``).

        There's no ``is_mpfr`` here -- this is the plain double-precision
        convenience path. For arbitrary precision, build the pieces
        yourself and set it on the solver directly (see
        ``make_nbody_solver``), or use ``solve_hierarchy``, which does
        exactly that.

        Raises ``ValueError`` for a single-body template: a lone static
        body has no dynamics (every Taylor coefficient past order 0 is
        exactly zero), which ``TidesSolver``'s adaptive step-size heuristic
        can't handle -- there's nothing to integrate, so this refuses
        rather than trying and failing with a cryptic error. Use
        ``solve_hierarchy`` for a uniform call that works for every
        template regardless of body count.

        Returns
        -------
        t_hist, states, p_init, nodes
            ``states`` has shape ``(len(t_hist), 6*N)`` and can be unpacked
            with ``exotides.nbody.unpack_state``.
        """
        if cls.get(key).n_bodies == 1:
            raise ValueError(
                f"{key}: a single-body template has no dynamics to integrate -- "
                "use solve_hierarchy instead, which holds it fixed rather than integrating"
            )

        v_init, p_init, nodes, solver = cls._build_nbody_run(
            key, masses, elements=elements, G=G, names=names, physics=physics,
            use_numba=use_numba, speed_of_light=speed_of_light, tolrel=tolrel,
            tolabs=tolabs, maxord=maxord, minord=minord, nordinc=nordinc,
            defect_error_control=defect_error_control,
            stepsize_controller=stepsize_controller,
        )
        t_hist, states = solver.solve(v_init, p_init, tini=tini, tend=tend, dt=dt)
        return t_hist, states, p_init, nodes

    @classmethod
    def solve_hierarchy(
        cls,
        key,
        masses,
        *,
        elements=None,
        G=1.0,
        names=None,
        tini=0.0,
        tend=10.0,
        dt=0.1,
        is_mpfr=None,
        physics="newtonian",
        use_numba=None,
        speed_of_light=None,
        tolrel=1e-12,
        tolabs=1e-12,
        maxord=32,
        minord=8,
        nordinc=5,
        defect_error_control=False,
        stepsize_controller="pytides",
    ):
        """
        Like ``solve_nbody`` (built from the same ``_build_nbody_run``, in
        turn built from ``initial_conditions``/``make_nbody_solver``), but
        works uniformly for *every* template regardless of body count, and
        accepts ``is_mpfr``.

        A single-body template is never handed to ``TidesSolver.solve`` --
        see ``solve_nbody`` for why that fails -- instead the body is just
        held fixed at its initial state across ``[tini, tend]``, since
        there's nothing to integrate (the ``solver`` ``_build_nbody_run``
        still builds in that case is simply discarded unused).

        ``is_mpfr`` is passed straight to the ``TidesSolver`` built here
        (set as its ``is_mpfr`` attribute before ``.solve()`` is called --
        see ``make_nbody_solver``, which has none of its own): ``None``
        (default) auto-detects, ``True``/``False`` force arbitrary/double
        precision. Not meaningful (ignored) for a single-body template,
        since there's no integration to run at any precision.

        Returns
        -------
        t_hist, states, p_init, nodes
            ``states`` has shape ``(len(t_hist), 6*N)`` and can be unpacked
            with ``exotides.nbody.unpack_state``.
        """
        v_init, p_init, nodes, solver = cls._build_nbody_run(
            key, masses, elements=elements, G=G, names=names, physics=physics,
            use_numba=use_numba, speed_of_light=speed_of_light, tolrel=tolrel,
            tolabs=tolabs, maxord=maxord, minord=minord, nordinc=nordinc,
            defect_error_control=defect_error_control,
            stepsize_controller=stepsize_controller,
        )
        if cls.get(key).n_bodies == 1:
            t_hist = np.array([tini, tend], dtype=np.float64)
            states = np.array([v_init, v_init], dtype=np.float64)
            return t_hist, states, p_init, nodes

        solver.is_mpfr = is_mpfr
        t_hist, states = solver.solve(v_init, p_init, tini=tini, tend=tend, dt=dt)
        return t_hist, states, p_init, nodes

    @classmethod
    def validate(cls, template):
        """Validate semantic limits and parent-child type rules."""
        if template.parents[0] is not None or template.body_types[0] != "star":
            raise ValueError(f"{template.key}: body 0 must be the root star")

        if template.count_type("star") > MAX_STARS:
            raise ValueError(f"{template.key}: too many stars")
        if template.count_type("planet") > MAX_PLANETS:
            raise ValueError(f"{template.key}: too many planets")
        if template.count_type("moon") > MAX_MOONS:
            raise ValueError(f"{template.key}: too many moons")

        planet_parents = []
        for body_idx in range(1, template.n_bodies):
            parent_idx = template.parents[body_idx]
            if parent_idx is None or parent_idx >= body_idx:
                raise ValueError(f"{template.key}: invalid parent for body {body_idx}")

            body_type = template.body_types[body_idx]
            parent_type = template.body_types[parent_idx]
            orbit_class = template.orbit_classes[body_idx]

            if body_type == "star" and parent_type != "star":
                raise ValueError(f"{template.key}: stars must orbit stars")
            if body_type == "planet":
                if parent_type != "star":
                    raise ValueError(f"{template.key}: planets must orbit stars or stellar subsystems")
                if orbit_class not in {"S", "P"}:
                    raise ValueError(f"{template.key}: planets must be marked as S or P")
                if orbit_class == "P" and not cls._has_binary_subsystem(template, parent_idx):
                    raise ValueError(f"{template.key}: P-type planets require a binary-star subsystem")
                planet_parents.append(parent_idx if orbit_class == "S" else f"P:{parent_idx}")
            if body_type == "moon" and parent_type != "planet":
                raise ValueError(f"{template.key}: moons must orbit planets")

        # No further restriction on top of the per-planet checks above: two
        # S-type planets may share a star (binary_two_s_planets_same_star),
        # orbit different stars (binary_two_s_planets), or both be P-type
        # circumbinary planets (binary_two_p_planets) -- Jacobi ordering by
        # semi-major axis (HierarchicalSystem._resolve) handles every
        # combination correctly.

        return True

    @staticmethod
    def _has_binary_subsystem(template, parent_idx):
        if template.body_types[parent_idx] != "star":
            return False
        for idx, body_type in enumerate(template.body_types):
            if idx != parent_idx and body_type == "star" and template.parents[idx] == parent_idx:
                return True
        return False

    @staticmethod
    def _subtree_mass(template, masses, idx):
        """Total mass of ``idx`` and all its descendants (mirrors ``Node.sys_mass``)."""
        total = masses[idx]
        for child_idx, parent_idx in enumerate(template.parents):
            if parent_idx == idx:
                total += HierarchicalSystemTemplates._subtree_mass(template, masses, child_idx)
        return total

    @staticmethod
    def _default_semi_major_axis(template, idx, base_a):
        """The ``a`` that ``default_elements`` assigns to body ``idx`` (order-independent)."""
        body_type = template.body_types[idx]
        orbit_class = template.orbit_classes[idx]
        count = sum(1 for j in range(1, idx + 1) if template.body_types[j] == body_type)
        if body_type == "star":
            return base_a * (8.0 + 5.0 * count)
        if body_type == "planet" and orbit_class == "P":
            return base_a * (25.0 + 5.0 * (count - 1))
        if body_type == "planet":
            return base_a * (1.0 + 0.2 * count)
        return None  # moons don't gate anyone else's ordering

    @staticmethod
    def _check_moon_hill_stability(system, template, body_idx, names, moon_elements):
        """Warn about a moon whose orbit reaches outside its planet's Hill sphere."""
        planet_idx = template.parents[body_idx]
        star_idx = template.parents[planet_idx]
        planet_node = system.nodes[names[planet_idx]]
        star_node = system.nodes[names[star_idx]]

        # Same Jacobi-ordering convention as HierarchicalSystem._resolve():
        # the mass the planet's own orbit is anchored to is its parent
        # star's own mass plus any *other* sibling that orbits closer in
        # (e.g. a sibling star in a binary that's tighter than the planet's
        # own orbit) -- a wider sibling is an external perturber and is
        # excluded.
        planet_a = planet_node.elements["a"]
        central_mass = star_node.mass + sum(
            sibling.sys_mass for sibling in star_node.children
            if sibling is not planet_node and sibling.elements["a"] < planet_a
        )

        r_hill = planet_a * (planet_node.mass / (3.0 * central_mass)) ** (1.0 / 3.0)
        a_moon = moon_elements["a"]
        if a_moon > MOON_HILL_FRACTION * r_hill:
            warnings.warn(
                f"{template.key}: moon semi-major axis {a_moon:g} exceeds "
                f"{MOON_HILL_FRACTION:g} * Hill radius ({r_hill:g}) of parent planet "
                f"'{names[planet_idx]}'; the orbit is unlikely to be stable against "
                "stellar perturbation",
                stacklevel=2,
            )

    @staticmethod
    def _body_elements(elements, body_idx):
        if isinstance(elements, dict):
            try:
                return elements[body_idx]
            except KeyError as exc:
                raise ValueError(f"Missing orbital elements for body {body_idx}") from exc
        try:
            body_elements = elements[body_idx]
        except IndexError as exc:
            raise ValueError(f"Missing orbital elements for body {body_idx}") from exc
        if body_elements is None:
            raise ValueError(f"Missing orbital elements for body {body_idx}")
        return body_elements

    @classmethod
    def default_elements(cls, key, masses=None, base_a=1.0, eccentricity=0.01):
        """
        Generate simple default elements for smoke tests.

        S-type planets use planetary-scale semi-major axes. P-type planets are
        placed outside the binary. Moons are placed at
        ``_DEFAULT_MOON_HILL_FRACTION`` of their parent planet's Hill radius,
        comfortably under the hard limit enforced by ``build()``.

        ``masses`` should match the ``masses`` that will be passed to
        ``build``/``initial_conditions``/``solve_nbody``; it is only needed to
        size moon orbits correctly. If omitted, it defaults to the same
        star=1.0 / planet-or-moon=1e-4 convention used throughout this
        module's smoke tests.
        """
        template = cls.get(key)
        if masses is None:
            masses = [1.0 if body_type == "star" else 1.0e-4 for body_type in template.body_types]

        elements = {0: None}
        type_counts = {"star": 0, "planet": 0, "moon": 0}

        for body_idx in range(1, template.n_bodies):
            body_type = template.body_types[body_idx]
            orbit_class = template.orbit_classes[body_idx]
            type_counts[body_type] += 1
            count = type_counts[body_type]

            if body_type == "star":
                a = base_a * (8.0 + 5.0 * count)
                inc = 0.2 * count
            elif body_type == "planet" and orbit_class == "P":
                a = base_a * (25.0 + 5.0 * (count - 1))
                inc = 0.08
            elif body_type == "planet":
                a = base_a * (1.0 + 0.2 * count)
                inc = 0.03 * count
            else:
                planet_idx = template.parents[body_idx]
                star_idx = template.parents[planet_idx]
                a_planet = elements[planet_idx]["a"]
                # Same Jacobi-ordering convention as
                # HierarchicalSystem._resolve(): include a sibling of the
                # planet only if its own orbit is interior to the planet's
                # (smaller "a") -- e.g. the other star in a binary, for a
                # wide P-type planet; a narrower S-type planet excludes it.
                central_mass = masses[star_idx] + sum(
                    cls._subtree_mass(template, masses, sibling_idx)
                    for sibling_idx, parent_idx in enumerate(template.parents)
                    if parent_idx == star_idx
                    and sibling_idx != planet_idx
                    and cls._default_semi_major_axis(template, sibling_idx, base_a) < a_planet
                )
                r_hill = a_planet * (masses[planet_idx] / (3.0 * central_mass)) ** (1.0 / 3.0)
                a = _DEFAULT_MOON_HILL_FRACTION * r_hill
                inc = 0.05

            elements[body_idx] = {
                "a": a,
                "e": eccentricity,
                "i": inc,
                "lan": 0.1 * body_idx,
                "aop": 0.2 * body_idx,
                "ta": 0.3 * body_idx,
            }
        return elements
