"""
exotides/orbital.py

Orbital mechanics utilities for the TIDES N-body package.

Contents
--------
- keplerian_to_cartesian  - convert classical Keplerian elements to Cartesian
                            relative position and velocity vectors.
- cartesian_to_keplerian  - the inverse conversion, to osculating elements.
- HierarchicalSystem      - build barycentric initial conditions for
                            hierarchical N-body configurations defined via
                            a tree of orbital elements.

keplerian_to_cartesian and HierarchicalSystem support double precision
(float) and multiple precision (gmpy2.mpfr) transparently;
cartesian_to_keplerian always works in float64.

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

import math
import numpy as np
from .core import HAS_GMPY2, gmpy2


# ---------------------------------------------------------------------------
# Scalar trig helpers
# ---------------------------------------------------------------------------

def _sin(x):
    if HAS_GMPY2 and isinstance(x, gmpy2.mpfr):
        return gmpy2.sin(x)
    return math.sin(x)

def _cos(x):
    if HAS_GMPY2 and isinstance(x, gmpy2.mpfr):
        return gmpy2.cos(x)
    return math.cos(x)

def _sqrt(x):
    if HAS_GMPY2 and isinstance(x, gmpy2.mpfr):
        return gmpy2.sqrt(x)
    return math.sqrt(x)

# ---------------------------------------------------------------------------
# Keplerian elements → Cartesian state
# ---------------------------------------------------------------------------

def keplerian_to_cartesian(a, e, i, lan, aop, ta, mu):
    """
    Convert classical Keplerian orbital elements to relative Cartesian state.

    All angular quantities must be given in **radians**.

    Parameters
    ----------
    a   : semi-major axis
    e   : eccentricity  (0 ≤ e < 1 for elliptic orbits)
    i   : inclination
    lan : longitude of the ascending node (Ω)
    aop : argument of periapsis (ω)
    ta  : true anomaly (ν)
    mu  : gravitational parameter  G * (m1 + m2)

    Returns
    -------
    pos : ndarray, shape (3,)
        Relative position vector in the inertial reference frame.
    vel : ndarray, shape (3,)
        Relative velocity vector in the inertial reference frame.
    """
    is_mpfr = HAS_GMPY2 and any(
        isinstance(x, gmpy2.mpfr) for x in [a, e, i, lan, aop, ta, mu]
    )

    # Work first in the perifocal frame of the orbital plane, where Kepler's
    # equations are straightforward.
    p_param  = a * (1.0 - e ** 2)
    r        = p_param / (1.0 + e * _cos(ta))
    x_peri   = r * _cos(ta)
    y_peri   = r * _sin(ta)
    z_peri   = a * 0.0   # preserves mpfr zero if needed

    # Velocity in the same perifocal frame.
    h        = _sqrt(mu * p_param)
    vx_peri  = -(mu / h) * _sin(ta)
    vy_peri  =  (mu / h) * (e + _cos(ta))
    vz_peri  = a * 0.0

    # Rotation matrix from the orbital plane to the inertial frame:
    # R = Rz(lan) * Rx(i) * Rz(aop), with active rotations.
    cos_lan = _cos(lan);  sin_lan = _sin(lan)
    cos_aop = _cos(aop);  sin_aop = _sin(aop)
    cos_i   = _cos(i);    sin_i   = _sin(i)

    r11 =  cos_lan * cos_aop - sin_lan * sin_aop * cos_i
    r12 = -cos_lan * sin_aop - sin_lan * cos_aop * cos_i
    r13 =  sin_lan * sin_i

    r21 =  sin_lan * cos_aop + cos_lan * sin_aop * cos_i
    r22 = -sin_lan * sin_aop + cos_lan * cos_aop * cos_i
    r23 = -cos_lan * sin_i

    r31 =  sin_aop * sin_i
    r32 =  cos_aop * sin_i
    r33 =  cos_i

    x  = r11 * x_peri + r12 * y_peri + r13 * z_peri
    y  = r21 * x_peri + r22 * y_peri + r23 * z_peri
    z  = r31 * x_peri + r32 * y_peri + r33 * z_peri

    vx = r11 * vx_peri + r12 * vy_peri + r13 * vz_peri
    vy = r21 * vx_peri + r22 * vy_peri + r23 * vz_peri
    vz = r31 * vx_peri + r32 * vy_peri + r33 * vz_peri

    dtype = object if is_mpfr else np.float64
    return np.array([x, y, z], dtype=dtype), np.array([vx, vy, vz], dtype=dtype)

def cartesian_to_keplerian(pos, vel, mu):
    """
    Convert a Cartesian relative state to osculating Keplerian elements.

    Parameters
    ----------
    pos, vel : array-like, shape (3,)
        Relative position and velocity vectors.
    mu : scalar
        Gravitational parameter ``G * (m_primary + m_secondary)``.

    Returns
    -------
    dict
        Keys ``a``, ``e``, ``i``, ``lan``, ``aop`` and ``ta``. Angular
        quantities are in radians.
    """
    r_vec = np.asarray(pos, dtype=np.float64)
    v_vec = np.asarray(vel, dtype=np.float64)

    r = np.linalg.norm(r_vec)
    v2 = float(np.dot(v_vec, v_vec))

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    if h == 0.0:
        raise ValueError("Angular momentum is zero; orbital plane is undefined")

    k_vec = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)

    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = np.linalg.norm(e_vec)

    energy = 0.5 * v2 - mu / r
    a = -mu / (2.0 * energy) if energy != 0.0 else np.inf

    inc = math.acos(np.clip(h_vec[2] / h, -1.0, 1.0))

    if n > 0.0:
        lan = math.atan2(n_vec[1], n_vec[0])
    else:
        lan = 0.0

    if n > 0.0 and e > 0.0:
        aop = math.atan2(
            np.dot(np.cross(n_vec, e_vec), h_vec) / (n * e * h),
            np.dot(n_vec, e_vec) / (n * e),
        )
    elif e > 0.0:
        # Orbital plane coincides with the reference (x, y) plane (i = 0 or i = pi):
        # the ascending node is undefined, so by convention lan = 0 and aop absorbs
        # the full in-plane periapsis angle, measured in the direction of motion
        # (the sign flip handles the retrograde case, where that direction is
        # clockwise rather than counter-clockwise as seen from +z).
        sign = 1.0 if h_vec[2] >= 0.0 else -1.0
        aop = math.atan2(sign * e_vec[1], e_vec[0])
    else:
        aop = 0.0

    if e > 0.0:
        ta = math.atan2(
            np.dot(np.cross(e_vec, r_vec), h_vec) / (e * r * h),
            np.dot(e_vec, r_vec) / (e * r),
        )
    else:
        ta = 0.0

    return {
        "a": a,
        "e": e,
        "i": inc,
        "lan": lan,
        "aop": aop,
        "ta": ta,
    }


# ---------------------------------------------------------------------------
# Hierarchical system builder
# ---------------------------------------------------------------------------

class HierarchicalSystem:
    """
    Build barycentric Cartesian initial conditions for hierarchical N-body
    configurations defined as a tree of Keplerian elements.

    Usage example (Sun - Earth - Moon)::

        sys = HierarchicalSystem(G=1.0)
        sys.add_body("Sun",   mass=1.0)
        sys.add_body("Earth", mass=3e-6, parent_name="Sun",
                     elements={"a": 1.0, "e": 0.0, "i": 0.0,
                                "lan": 0.0, "aop": 0.0, "ta": 0.0})
        sys.add_body("Moon",  mass=3.7e-8, parent_name="Earth",
                     elements={"a": 0.00257, "e": 0.0, "i": 0.0897,
                                "lan": 0.0, "aop": 0.0, "ta": 0.0})
        v_init, p_init, nodes = sys.generate()

    Built in plain ``float64`` by default. Passing ``G`` as a ``gmpy2.mpfr``
    builds ``v_init``/``p_init`` at that same arbitrary precision instead --
    ``keplerian_to_cartesian`` already dispatches its trig/sqrt to gmpy2
    transparently (see module docstring), so the only thing needed for a
    genuinely full-precision system is for every number handed to
    ``add_body`` (``mass``, and each ``elements`` value) to already be a
    ``gmpy2.mpfr`` too -- e.g. ``gmpy2.mpfr("0.05")`` rather than the
    Python float ``0.05``, and ``gmpy2.mpfr(65) * gmpy2.const_pi() / 180``
    rather than ``math.radians(65.0)`` -- since mixing in even one plain
    float reintroduces a float64-rounded seed value into the tree (see
    test_lagrange_three_body.py's module docstring for why that seed can
    matter for unstable/chaotic configurations even when everything
    downstream is computed at higher precision). Passing ``is_mpfr=True``
    to ``exotides.core.TidesSolver`` alone -- without building the initial
    condition itself this way -- only gets you higher-precision
    *integration*, not a higher-precision *starting point*.

    Parameters
    ----------
    G : float or gmpy2.mpfr
        Gravitational constant.
    """

    # ------------------------------------------------------------------
    # Internal node
    # ------------------------------------------------------------------

    class Node:
        """Represents one body in the hierarchy."""
        def __init__(self, name, mass, parent_name=None, elements=None):
            self.name        = name
            self.mass        = mass
            self.parent_name = parent_name
            self.elements    = elements    # dict: a, e, i, lan, aop, ta
            self.children    = []
            self.pos         = None
            self.vel         = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, G=1.0):
        self.G       = G if (HAS_GMPY2 and isinstance(G, gmpy2.mpfr)) else float(G)
        self.nodes   = {}
        self.root    = None

    def _is_mpfr(self):
        """Whether this system is being built at gmpy2 arbitrary precision (set by ``G``'s type)."""
        return HAS_GMPY2 and isinstance(self.G, gmpy2.mpfr)

    #: Keplerian element keys required in ``elements`` for every non-root body.
    _REQUIRED_ELEMENT_KEYS = ("a", "e", "i", "lan", "aop", "ta")

    def add_body(self, name, mass, parent_name=None, elements=None):
        """
        Register a body in the hierarchy.

        Parameters
        ----------
        name : str
            Unique identifier for the body.
        mass : float
            Mass of the body. Must be positive.
        parent_name : str or None
            Name of the body this one orbits.  Set to ``None`` for the
            primary (root) body.
        elements : dict or None
            Keplerian elements in radians:
            ``{"a", "e", "i", "lan", "aop", "ta"}``. They describe the orbit
            of the barycenter of this body's subsystem (itself plus any
            descendants on tighter orbits) around the barycenter of
            everything interior to it (its parent plus any bodies on orbits
            tighter than this one) -- see ``_build_subsystems``.
            Required for all non-root bodies; ``a`` must be positive (this
            builder only supports elliptic orbits, ``0 <= e < 1`` -- see
            ``keplerian_to_cartesian``).

        Raises
        ------
        ValueError
            If ``name`` was already used, ``mass`` isn't positive, a second
            root body is added, the parent is unknown, ``elements`` is
            missing/incomplete for a non-root body, or ``elements["a"]``
            isn't positive.
        """
        if name in self.nodes:
            raise ValueError(f"Body '{name}' was already added -- names must be unique.")
        if not (mass > 0):
            raise ValueError(f"Body '{name}': mass must be positive, got {mass!r}.")

        if parent_name is not None:
            if elements is None:
                raise ValueError(
                    f"Body '{name}': elements are required for non-root bodies "
                    f"(orbiting '{parent_name}')."
                )
            missing = [key for key in self._REQUIRED_ELEMENT_KEYS if key not in elements]
            if missing:
                raise ValueError(
                    f"Body '{name}': elements is missing required key(s) {missing} "
                    f"-- expected all of {list(self._REQUIRED_ELEMENT_KEYS)}."
                )
            if not (elements["a"] > 0):
                raise ValueError(
                    f"Body '{name}': elements['a'] (semi-major axis) must be positive, "
                    f"got {elements['a']!r}."
                )

        node = self.Node(name, mass, parent_name, elements)
        self.nodes[name] = node

        if parent_name is None:
            if self.root is not None:
                raise ValueError("Only one primary (root) body is allowed.")
            self.root = node
        else:
            if parent_name not in self.nodes:
                raise ValueError(f"Parent body '{parent_name}' not found. "
                                 "Add bodies in hierarchical order.")
            parent = self.nodes[parent_name]
            parent.children.append(node)

    # ------------------------------------------------------------------
    # Coordinate resolution
    # ------------------------------------------------------------------

    class Subsystem:
        """
        One node of the binary tree of nested subsystems: either a single
        body (leaf) or the Keplerian orbit joining two subsystems, ``inner``
        (everything interior to that orbit) and ``outer`` (the subsystem
        bound to the body whose ``elements`` describe that orbit).
        """
        def __init__(self, mass, bodies, inner=None, outer=None, elements=None):
            self.mass     = mass
            self.bodies   = bodies
            self.inner    = inner
            self.outer    = outer
            self.elements = elements

    def _build_subsystems(self):
        """
        Turn the parent/child tree into a binary tree of nested subsystems
        and return its root (the whole system).

        Orbits are merged in ascending semi-major-axis order: each body's
        current subsystem (itself plus every descendant already merged into
        it, i.e. on a tighter orbit) is joined to its parent's current
        subsystem (the parent plus everything already interior to the
        body's orbit). This is the same "smaller a is interior" ordering
        rule used to tell S-type from P-type orbits, applied consistently to
        both sides of every orbit -- so e.g. a 2+2 quadruple resolves to
        (A,B) + (C,D) on the outer orbit, and a third star chained under one
        component of a tighter inner binary orbits the inner pair's
        barycenter. Each merge joins two distinct subsystems (the parent
        links form a tree), so the result always has exactly one root.
        """
        owner = {name: self.Subsystem(node.mass, [node]) for name, node in self.nodes.items()}
        orbiting = sorted(
            (node for node in self.nodes.values() if node.parent_name is not None),
            key=lambda node: node.elements["a"],
        )
        for node in orbiting:
            inner = owner[node.parent_name]
            outer = owner[node.name]
            merged = self.Subsystem(inner.mass + outer.mass, inner.bodies + outer.bodies,
                                    inner, outer, node.elements)
            for body in merged.bodies:
                owner[body.name] = merged
        return owner[self.root.name]

    def _place(self, subsystem, pos, vel):
        """
        Place ``subsystem``'s barycenter at (pos, vel) and recurse: the
        orbit's elements give the relative vector from ``inner``'s
        barycenter to ``outer``'s, with mu = G * (M_inner + M_outer), and
        each side is offset from the common barycenter by its mass
        fraction. Exact for any hierarchy, and barycentric by construction
        when started from the origin.
        """
        if subsystem.inner is None:
            body = subsystem.bodies[0]
            body.pos = pos
            body.vel = vel
            return

        el = subsystem.elements
        r_rel, v_rel = keplerian_to_cartesian(
            el["a"], el["e"], el["i"], el["lan"], el["aop"], el["ta"], self.G * subsystem.mass
        )
        f_inner = subsystem.outer.mass / subsystem.mass
        f_outer = subsystem.inner.mass / subsystem.mass
        self._place(subsystem.inner, pos - f_inner * r_rel, vel - f_inner * v_rel)
        self._place(subsystem.outer, pos + f_outer * r_rel, vel + f_outer * v_rel)

    def _zero_vector(self):
        zero = gmpy2.mpfr(0.0) if self._is_mpfr() else 0.0
        return np.array([zero, zero, zero], dtype=object if self._is_mpfr() else np.float64)

    # ------------------------------------------------------------------
    # State generation
    # ------------------------------------------------------------------

    def generate(self):
        """
        Compute barycentric Cartesian initial conditions for the whole system.

        Returns
        -------
        v_init : ndarray, shape (6*N,)
            Flat state vector in pre-order traversal order.
        p_init : ndarray, shape (1+N,)
            Parameters: ``[G, mass0, mass1, ...]``.
        ordered_nodes : list of Node
            Nodes in the same pre-order as v_init.

        Raises
        ------
        ValueError
            If no primary body has been added.
        """
        if self.root is None:
            raise ValueError("No primary body has been added.")

        self._place(self._build_subsystems(), self._zero_vector(), self._zero_vector())

        # Pre-order traversal: this order determines how positions,
        # velocities and masses are packed into the output vectors.
        ordered_nodes = []

        def traverse(node):
            ordered_nodes.append(node)
            for child in node.children:
                traverse(child)

        traverse(self.root)

        N     = len(ordered_nodes)
        dtype = object if self._is_mpfr() else np.float64

        v_init    = np.empty(6 * N,     dtype=dtype)
        p_init    = np.empty(1 + N,     dtype=dtype)
        p_init[0] = self.G

        for idx, node in enumerate(ordered_nodes):
            v_init[6 * idx     : 6 * idx + 3] = node.pos
            v_init[6 * idx + 3 : 6 * idx + 6] = node.vel
            p_init[1 + idx]                    = node.mass

        return v_init, p_init, ordered_nodes
