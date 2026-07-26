"""
exotides/orbital.py

Orbital mechanics utilities for the TIDES N-body package.

Contents
--------
- keplerian_to_cartesian  - convert classical Keplerian elements to Cartesian
                            relative position and velocity vectors.
- HierarchicalSystem      - build barycentric initial conditions for
                            hierarchical N-body configurations defined via
                            a tree of orbital elements.

Both routines support double precision (float) and multiple precision
(gmpy2.mpfr) transparently.
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

    # Primero se calcula en el plano orbital perifocal, donde las ecuaciones
    # de Kepler son directas.
    p_param  = a * (1.0 - e ** 2)
    r        = p_param / (1.0 + e * _cos(ta))
    x_peri   = r * _cos(ta)
    y_peri   = r * _sin(ta)
    z_peri   = a * 0.0   # preserves mpfr zero if needed

    # Velocidad en el mismo sistema perifocal.
    h        = _sqrt(mu * p_param)
    vx_peri  = -(mu / h) * _sin(ta)
    vy_peri  =  (mu / h) * (e + _cos(ta))
    vz_peri  = a * 0.0

    # Matriz de rotación desde el plano orbital al marco inercial:
    # R = Rz(-lan) * Rx(-i) * Rz(-aop).
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
            self.sys_mass    = mass        # total mass of this body + descendants

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
            ``{"a", "e", "i", "lan", "aop", "ta"}``.
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
            zero      = gmpy2.mpfr(0.0) if self._is_mpfr() else 0.0
            node.pos  = np.array([zero, zero, zero], dtype=object if self._is_mpfr() else np.float64)
            node.vel  = np.array([zero, zero, zero], dtype=object if self._is_mpfr() else np.float64)
        else:
            if parent_name not in self.nodes:
                raise ValueError(f"Parent body '{parent_name}' not found. "
                                 "Add bodies in hierarchical order.")
            parent = self.nodes[parent_name]
            parent.children.append(node)

            # Propaga la masa total del subsistema hacia arriba. Esa masa se
            # usa después para colocar baricentros correctamente.
            curr = parent
            while curr is not None:
                curr.sys_mass += node.sys_mass
                curr = self.nodes.get(curr.parent_name)

    # ------------------------------------------------------------------
    # Coordinate resolution
    # ------------------------------------------------------------------

    def _subtree_barycenter(self, node):
        """Mass-weighted (mass, pos, vel) of node's entire resolved subtree."""
        total_mass = node.mass
        wpos = node.mass * node.pos
        wvel = node.mass * node.vel
        for child in node.children:
            if child.pos is None:
                continue
            m, p, v = self._subtree_barycenter(child)
            total_mass += m
            wpos = wpos + m * p
            wvel = wvel + m * v
        return total_mass, wpos / total_mass, wvel / total_mass

    def _local_interior(self, node, exclude_child):
        """
        (mass, pos, vel) of node's own local contribution to whatever a
        deeper descendant orbits: node's own mass, plus any *other* already-
        resolved child of node (using that child's whole resolved subtree)
        whose orbit is interior to ``exclude_child``'s (smaller semi-major
        axis). A sibling that orbits farther out is an external perturber
        and is excluded.
        """
        total_mass = node.mass
        wpos = node.mass * node.pos
        wvel = node.mass * node.vel
        exclude_a = exclude_child.elements["a"] if exclude_child is not None else None
        for other in node.children:
            if other is exclude_child or other.pos is None:
                continue
            if exclude_a is None or other.elements["a"] < exclude_a:
                m, p, v = self._subtree_barycenter(other)
                total_mass += m
                wpos = wpos + m * p
                wvel = wvel + m * v
        return total_mass, wpos / total_mass, wvel / total_mass

    def _interior_reference(self, node, child):
        """
        (mass, pos, vel) of the full effective "interior" reference frame
        ``child`` orbits: node's own local interior (see
        ``_local_interior``) combined with node's entire ancestor chain,
        each ancestor contributing its own local interior in turn. This is
        what lets a body several levels deep (e.g. a third star orbiting
        one component of an inner binary) correctly feel the *whole* inner
        structure's mass, not just its immediate parent's.
        """
        total_mass, pos, vel = self._local_interior(node, child)
        wpos = total_mass * pos
        wvel = total_mass * vel

        # Keep walking up only while each link is *interior* to child's own
        # orbit (smaller semi-major axis) -- same ordering rule as siblings,
        # applied going up the tree. Otherwise that ancestor (and everything
        # further up) is objectively farther away than child's own orbital
        # scale and must be treated as an external perturber, not folded
        # into mu: e.g. a planet tightly orbiting one star of a binary
        # should ignore the companion star if the binary is wider than the
        # planet's own orbit, even though the companion is architecturally
        # an "uncle" via that star's parent.
        curr = node
        while curr.parent_name is not None and curr.elements["a"] < child.elements["a"]:
            parent = self.nodes[curr.parent_name]
            m, p, v = self._local_interior(parent, curr)
            total_mass += m
            wpos = wpos + m * p
            wvel = wvel + m * v
            curr = parent

        return total_mass, wpos / total_mass, wvel / total_mass

    def _resolve(self, node):
        """
        Recursively assign Cartesian coordinates starting from *node*,
        using the Keplerian elements of its children, in an *unshifted*
        working frame (the global root stays fixed at the origin
        throughout -- see ``generate()`` for the barycentering step).

        Children of the same parent are resolved in ascending semi-major-
        axis order (innermost first), and each child's orbit is anchored to
        the full effective mass/position of everything interior to it (its
        parent's local system *and* the parent's entire ancestor chain) --
        proper Jacobi coordinates, not just an immediate-parent
        approximation.

        This deliberately does *not* re-center anything after each
        insertion (an earlier version did, incrementally, and that scheme
        turned out to only be correct when a body's interior reference
        never folds in an ancestor beyond its immediate parent -- e.g. it
        silently broke the barycentric constraint for genuinely nested
        chains such as a third star orbiting an inner binary's barycenter,
        where the correction needs to reach further up the tree than
        whichever subtree looked like the obvious target). Working in a
        single unshifted frame and barycentering once at the very end
        sidesteps the question of "how far up does this correction reach"
        entirely: every ``r_rel``/``v_rel`` computed here is a relative
        vector, unaffected by whatever rigid translation the final
        barycentering step turns out to apply.
        """
        for child in sorted(node.children, key=lambda c: c.elements["a"]):
            el = child.elements

            central_mass, central_pos, central_vel = self._interior_reference(node, child)

            mu = self.G * (central_mass + child.sys_mass)
            r_rel, v_rel = keplerian_to_cartesian(
                el["a"], el["e"], el["i"], el["lan"], el["aop"], el["ta"], mu
            )

            child.pos = central_pos + r_rel
            child.vel = central_vel + v_rel

            self._resolve(child)

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

        self._resolve(self.root)

        # Recorrido en preorden: este orden determina cómo se empaquetan
        # posiciones, velocidades y masas en los vectores de salida.
        ordered_nodes = []

        def traverse(node):
            ordered_nodes.append(node)
            for child in node.children:
                traverse(child)

        traverse(self.root)

        # Single barycentering pass over the whole system, using each
        # body's own bare mass -- correct regardless of how many levels
        # deep the tree goes, since every position/velocity assigned by
        # _resolve() is a chain of relative vectors from the (arbitrarily
        # placed) root: shifting all of them by the same rigid amount
        # changes no relative geometry, only recenters the ensemble on its
        # true center of mass. See _resolve()'s docstring for why this
        # replaced an earlier incremental-shift scheme that broke down for
        # genuinely nested (3+ level) hierarchies.
        total_mass = sum(n.mass for n in ordered_nodes)
        R_cm = sum(n.mass * n.pos for n in ordered_nodes) / total_mass
        V_cm = sum(n.mass * n.vel for n in ordered_nodes) / total_mass
        for n in ordered_nodes:
            n.pos = n.pos - R_cm
            n.vel = n.vel - V_cm

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
