"""
Python/tests/helpers/systems.py

Generic HierarchicalSystem builder, consolidating the "add N bodies, then
generate()" pattern that used to be hand-rolled once per test file
(build_kozai_system, build_two_body, build_triple_system, ...), plus a
``build_two_body_system`` convenience specialization for the common bare
star+planet case.
"""

from exotides.orbital import HierarchicalSystem


def build_system(bodies, G=1.0):
    """
    Build barycentric initial conditions for a chain/tree of bodies.

    Parameters
    ----------
    bodies : sequence of dict
        One entry per body, added in order (parents must come before their
        children), each with keys ``name``, ``mass``, and optionally
        ``parent_name`` (``None``/omitted for the root) and ``elements``
        (required for every non-root body: a dict with ``a, e, i, lan,
        aop, ta``).
    G : float
        Gravitational constant.

    Returns
    -------
    v_init, p_init, nodes
        Same as ``HierarchicalSystem.generate()``.
    """
    system = HierarchicalSystem(G=G)
    for body in bodies:
        system.add_body(
            body["name"],
            mass=body["mass"],
            parent_name=body.get("parent_name"),
            elements=body.get("elements"),
        )
    return system.generate()


def build_two_body_system(
    mass_star=1.0, mass_planet=1.0e-6, a=1.0, e=0.4, i=0.0, lan=0.0, aop=0.0, ta=0.0, G=1.0,
):
    """
    Convenience wrapper around ``build_system`` for the common "bare star +
    one planet, no hierarchy template" case -- e.g. testing a physics
    correction (1PN, ...) in isolation, without going through
    ``exotides.hierarchy.HierarchicalSystemTemplates``.
    """
    return build_system([
        {"name": "Star", "mass": mass_star},
        {
            "name": "Planet", "mass": mass_planet, "parent_name": "Star",
            "elements": {"a": a, "e": e, "i": i, "lan": lan, "aop": aop, "ta": ta},
        },
    ], G=G)
