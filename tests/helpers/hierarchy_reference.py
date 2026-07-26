"""
Python/tests/helpers/hierarchy_reference.py

Generic helpers for picking "the smaller body" of a hierarchy template and
the correct (possibly multi-body, Jacobi-ordered) reference frame its
osculating elements should be measured against -- used by test_hierarchy.py
to plot every template's smaller body automatically.
"""

import math


def select_smaller_body(template, masses, elements):
    """
    Pick the "smaller body" whose orbit best represents a template: the
    smallest-mass body (excluding the root, which has no orbit). Ties are
    broken by picking the most exterior orbit (largest own semi-major
    axis) -- e.g. a lone moon is always the unique lightest body, so it's
    picked automatically; two tied-mass S-type planets or stellar
    companions resolve to whichever orbits farther out. A fully symmetric
    tie is broken deterministically by the highest body index.
    """
    candidates = list(range(1, template.n_bodies))
    min_mass = min(masses[idx] for idx in candidates)
    tied = [idx for idx in candidates if math.isclose(masses[idx], min_mass, rel_tol=1e-9)]
    if len(tied) == 1:
        return tied[0]

    max_a = max(elements[idx]["a"] for idx in tied)
    outermost = [idx for idx in tied if math.isclose(elements[idx]["a"], max_a, rel_tol=1e-9)]
    return max(outermost)


def _children_of(template, node_idx):
    return [idx for idx, parent_idx in enumerate(template.parents) if parent_idx == node_idx]


def _subtree_indices(template, node_idx):
    result = [node_idx]
    for child_idx in _children_of(template, node_idx):
        result += _subtree_indices(template, child_idx)
    return result


def _local_interior_indices(template, elements, node_idx, exclude_idx):
    exclude_a = elements[exclude_idx]["a"] if exclude_idx is not None else None
    result = [node_idx]
    for child_idx in _children_of(template, node_idx):
        if child_idx == exclude_idx:
            continue
        if exclude_a is None or elements[child_idx]["a"] < exclude_a:
            result += _subtree_indices(template, child_idx)
    return result


def reference_template_indices(template, elements, target_idx):
    """
    Template body indices forming the reference frame a body's osculating
    elements should be measured against: its parent, plus any other child
    of the parent with a smaller own semi-major axis, plus (walking up)
    each ancestor's own local contribution, so long as that ancestor link
    is itself interior to the target's orbit. Mirrors
    ``HierarchicalSystem._local_interior``/``_interior_reference``
    (src/exotides/orbital.py) -- same Jacobi-ordering rule needed to get e.g. a
    P-type/circumbinary planet's reference right (both binary components
    combined, not just its immediate template parent), computed directly
    from the template's static topology instead of a resolved
    ``HierarchicalSystem``.
    """
    parent_idx = template.parents[target_idx]
    indices = _local_interior_indices(template, elements, parent_idx, target_idx)

    curr_idx = parent_idx
    while template.parents[curr_idx] is not None and elements[curr_idx]["a"] < elements[target_idx]["a"]:
        grandparent_idx = template.parents[curr_idx]
        indices += _local_interior_indices(template, elements, grandparent_idx, curr_idx)
        curr_idx = grandparent_idx

    return indices


def state_index_by_name(nodes, name):
    """
    Map a body's semantic name to its index in the state-vector/``nodes``
    order. Needed because that order is a preorder tree traversal
    (HierarchicalSystem.generate()), which is *not* always the same as a
    template's own body-index numbering (topological: parent index < child
    index, but not necessarily a preorder) -- e.g. in binary_two_s_planets,
    Star B's child Planet B is visited before Star A's other child
    Planet A, so naively reusing template indices to index into
    positions/nodes would silently grab the wrong body's trajectory.
    """
    return [node.name for node in nodes].index(name)
