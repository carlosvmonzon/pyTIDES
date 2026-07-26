"""
exotides/events.py

Generic zero-crossing event detection during integration (the ``events``
parameter of ``exotides.core.TidesSolver.solve``), plus an N-body collision
convenience builder.

This ports the original C TIDES library's event-detection subsystem
(``doubEVENTS.c``/``mpfrEVENTS.c``: ``dp_tides_find_zeros``,
``dp_tides_find_extrema``, ``dp_tides_events``, etc., listed in
``C/libTIDES/doubEVENTS.h``) -- a capability that existed in the original C
code and its tutorial (``events.nb``) but was never ported to this Python
package. The C version locates a zero of a user-supplied scalar function of
the state by root-finding directly on the dense Taylor polynomial already
computed for the current integration step (no re-integration or fixed-step
shrinking needed); ``TidesSolver.solve`` does the same here, bisecting on
its own ``horner`` evaluation within each accepted step.

Unlike REBOUND's collision handling (which merges or bounces particles
mid-simulation), pyTIDES targets a *fixed* hierarchy-template tree topology
(exotides/hierarchy.py) where merging two bodies would invalidate the tree
structure the rest of the system assumes. So collisions here are treated as
a *terminal* event: integration stops exactly at the moment of contact and
the caller decides what to do next (flag it, restart with a different
template, etc.), rather than the simulation silently continuing with a
mutated body list.
"""

import math


class Event:
    """
    A zero-crossing event watched for during ``TidesSolver.solve``.

    Parameters
    ----------
    function : callable
        ``function(t, v) -> scalar``, evaluated with the absolute time
        ``t`` and the current state vector ``v`` (length ``nvar``).
    terminal : bool
        If True (default), integration stops exactly at the crossing --
        see ``TidesSolver.solve``'s ``events`` parameter. If False, the
        crossing is recorded in ``TidesSolver.last_events`` but
        integration continues.
    direction : int
        0 (default): trigger on any sign change. +1: only rising
        (function goes from negative to positive). -1: only falling.
    name : str, optional
        Label recorded in ``TidesSolver.last_events`` (defaults to the
        function's own ``__name__``).
    """

    def __init__(self, function, *, terminal=True, direction=0, name=None):
        self.function = function
        self.terminal = terminal
        self.direction = direction
        self.name = name or getattr(function, "__name__", "event")


def collision_event(body_i, body_j, radius_i, radius_j, *, name=None):
    """
    Terminal ``Event`` that fires when bodies ``body_i``/``body_j``
    (0-indexed, in the usual N-body state layout
    ``v[6*k+0..2] = x, y, z`` -- see exotides/nbody.py) come within
    ``radius_i + radius_j`` of each other -- a physical collision.
    """
    r_sum = radius_i + radius_j

    def _distance_minus_radii(t, v):
        dx = v[6 * body_j + 0] - v[6 * body_i + 0]
        dy = v[6 * body_j + 1] - v[6 * body_i + 1]
        dz = v[6 * body_j + 2] - v[6 * body_i + 2]
        return math.sqrt(dx * dx + dy * dy + dz * dz) - r_sum

    return Event(
        _distance_minus_radii, terminal=True, direction=-1,
        name=name or f"collision(body{body_i}, body{body_j})",
    )


def all_pairs_collision_events(n_bodies, radii):
    """
    One ``collision_event`` per unique pair of bodies, given a ``radii``
    sequence of length ``n_bodies`` (physical radius of each body, same
    length units as position).
    """
    events = []
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            events.append(collision_event(i, j, radii[i], radii[j]))
    return events
