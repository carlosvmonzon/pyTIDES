"""
Python/tests/helpers/event_helpers.py

Shared assertions for terminal-collision-event tests (test_events.py).
"""

import re

import numpy as np

from exotides.nbody import unpack_state


def print_collision(label, hit, states, nodes):
    """Print which bodies (by name) collided, when, and at what separation."""
    match = re.match(r"collision\(body(\d+), body(\d+)\)", hit["name"])
    i, j = int(match.group(1)), int(match.group(2))
    positions, _ = unpack_state(states[-1])
    separation = np.linalg.norm(positions[j] - positions[i])
    print(f"[{label}] collision: {nodes[i].name} <-> {nodes[j].name} "
          f"at t={float(hit['time']):.4f} yr (separation={separation:.6g} AU)")


def assert_single_terminal_collision(solver, t_hist, states, radii, tend, *, label=None, nodes=None):
    """
    Assert exactly one terminal collision event fired -- well before
    ``tend`` (i.e. the collision itself, not just reaching the end of the
    requested span, is what stopped integration) -- and that the
    separation of whichever pair collided equals the sum of their radii
    (``radii``, one entry per body, indexed the same as ``states``).

    If ``label`` and ``nodes`` are both given, also prints the collision
    (via ``print_collision``) once the assertions pass.
    """
    assert len(solver.last_events) == 1
    hit = solver.last_events[0]
    assert hit["terminal"] is True

    match = re.match(r"collision\(body(\d+), body(\d+)\)", hit["name"])
    assert match, f"unexpected event name: {hit['name']}"
    i, j = int(match.group(1)), int(match.group(2))

    assert float(t_hist[-1]) < tend
    assert float(t_hist[-1]) == float(hit["time"])

    positions, _ = unpack_state(states[-1])
    separation = np.linalg.norm(positions[j] - positions[i])
    r_sum = radii[i] + radii[j]
    assert abs(separation - r_sum) < 1e-9, (
        f"separation at recorded event ({separation:.9g}) should equal r_sum ({r_sum:.9g})"
    )

    if label is not None and nodes is not None:
        print_collision(label, hit, states, nodes)

    return hit
