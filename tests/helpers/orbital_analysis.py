"""
Python/tests/helpers/orbital_analysis.py

Osculating-orbit analysis shared across tests that need more than a saved
plot from exotides.plotting.plot_orbital_elements (e.g. computing a precession
*rate*, or comparing two runs' angle time series directly). Consolidates
what used to be two near-duplicate eccentricity-vector-angle computations
(test_relativity.py's measure_precession_per_orbit and a locally nested
pericenter_angles helper).
"""

import math

import numpy as np


def eccentricity_vector_angles(positions, velocities, mu, body_idx=1, parent_idx=0):
    """
    Unwrapped in-plane eccentricity-vector angle (pericenter longitude) time
    series for one body relative to another. Used instead of the osculating
    argument-of-pericenter from cartesian_to_keplerian, since aop/lan are
    degenerate for a planar orbit (i=0 -> ascending node undefined) -- the
    eccentricity vector's angle in the orbital plane has no such
    degeneracy.
    """
    n = len(positions)
    angle = np.empty(n)
    for idx in range(n):
        r_vec = positions[idx, body_idx] - positions[idx, parent_idx]
        v_vec = velocities[idx, body_idx] - velocities[idx, parent_idx]
        r = np.linalg.norm(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
        angle[idx] = math.atan2(e_vec[1], e_vec[0])
    return np.unwrap(angle)


def precession_rate(t_hist, positions, velocities, mu, body_idx=1, parent_idx=0):
    """Linear-fit precession rate (rad per unit time) of the pericenter longitude."""
    angles = eccentricity_vector_angles(positions, velocities, mu, body_idx, parent_idx)
    rate, _ = np.polyfit(t_hist, angles, 1)
    return rate
