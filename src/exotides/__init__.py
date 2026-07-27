"""
exotides/__init__.py

Public API for the TIDES Python package.

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

from .core import (
    TidesSolver,
    mul_mc,
    div_mc,
    inv_mc,
    exp_mc,
    pow_mc_c,
    log_mc,
    sin_mc,
    cos_mc,
    vector_norm,
    HAS_GMPY2,
    gmpy2,
)

from .nbody import (
    nbody_mincseries,
    pack_state,
    unpack_state,
    compute_energy,
)

from .orbital import (
    keplerian_to_cartesian,
    cartesian_to_keplerian,
    HierarchicalSystem,
)

from .hierarchy import (
    HierarchyTemplate,
    HierarchicalSystemTemplates,
)

from .plotting import save_figure

from .events import Event, collision_event, all_pairs_collision_events

# API pública del paquete: importar desde exotides expone solo estas funciones y
# clases, manteniendo internos los detalles de implementación.
__all__ = [
    # Solver & algebra
    "TidesSolver",
    "mul_mc", "div_mc", "inv_mc", "exp_mc", "pow_mc_c", "log_mc",
    "sin_mc", "cos_mc",
    "HAS_GMPY2", "gmpy2", "vector_norm",
    # N-body
    "nbody_mincseries",
    "pack_state", "unpack_state",
    "compute_energy",
    # Orbital mechanics
    "keplerian_to_cartesian",
    "cartesian_to_keplerian",
    "HierarchicalSystem",
    "HierarchyTemplate",
    "HierarchicalSystemTemplates",
    # Plotting
    "save_figure",
    # Events
    "Event",
    "collision_event",
    "all_pairs_collision_events",
]
