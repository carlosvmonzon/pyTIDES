"""
exotides/__init__.py

Public API for the TIDES Python package.
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
