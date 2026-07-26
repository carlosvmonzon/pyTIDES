"""
Python/tests/helpers/catalog_runner.py

Shared "solve one hierarchy-template configuration and save its 3D plot
(+ optional animation)" loop body, consolidating what used to be two
near-identical implementations (test_hierarchy.py's plot_examples and
test_exoplanet_systems.py's plot_real_systems).
"""

from exotides.hierarchy import HierarchicalSystemTemplates
from exotides.nbody import unpack_state
from exotides.plotting import plot_hierarchy


def solve_and_plot_hierarchy(
    output_key, template_key, masses, elements, tend, dt, title, output_dir,
    *, G=1.0, names=None, animate=False, solver_settings=None, is_mpfr=None
):
    """
    Solve one hierarchy-template configuration (``solve_hierarchy``) and
    save its 3D trajectory plot (``plot_hierarchy``, + optional animation).

    ``output_dir=None`` skips plotting/animation entirely (only the solve
    runs) -- the fast path used under pytest, where numeric assertions
    still run but no figures get written; see docs/design-notes.md and the
    test suite's ``output_dir=None`` default convention.

    Returns
    -------
    t_hist, states, p_init, nodes, output_paths
    """
    template = HierarchicalSystemTemplates.get(template_key)
    solver_settings = dict(solver_settings or {})

    t_hist, states, p_init, nodes = HierarchicalSystemTemplates.solve_hierarchy(
        template_key, masses, elements=elements, G=G, names=names,
        tend=tend, dt=dt, is_mpfr=is_mpfr, **solver_settings,
    )

    if template.n_bodies == 1 or output_dir is None:
        # A single static body has no trajectory to show -- relative to
        # itself (ref_idx=0) it's just a fixed point at the origin, not a
        # plot worth generating. output_dir=None means "skip plotting".
        output_paths = []
    else:
        positions, _ = unpack_state(states)
        assert positions.shape == (len(t_hist), template.n_bodies, 3)
        output_paths = plot_hierarchy(output_key, title, states, nodes, output_dir, animate=animate)

    return t_hist, states, p_init, nodes, output_paths
