"""
exotides/plotting.py

Plotting helpers for the N-body examples and the hierarchy-template
catalog (exotides/hierarchy.py).

Contents
--------
- save_figure            - save a figure under a per-example ``figures/``
                            subfolder.
- plot_orbit / animate_orbit
                         - static/animated trajectory of one solved N-body
                           system, positions relative to a chosen body or
                           (by default) the mass-weighted center of mass. 3D,
                           unless every orbit is coplanar, in which case a 2D
                           (x, y) plot is used instead. ``body_indices``
                           restricts which bodies are actually drawn (e.g.
                           just an inner orbit within a wider hierarchy).
- plot_orbital_elements  - one body's osculating (a, e, i) vs. time, measured
                           against its correct (possibly multi-body, see the
                           Jacobi-ordering note in README.md) reference frame.
- plot_hierarchy         - plot_orbit/animate_orbit bundle for an already-solved
                           hierarchy-template system (relative to its root
                           star, body 0) -- the plotting half of
                           exotides.hierarchy.HierarchicalSystemTemplates.
                           solve_hierarchy's solve half.
- plot_pn_comparison     - side-by-side Newtonian vs. 1PN relative orbit and
                           pericenter-precession comparison for any catalog
                           template (exotides/relativity.py).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from .core import as_float64
from .nbody import unpack_state
from .orbital import cartesian_to_keplerian


# Cap on points drawn per fading trail segment in animate_orbit -- plenty
# dense for a smooth-looking curve regardless of how fine the underlying
# solve dt is (see animate_orbit's set_trail). Cheap to render even at
# several hundred segments per body per frame, so err on the high side --
# a low cap is what produces a visibly faceted/polygonal curve when the
# trail window spans many periods of a fast orbit.
_MAX_TRAIL_POINTS = 600


def save_figure(fig, filename: str, output_dir: str | Path | None = None) -> Path:
    """Save a matplotlib figure inside a dedicated figures subfolder.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str
        File name to use for the output image.
    output_dir : str | Path | None
        Base directory where the figures folder will be created. If None,
        the current working directory is used.
    """
    # Centraliza la carpeta de figuras para que los ejemplos no dependan del
    # directorio desde el que se ejecuten.
    base_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / filename
    fig.savefig(output_path)
    return output_path


def _is_coplanar(rel_positions: np.ndarray, rtol: float = 1e-9) -> bool:
    """True if every body's z-coordinate is negligible next to the x/y spread.

    Used to pick a 2D (x, y) plot over a 3D one for hierarchies whose
    orbits all share a common plane (e.g. all inclinations zero) -- a 3D
    axes box only adds visual clutter (foreshortening, an unused z-axis)
    when the trajectories are flat.
    """
    xy_extent = float(np.max(np.abs(rel_positions[:, :, :2]))) or 1.0
    z_extent = float(np.max(np.abs(rel_positions[:, :, 2])))
    return z_extent <= rtol * xy_extent


def _reference_positions(positions: np.ndarray, nodes, ref_idx: int | None) -> np.ndarray:
    """Per-frame reference point to plot ``positions`` relative to, shape ``(T, 1, 3)``.

    ``ref_idx=None`` uses the mass-weighted center of mass of all bodies
    (masses read from ``node.mass``); an integer uses that body's position.
    """
    if ref_idx is None:
        masses = np.array([float(node.mass) for node in nodes])
        return np.average(positions, axis=1, weights=masses)[:, None, :]
    return positions[:, ref_idx : ref_idx + 1, :]


def _hierarchy_axes(rel_positions: np.ndarray, title: str, coplanar: bool):
    extent = float(np.max(np.abs(rel_positions))) * 1.05 or 1.0

    fig = plt.figure(figsize=(6, 6))
    if coplanar:
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_zlim(-extent, extent)
        ax.view_init(elev=22, azim=35)
    return fig, ax


def plot_orbit(
    key: str,
    title: str,
    states: np.ndarray,
    nodes,
    output_dir: str | Path | None = None,
    *,
    ref_idx: int | None = None,
    body_indices=None,
) -> Path:
    """Save a static trajectory plot of one solved N-body system.

    Positions are plotted relative to ``nodes[ref_idx]``. If ``ref_idx`` is
    None (the default), the reference is the mass-weighted center of mass of
    all bodies (masses read from ``node.mass``), computed frame by frame so
    it absorbs any center-of-mass drift instead of leaving the system
    sliding across the frame -- the natural choice for a system with no
    hierarchy root. ``ref_idx`` is always resolved against the *full*
    ``nodes``/``states``, even if ``body_indices`` excludes it from being
    drawn.

    ``body_indices``, if given, restricts which bodies are actually drawn
    (e.g. ``[1]`` to plot only the inner orbit of a hierarchical triple,
    dropping a much wider outer companion that would otherwise dominate the
    axis scale) -- indices are into the original, full ``nodes``, not the
    drawn subset.

    Saved as a 2D (x, y) plot -- ``{key}_2d.png`` -- when every drawn orbit
    is coplanar (see ``_is_coplanar``), otherwise as a 3D plot --
    ``{key}_3d.png``.
    """
    positions, _ = unpack_state(states)
    positions = as_float64(positions)
    rel_positions = positions - _reference_positions(positions, nodes, ref_idx)
    if body_indices is not None:
        rel_positions = rel_positions[:, body_indices, :]
        nodes = [nodes[i] for i in body_indices]
    x = rel_positions[:, :, 0]
    y = rel_positions[:, :, 1]
    z = rel_positions[:, :, 2]
    coplanar = _is_coplanar(rel_positions)

    fig, ax = _hierarchy_axes(rel_positions, title, coplanar)
    colors = plt.cm.tab10(np.linspace(0, 1, len(nodes)))
    for body_idx, node in enumerate(nodes):
        if coplanar:
            ax.plot(x[:, body_idx], y[:, body_idx], lw=1.0, color=colors[body_idx], label=node.name)
            ax.scatter(x[-1, body_idx], y[-1, body_idx], color=colors[body_idx], s=30)
        else:
            ax.plot(x[:, body_idx], y[:, body_idx], z[:, body_idx], lw=1.0, color=colors[body_idx], label=node.name)
            ax.scatter(x[-1, body_idx], y[-1, body_idx], z[-1, body_idx], color=colors[body_idx], s=30)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    suffix = "2d" if coplanar else "3d"
    output_path = save_figure(fig, f"{key}_{suffix}.png", output_dir=output_dir)
    plt.close(fig)
    return output_path


def animate_orbit(
    key: str,
    title: str,
    states: np.ndarray,
    nodes,
    output_dir: str | Path | None = None,
    *,
    frame_count: int = 200,
    fps: int = 30,
    ref_idx: int | None = None,
    body_indices=None,
    trail_frames: int | None = None,
) -> Path:
    """Save a GIF animation of one solved N-body system.

    Positions are plotted relative to ``nodes[ref_idx]``. If ``ref_idx`` is
    None (the default), the reference is the mass-weighted center of mass of
    all bodies (masses read from ``node.mass``), computed frame by frame --
    see ``plot_orbit`` for why. ``body_indices`` restricts which bodies are
    drawn -- see ``plot_orbit`` (same convention: indices into the full,
    original ``nodes``). Saved as a 2D (x, y) animation -- ``{key}_2d.gif``
    -- when every drawn orbit is coplanar (see ``_is_coplanar``), otherwise
    as a 3D animation -- ``{key}_3d.gif``.

    ``fps`` controls playback speed (lower = slower). ``trail_frames``, if
    given, keeps only the last N *animation* frames (out of ``frame_count``,
    not raw trajectory points) behind each body, fading from transparent
    (oldest) to fully opaque (current position) -- a "comet tail" instead
    of the default trail that keeps the entire history from t=0 onward.
    """
    positions, _ = unpack_state(states)
    positions = as_float64(positions)
    rel_positions = positions - _reference_positions(positions, nodes, ref_idx)
    if body_indices is not None:
        rel_positions = rel_positions[:, body_indices, :]
        nodes = [nodes[i] for i in body_indices]
    frame_indices = np.linspace(0, rel_positions.shape[0] - 1, frame_count, dtype=int)

    x = rel_positions[:, :, 0]
    y = rel_positions[:, :, 1]
    z = rel_positions[:, :, 2]
    coplanar = _is_coplanar(rel_positions)

    fig, ax = _hierarchy_axes(rel_positions, title, coplanar)
    colors = plt.cm.tab10(np.linspace(0, 1, len(nodes)))
    markers = [
        ax.plot([], [], "o", color=colors[idx], ms=5)[0] for idx in range(len(nodes))
    ]

    if trail_frames is None:
        trails = [
            ax.plot([], [], lw=1.0, color=colors[idx], label=node.name)[0]
            for idx, node in enumerate(nodes)
        ]

        def set_trail(body_idx, pos, start_pos):
            frame_idx = frame_indices[pos]
            if coplanar:
                trails[body_idx].set_data(x[: frame_idx + 1, body_idx], y[: frame_idx + 1, body_idx])
            else:
                trails[body_idx].set_data(x[: frame_idx + 1, body_idx], y[: frame_idx + 1, body_idx])
                trails[body_idx].set_3d_properties(z[: frame_idx + 1, body_idx])
    else:
        # LineCollection/Line3DCollection give per-segment alpha (the
        # fade), but don't reliably carry a legend entry the way Line2D
        # does across matplotlib versions -- an invisible same-color
        # Line2D per body carries the label instead.
        for idx, node in enumerate(nodes):
            ax.plot([], [], lw=1.0, color=colors[idx], label=node.name)
        collection_cls = LineCollection if coplanar else Line3DCollection
        trails = [collection_cls([], lw=1.0, color=colors[idx]) for idx in range(len(nodes))]
        for trail in trails:
            # autolim=False: axis limits are already fixed by
            # _hierarchy_axes, and autolim would otherwise choke on an
            # empty collection before the first update() call fills it in.
            ax.add_collection(trail) if coplanar else ax.add_collection3d(trail, autolim=False)

        def set_trail(body_idx, pos, start_pos):
            # Dense range over every raw trajectory point between the two
            # animation frames (not just frame_indices[start_pos:pos+1] --
            # those are only ~frame_count points spread over the *entire*
            # run, often coarser than one orbital period, which would
            # connect near-random points on the fast inner orbit into a
            # star/zigzag instead of tracing it). Capped to
            # _MAX_TRAIL_POINTS so a very fine solve dt doesn't balloon the
            # segment count per frame -- that many points is already far
            # denser than needed for a smooth-looking curve.
            start_idx, end_idx = frame_indices[start_pos], frame_indices[pos]
            n_raw = end_idx - start_idx + 1
            if n_raw < 2:
                trails[body_idx].set_segments([])
                return
            sample_idx = np.linspace(start_idx, end_idx, min(n_raw, _MAX_TRAIL_POINTS), dtype=int)
            if coplanar:
                points = np.column_stack([x[sample_idx, body_idx], y[sample_idx, body_idx]])
            else:
                points = np.column_stack([x[sample_idx, body_idx], y[sample_idx, body_idx], z[sample_idx, body_idx]])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            trails[body_idx].set_segments(segments)
            # Oldest segment ~transparent, most recent ~opaque -- the fade.
            trails[body_idx].set_alpha(np.linspace(0.05, 1.0, len(segments)))

    ax.legend(loc="upper right", fontsize=8)

    def update(pos: int):
        frame_idx = frame_indices[pos]
        start_pos = 0 if trail_frames is None else max(0, pos - trail_frames)
        for body_idx in range(len(nodes)):
            set_trail(body_idx, pos, start_pos)
            if coplanar:
                markers[body_idx].set_data([x[frame_idx, body_idx]], [y[frame_idx, body_idx]])
            else:
                markers[body_idx].set_data([x[frame_idx, body_idx]], [y[frame_idx, body_idx]])
                markers[body_idx].set_3d_properties([z[frame_idx, body_idx]])
        return [*trails, *markers]

    animation = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    base_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    suffix = "2d" if coplanar else "3d"
    output_path = base_dir / "figures" / f"{key}_{suffix}.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def plot_hierarchy(
    key: str,
    title: str,
    states: np.ndarray,
    nodes,
    output_dir: str | Path | None = None,
    *,
    animate: bool = False,
) -> list:
    """Save the trajectory plot (and, if requested, animation) of one solved hierarchy-template system.

    The plotting half of ``exotides.hierarchy.HierarchicalSystemTemplates.
    solve_hierarchy``'s solve half -- takes the ``(states, nodes)`` it
    returns directly, no re-solving involved. Always relative to ``nodes[0]``
    (``ref_idx=0``): every hierarchy template is validated to have the root
    star there (see ``HierarchicalSystemTemplates.validate``), which is what
    these plots have always shown the system relative to -- ``plot_orbit``/
    ``animate_orbit``'s own default (``ref_idx=None``, center of mass) is
    meant for non-hierarchical systems with no such natural root.

    Returns
    -------
    list of Path
        ``[plot_path]``, or ``[plot_path, animation_path]`` if ``animate``.
    """
    output_paths = [plot_orbit(key, title, states, nodes, output_dir, ref_idx=0)]
    if animate:
        output_paths.append(animate_orbit(key, title, states, nodes, output_dir, ref_idx=0))
    return output_paths


def plot_orbital_elements(
    key: str,
    title: str,
    t_hist,
    states: np.ndarray,
    p_init,
    body_idx: int,
    parent_idx,
    output_dir: str | Path | None = None,
    *,
    time_label: str = "time",
    angles: bool = False,
) -> Path | tuple[Path, Path]:
    """Save a 3-panel plot of one body's osculating (a, e, i) vs time.

    ``parent_idx`` is either a single body index (the simple two-body case,
    ``mu = G * (m_parent + m_body)``) or a sequence of indices -- for bodies
    whose true reference isn't just their immediate hierarchy parent (e.g. a
    P-type/circumbinary planet, whose orbit is around the *combined*
    binary barycenter, not one star alone; see the Jacobi-ordering note in
    README.md). In the sequence case, the reference position/velocity is
    the mass-weighted barycenter of those bodies (masses read from
    ``p_init``), and ``mu = G * (sum of those masses + m_body)`` -- so
    secular/periodic perturbation from the rest of the system shows up as
    real variation in the three panels, rather than as a spurious artifact
    of using the wrong reference mass/frame.

    If ``angles`` is True, also saves a second 3-panel plot of the
    osculating (lan, aop, ta) angles to ``{key}_angles.png``, and returns
    ``(elements_path, angles_path)`` instead of just ``elements_path``. Note
    ``lan``/``aop`` fall back to 0 wherever ``cartesian_to_keplerian`` finds
    the ascending node undefined (a coplanar instant, i.e. ``i`` exactly 0
    or 180 degrees) -- an artifact of that degeneracy, not a real jump.
    """
    positions, velocities = unpack_state(states)
    t = np.asarray(t_hist, dtype=np.float64)

    parent_indices = [parent_idx] if isinstance(parent_idx, int) else list(parent_idx)
    parent_masses = np.array([float(p_init[1 + idx]) for idx in parent_indices])
    total_parent_mass = float(parent_masses.sum())
    mu = float(p_init[0]) * (total_parent_mass + float(p_init[1 + body_idx]))

    n = len(t)
    a = np.empty(n)
    e = np.empty(n)
    inc = np.empty(n)
    lan = np.empty(n)
    aop = np.empty(n)
    ta = np.empty(n)
    for idx in range(n):
        ref_pos = np.average(positions[idx, parent_indices, :], axis=0, weights=parent_masses)
        ref_vel = np.average(velocities[idx, parent_indices, :], axis=0, weights=parent_masses)
        rel_pos = positions[idx, body_idx] - ref_pos
        rel_vel = velocities[idx, body_idx] - ref_vel
        elements = cartesian_to_keplerian(rel_pos, rel_vel, mu)
        a[idx] = elements["a"]
        e[idx] = elements["e"]
        inc[idx] = math.degrees(elements["i"])
        lan[idx] = math.degrees(elements["lan"])
        aop[idx] = math.degrees(elements["aop"])
        ta[idx] = math.degrees(elements["ta"])

    fig, (ax_a, ax_e, ax_i) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax_a.plot(t, a, color="tab:blue", lw=1.2)
    ax_a.set_ylabel("semi-major axis, a")
    ax_a.grid(True, alpha=0.3)

    ax_e.plot(t, e, color="tab:orange", lw=1.2)
    ax_e.set_ylabel("eccentricity, e")
    ax_e.grid(True, alpha=0.3)

    ax_i.plot(t, inc, color="tab:green", lw=1.2)
    ax_i.set_ylabel("inclination, i [deg]")
    ax_i.set_xlabel(time_label)
    ax_i.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    output_path = save_figure(fig, f"{key}_elements.png", output_dir=output_dir)
    plt.close(fig)

    if not angles:
        return output_path

    fig_ang, (ax_lan, ax_aop, ax_ta) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax_lan.plot(t, lan, color="tab:red", lw=1.2)
    ax_lan.set_ylabel("longitude of ascending node,\nlan [deg]")
    ax_lan.grid(True, alpha=0.3)

    ax_aop.plot(t, aop, color="tab:purple", lw=1.2)
    ax_aop.set_ylabel("argument of pericenter,\naop [deg]")
    ax_aop.grid(True, alpha=0.3)

    ax_ta.plot(t, ta, color="tab:brown", lw=1.2)
    ax_ta.set_ylabel("true anomaly, ta [deg]")
    ax_ta.set_xlabel(time_label)
    ax_ta.grid(True, alpha=0.3)

    fig_ang.suptitle(title)
    fig_ang.tight_layout()

    angles_path = save_figure(fig_ang, f"{key}_angles.png", output_dir=output_dir)
    plt.close(fig_ang)
    return output_path, angles_path


def pn_comparison_diagnostics(
    key: str,
    masses,
    elements,
    tend,
    dt,
    speed_of_light: float,
    *,
    body_idx: int = 1,
    parent_idx=0,
    G: float = 1.0,
    names=None,
    solver_settings: dict | None = None,
) -> dict:
    """
    Newtonian-vs-1PN integration and pericenter-drift diagnostics for one
    hierarchy-template configuration -- the numerical half of
    ``plot_pn_comparison``, split out so a caller that only needs the
    ``diagnostics`` dict (e.g. a fast pytest check with no figure to save)
    can skip matplotlib entirely. See ``plot_pn_comparison`` for the meaning
    of every parameter.

    Returns
    -------
    dict with keys ``t_n``, ``rel_n``, ``angles_n``, ``t_p``, ``rel_p``,
    ``angles_p`` (the raw data ``plot_pn_comparison`` plots) and
    ``diagnostics`` (``{"newtonian_drift_deg": float, "pn_drift_deg": float}``).
    """
    from .hierarchy import HierarchicalSystemTemplates

    solver_settings = dict(solver_settings or {})

    t_n, states_n, p_n, _ = HierarchicalSystemTemplates.solve_nbody(
        key, masses, elements=elements, G=G, names=names, tend=tend, dt=dt,
        physics="newtonian", **solver_settings,
    )
    t_p, states_p, _, _ = HierarchicalSystemTemplates.solve_nbody(
        key, masses, elements=elements, G=G, names=names, tend=tend, dt=dt,
        physics="pn", speed_of_light=speed_of_light, **solver_settings,
    )

    positions_n, velocities_n = unpack_state(states_n)
    positions_p, velocities_p = unpack_state(states_p)
    # See as_float64: np.linalg.norm/np.sqrt below can't run on dtype=object
    # gmpy2.mpfr arrays (is_mpfr=True in solver_settings), and this plot is
    # display-only anyway -- no precision loss that matters.
    positions_n = as_float64(positions_n)
    velocities_n = as_float64(velocities_n)
    positions_p = as_float64(positions_p)
    velocities_p = as_float64(velocities_p)

    parent_indices = [parent_idx] if isinstance(parent_idx, int) else list(parent_idx)
    parent_masses = np.array([float(p_n[1 + idx]) for idx in parent_indices])
    mu = G * (float(parent_masses.sum()) + float(p_n[1 + body_idx]))

    def _reference(positions, velocities, idx):
        ref_pos = np.average(positions[idx, parent_indices, :], axis=0, weights=parent_masses)
        ref_vel = np.average(velocities[idx, parent_indices, :], axis=0, weights=parent_masses)
        return ref_pos, ref_vel

    rel_n = positions_n[:, body_idx] - np.array(
        [_reference(positions_n, velocities_n, idx)[0] for idx in range(len(positions_n))]
    )
    rel_p = positions_p[:, body_idx] - np.array(
        [_reference(positions_p, velocities_p, idx)[0] for idx in range(len(positions_p))]
    )

    def _pericenter_angles_deg(positions, velocities):
        angle = np.empty(len(positions))
        for idx in range(len(positions)):
            ref_pos, ref_vel = _reference(positions, velocities, idx)
            r_vec = positions[idx, body_idx] - ref_pos
            v_vec = velocities[idx, body_idx] - ref_vel
            r = np.linalg.norm(r_vec)
            h_vec = np.cross(r_vec, v_vec)
            e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
            angle[idx] = math.atan2(e_vec[1], e_vec[0])
        return np.degrees(np.unwrap(angle))

    angles_n = _pericenter_angles_deg(positions_n, velocities_n)
    angles_p = _pericenter_angles_deg(positions_p, velocities_p)

    return {
        "t_n": t_n, "rel_n": rel_n, "angles_n": angles_n,
        "t_p": t_p, "rel_p": rel_p, "angles_p": angles_p,
        "diagnostics": {
            "newtonian_drift_deg": float(abs(angles_n[-1] - angles_n[0])),
            "pn_drift_deg": float(abs(angles_p[-1] - angles_p[0])),
        },
    }


def plot_pn_comparison(
    key: str,
    masses,
    elements,
    tend,
    dt,
    speed_of_light: float,
    output_dir: str | Path | None = None,
    *,
    body_idx: int = 1,
    parent_idx=0,
    G: float = 1.0,
    names=None,
    solver_settings: dict | None = None,
) -> tuple[Path, dict]:
    """
    Integrate one hierarchy-template configuration twice -- plain Newtonian
    gravity and with the 1PN pairwise correction (``physics="newtonian"``
    vs. ``physics="pn"``, see ``exotides/hierarchy.py`` and
    ``exotides/relativity.py``) -- and plot both the relative-orbit
    trajectory and the pericenter-longitude drift of ``body_idx`` relative
    to ``parent_idx``, side by side. Works with *any* catalog template, not
    just a bare two-body one -- point ``body_idx``/``parent_idx`` at
    whichever pair you want to examine (e.g. a planet and its host star, or
    one component of a stellar binary/triple).

    ``parent_idx`` is either a single body index or a sequence of indices --
    for bodies whose true reference isn't just one other body (e.g. the
    outer star of a nested triple, whose orbit is around the *combined*
    inner pair, not one component alone; see the Jacobi-ordering note in
    README.md and ``plot_orbital_elements``, which uses the same
    convention). The reference position/velocity is the mass-weighted
    barycenter of those bodies, and ``mu = G * (sum of those masses +
    m_body)`` -- correct even when ``body_idx`` and ``parent_idx`` are
    comparable-mass bodies (e.g. two stars in a binary), unlike a naive
    ``mu = G * m_parent`` that silently assumes ``body_idx`` is negligible.

    The Newtonian orbit stays a fixed, closed ellipse; the PN orbit visibly
    precesses. Delegates the actual integration/diagnostics to
    ``pn_comparison_diagnostics``.

    Returns
    -------
    output_path : Path
    diagnostics : dict
        ``{"newtonian_drift_deg": float, "pn_drift_deg": float}`` -- total
        pericenter-longitude drift (degrees) accumulated over ``tend`` for
        each physics choice, e.g. to sanity-check that the Newtonian run
        stayed flat while the PN run precessed measurably more.
    """
    result = pn_comparison_diagnostics(
        key, masses, elements, tend, dt, speed_of_light,
        body_idx=body_idx, parent_idx=parent_idx, G=G, names=names,
        solver_settings=solver_settings,
    )
    t_n, rel_n, angles_n = result["t_n"], result["rel_n"], result["angles_n"]
    t_p, rel_p, angles_p = result["t_p"], result["rel_p"], result["angles_p"]

    fig, (ax_orbit, ax_angle) = plt.subplots(1, 2, figsize=(11, 5))

    ax_orbit.plot(rel_n[:, 0], rel_n[:, 1], color="tab:blue", lw=1.2, label="Newtonian")
    ax_orbit.plot(rel_p[:, 0], rel_p[:, 1], color="tab:red", lw=1.0, alpha=0.8, label="1PN")
    ax_orbit.set_xlabel("x")
    ax_orbit.set_ylabel("y")
    ax_orbit.axis("equal")
    ax_orbit.set_title(f"{key}: relative orbit")
    ax_orbit.legend(loc="upper right", fontsize=8)
    ax_orbit.grid(True, alpha=0.3)

    ax_angle.plot(t_n, angles_n, color="tab:blue", lw=1.2, label="Newtonian")
    ax_angle.plot(t_p, angles_p, color="tab:red", lw=1.2, label="1PN")
    ax_angle.set_xlabel("time")
    ax_angle.set_ylabel("pericenter longitude [deg]")
    ax_angle.set_title("Apsidal precession")
    ax_angle.legend(loc="upper left", fontsize=8)
    ax_angle.grid(True, alpha=0.3)

    fig.suptitle(f"Newtonian vs. 1PN dynamics -- {key} (c={speed_of_light:g})")
    fig.tight_layout()

    output_path = save_figure(fig, f"{key}_newtonian_vs_pn.png", output_dir=output_dir)
    plt.close(fig)

    return output_path, result["diagnostics"]
