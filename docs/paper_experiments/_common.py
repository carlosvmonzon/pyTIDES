"""
docs/paper_experiments/_common.py

Shared helpers for the paper-reproduction scripts in this folder. These are
standalone analysis scripts for docs/paper.tex (Section "Numerical
validation"/"Performance"), independent of tests/helpers (which assumes
tests/ is on sys.path when the test scripts run) -- they only import the
installed `exotides` package plus plain numpy/matplotlib.

Each script writes its own figure(s) under figures/paper_figures/ and prints
the numeric results quoted in the paper text, so every number in the paper
is traceable to a specific, rerunnable script here.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PYTIDES_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PYTIDES_ROOT / "figures" / "paper_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, name, **kwargs):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"saved: {path}")
    return path


def build_two_body_system(
    mass_star=1.0, mass_planet=1.0e-6, a=1.0, e=0.4, i=0.0, lan=0.0, aop=0.0, ta=0.0, G=1.0,
):
    """Star + one negligible-mass planet, no hierarchy template involved."""
    from exotides.orbital import HierarchicalSystem

    system = HierarchicalSystem(G=G)
    system.add_body("Star", mass=mass_star)
    system.add_body(
        "Planet", mass=mass_planet, parent_name="Star",
        elements={"a": a, "e": e, "i": i, "lan": lan, "aop": aop, "ta": ta},
    )
    return system.generate()


def build_two_body_system_mpfr(
    mass_star=1.0, mass_planet=1.0e-6, a=1.0, e=0.4, i=0.0, lan=0.0, aop=0.0, ta=0.0, G=1.0,
):
    """
    Same as ``build_two_body_system``, but every scalar (G, masses,
    elements) is converted to ``gmpy2.mpfr`` *before* being handed to
    ``HierarchicalSystem`` -- required for a genuinely high-precision
    initial condition. ``keplerian_to_cartesian`` only dispatches its
    trig/sqrt calls to gmpy2 when its arguments are already ``gmpy2.mpfr``
    (see exotides/orbital.py's ``_sin``/``_cos``/``_sqrt``); building in
    float64 and casting the *result* to mpfr afterwards bakes in float64
    rounding (~1e-16) that no amount of later integration precision can
    remove (see tests/test_lagrange_three_body.py's docstring for the same
    point). Requires ``gmpy2.get_context().precision`` to already be set to
    the desired bit precision.
    """
    from exotides.core import to_mpfr
    from exotides.orbital import HierarchicalSystem

    G_mpfr = to_mpfr(G)
    system = HierarchicalSystem(G=G_mpfr)
    system.add_body("Star", mass=to_mpfr(mass_star))
    system.add_body(
        "Planet", mass=to_mpfr(mass_planet), parent_name="Star",
        elements={
            "a": to_mpfr(a), "e": to_mpfr(e), "i": to_mpfr(i),
            "lan": to_mpfr(lan), "aop": to_mpfr(aop), "ta": to_mpfr(ta),
        },
    )
    return system.generate()
