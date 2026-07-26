"""
tests/test_exoplanet_systems.py

Real multi-star exoplanet systems, mapped onto the hierarchy catalog in
src/exotides/hierarchy.py, integrated and rendered with src/exotides/plotting.py.

Units: solar masses, AU, and G = 4*pi**2, so Kepler's third law reduces to
the familiar T[yr]**2 = a[AU]**3 / M_total[Msun] (the same convention used
for Solar System ephemerides) -- periods come out directly in years.

Orbital parameters below are approximate, drawn from commonly cited
discovery/characterization papers from memory. They are illustrative, not a
precision reference -- do not use these numbers for anything scientific.
"""

import math

import matplotlib
matplotlib.use("Agg")

from helpers import PYTHON_DIR, solve_and_plot_hierarchy
from exotides.hierarchy import HierarchicalSystemTemplates

G_AU_MSUN_YR = 4.0 * math.pi ** 2

M_JUP = 9.543e-4    # Jupiter mass, in solar masses
M_EARTH = 3.003e-6  # Earth mass, in solar masses
M_MOON = 3.69e-8    # Moon mass, in solar masses


# Each entry: (name, template_key, masses, elements, tend, dt).
# ``name`` becomes the output filename and drives SOLVER_OVERRIDES below.
REAL_SYSTEMS = (
    (
        "51_pegasi_b",
        "star_planet",
        [1.04, 0.46 * M_JUP],
        {
            0: None,
            1: {"a": 0.0527, "e": 0.01, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        0.0596,  # ~5 orbital periods (P ~= 4.23 days)
        0.0003,
    ),
    (
        "sun_earth_moon",
        "star_planet_moon",
        [1.0, M_EARTH, M_MOON],
        {
            0: None,
            1: {"a": 1.0, "e": 0.0167, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 0.00257, "e": 0.0549, "i": 0.0897, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        3.0,  # 3 Earth years (~40 lunar months)
        0.005,
    ),
    (
        # 16 Cygni Bb: one of the first known highly eccentric exoplanets
        # (Cochran et al. 1997), orbiting the secondary star of a very wide
        # binary (separation of order 10^2-10^3 AU; the two components are
        # near-solar twins). A genuine, confirmed exoplanet in a real
        # multi-star system, unlike Alpha Centauri (no confirmed planet --
        # see the note at the bottom of this file).
        #
        # Star A itself is omitted here: at an ~860 AU separation its
        # gravity is dynamically negligible for the planet (ratio ~500x),
        # and it would render as a single static dot ~860 AU away while
        # making the planet's real, famously eccentric (e=0.689) orbit
        # around B shrink to an invisible speck -- so this renders just
        # star B and its planet, at the scale where the actual physics of
        # interest is visible.
        "16_cygni_bb",
        "star_planet",
        [1.00, 2.38 * M_JUP],
        {
            0: None,
            1: {"a": 1.693, "e": 0.689, "i": 45*2*math.pi/360, "lan": 0.3, "aop": 0.1, "ta": 0.0},
        },
        10.9,  # ~5 planet periods (P ~= 2.18 yr)
        0.036,
    ),
    (
        "gamma_cephei_ab",
        "binary_s_planet_primary",
        [1.40, 0.409, 1.85 * M_JUP],
        {
            0: None,
            1: {"a": 20.2, "e": 0.36, "i": 0.0, "lan": 0.0, "aop": 0.0, "ta": 0.0},
            2: {"a": 2.05, "e": 0.115, "i": 0.02, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        68.0,  # ~1 full period of the wide binary (P ~= 67.5 yr)
        0.136,
    ),
    (
        # Proxima Centauri b: real, confirmed planet around the nearest star
        # to the Sun. Proxima is itself the distant third member of the
        # Alpha Centauri triple (~8700 AU from the A-B pair, period on the
        # order of 10^5-10^6 years) -- omitted here exactly as Star A was
        # for 16 Cygni Bb: dynamically negligible for the planet, and at
        # that separation it would render as a static dot light-years off
        # scale while shrinking Proxima b's own orbit to an invisible speck.
        "proxima_centauri_b",
        "star_planet",
        [0.1221, 1.07 * M_EARTH],
        {
            0: None,
            1: {"a": 0.04856, "e": 0.02, "i": 0.05, "lan": 0.0, "aop": 0.0, "ta": 0.0},
        },
        0.306,  # ~10 orbital periods (P ~= 11.2 days)
        0.00102,
    ),
)

SOLVER_OVERRIDES = {}

# Per-system overrides for the template's generic default body names.
NAME_OVERRIDES = {
    "16_cygni_bb": ["16 Cygni B", "16 Cygni Bb"],
    "proxima_centauri_b": ["Proxima Centauri", "Proxima Centauri b"],
}

# Per-system full-title overrides, for systems whose real configuration (a
# wide binary/triple) isn't what's actually being rendered (see the comments
# on 16_cygni_bb and proxima_centauri_b in REAL_SYSTEMS above).
TITLE_OVERRIDES = {
    "16_cygni_bb": "16 Cygni Bb (S-type, companion not shown)",
    "proxima_centauri_b": "Proxima Centauri b (wide Alpha Cen AB pair not shown)",
}

def test_real_systems(output_dir=None):
    """
    Integrate every ``REAL_SYSTEMS`` entry through the hierarchy-template
    catalog, applying each system's ``NAME_OVERRIDES``/``TITLE_OVERRIDES``/
    ``SOLVER_OVERRIDES``. No numeric assertions -- illustrative, not a
    precision reference (see module docstring); the check is simply that
    every real system integrates without error. ``output_dir=None`` (the
    pytest default) skips the 3D trajectory plot/animation; passing a real
    directory (as ``main()`` does for standalone script execution) also
    saves them. Returns the list of every saved file path.
    """
    output_paths = []
    for name, key, masses, elements, tend, dt in REAL_SYSTEMS:
        template = HierarchicalSystemTemplates.get(key)
        solver_settings = {"tolrel": 1e-12, "tolabs": 1e-12, "maxord": 24, "minord": 6}
        solver_settings.update(SOLVER_OVERRIDES.get(name, {}))

        title = TITLE_OVERRIDES.get(name) or f"{name.replace('_', ' ').title()} ({template.title})"
        _, _, _, _, paths = solve_and_plot_hierarchy(
            name, key, masses, elements, tend, dt, title, output_dir,
            G=G_AU_MSUN_YR, names=NAME_OVERRIDES.get(name),
            animate=True, solver_settings=solver_settings,
        )
        for idx, path in enumerate(paths):
            label = "3D plot" if idx == 0 else "3D animation"
            print(f"{name}: {label} saved to {path}")
        output_paths.extend(paths)
    return output_paths


def main():
    test_real_systems(PYTHON_DIR)


if __name__ == "__main__":
    main()

