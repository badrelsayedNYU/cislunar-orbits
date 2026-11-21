"""Make every figure in docs/figures/ from data/results/ (run the run_*.py scripts first).

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr3bp.dynamics import propagate  # noqa: E402
from cr3bp.jpl import load_family  # noqa: E402
from cr3bp.units import LU_KM, MU, R_MOON_KM, VU_KMS  # noqa: E402

RES = ROOT / "data" / "results"
FIG = ROOT / "docs" / "figures"

# reference categorical palette (light), fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                              "#e87ba4", "#008300")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
    "axes.titlesize": 10, "axes.titleweight": "bold",
})

FAMS = [("lyapunov_L1", "L1 Lyapunov"), ("lyapunov_L2", "L2 Lyapunov"),
        ("halo_L1_S", "L1 halo (southern)"), ("halo_L2_S", "L2 halo (southern)")]


def load(name):
    return np.load(RES / f"{name}.npz")


def orbit_xyz(X0, T, rtol=1e-12, n=500):
    return propagate(X0, T, MU, t_eval=np.linspace(0, T, n), rtol=rtol, atol=rtol).X[:, :3]


def moon_sphere(ax, center_km=(0, 0, 0)):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    r = R_MOON_KM
    ax.plot_surface(center_km[0] + r * np.cos(u) * np.sin(v), center_km[1] + r * np.sin(u) * np.sin(v),
                    center_km[2] + r * np.cos(v), color="#b9b8b3", linewidth=0, alpha=0.9)


def fig_families():
    fig = plt.figure(figsize=(10, 4.6))
    for p, lp in enumerate(("L1", "L2")):
        ax = fig.add_subplot(1, 2, p + 1, projection="3d")
        for name, color, label in ((f"lyapunov_{lp}", BLUE, "Lyapunov"),
                                   (f"halo_{lp}_N", ORANGE, "halo, northern"),
                                   (f"halo_{lp}_S", AQUA, "halo, southern")):
            d = load(name)
            idx = np.linspace(0, len(d["T"]) - 1, 14 if "lyap" in name else 18).astype(int)
            if "lyap" in name:
                idx = idx[d["C"][idx] > 3.0]            # keep the plot near the Moon
            for j, i in enumerate(idx):
                xyz = (orbit_xyz(d["X0"][i], d["T"][i], d["rtol"][i]) - [1 - MU, 0, 0]) * LU_KM / 1e3
                ax.plot(*xyz.T, color=color, lw=0.7, label=label if j == 0 else None)
        moon_sphere_scaled(ax)
        ax.set_xlabel("x from Moon [10³ km]")
        ax.set_ylabel("y [10³ km]")
        ax.set_zlabel("z [10³ km]")
        ax.set_title(f"{lp} families (Earth-Moon CR3BP)")
        ax.view_init(elev=18, azim=-60)
        ax.set_box_aspect((1, 1, 1))
        _equal_3d(ax)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "families_3d.png")
    plt.close(fig)


def moon_sphere_scaled(ax):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    r = R_MOON_KM / 1e3
    ax.plot_surface(r * np.cos(u) * np.sin(v), r * np.sin(u) * np.sin(v), r * np.cos(v),
                    color="#8f8e89", linewidth=0, alpha=1.0)


def _equal_3d(ax):
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    c = lims.mean(axis=1)
    h = (lims[:, 1] - lims[:, 0]).max() / 2
    ax.set_xlim3d(c[0] - h, c[0] + h)
    ax.set_ylim3d(c[1] - h, c[1] + h)
    ax.set_zlim3d(c[2] - h, c[2] + h)


def fig_overlay():
    """Period and stability index vs Jacobi constant: our families (line) vs JPL (points)."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for k, (name, title) in enumerate(FAMS):
        d = load(name)
        cat = load_family(name)
        sub = np.arange(0, len(cat), max(1, len(cat) // 250))
        # only catalog points inside the C range we computed
        inside = (cat.jacobi >= d["C"].min()) & (cat.jacobi <= d["C"].max())
        sub = sub[inside[sub]]
        a0, a1 = axes[0, k], axes[1, k]
        a0.plot(d["C"], d["T"], color=BLUE, lw=1.6, label="this work", zorder=2)
        a0.scatter(cat.jacobi[sub], cat.period[sub], s=9, color=ORANGE, edgecolor="white",
                   linewidth=0.4, label="JPL catalog", zorder=3)
        a1.plot(d["C"], d["nu"], color=BLUE, lw=1.6, zorder=2)
        a1.scatter(cat.jacobi[sub], cat.stability[sub], s=9, color=ORANGE, edgecolor="white",
                   linewidth=0.4, zorder=3)
        a1.set_yscale("log")
        a0.set_title(title)
        a1.set_xlabel("Jacobi constant C")
        if k == 0:
            a0.set_ylabel("period T [TU]")
            a1.set_ylabel("stability index ν (log)")
            a0.legend(loc="best", fontsize=8)
    fig.suptitle("Computed families vs. the JPL Three-Body Periodic Orbit catalog", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "jpl_overlay.png")
    plt.close(fig)


def fig_errors():
    """Relative period error along each family against the interpolated catalog."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for k, (name, title) in enumerate(FAMS):
        d = load(name)
        e = np.abs(d["T"] - d[f"T_cat_{name}"]) / d[f"T_cat_{name}"]
        fold = d[f"near_fold_{name}"]
        i = np.arange(len(e))
        ok = np.isfinite(e)
        ax = axes[k]
        ax.semilogy(i[ok & ~fold], np.maximum(e[ok & ~fold], 1e-16), "o", ms=2.5, color=BLUE,
                    label="interpolated at matching C")
        if fold.any():
            ax.semilogy(i[ok & fold], np.maximum(e[ok & fold], 1e-16), "o", ms=3.5, color=ORANGE,
                        label="within ΔC = 1e-4 of a fold in C")
        s = json.loads((RES / f"{name}.json").read_text())["comparisons"][name]["pointwise"]
        ax.axhline(s["T_relerr_max"], color=INK2, lw=1, ls="--",
                   label="pointwise check, max over 60 orbits")
        ax.set_title(title)
        ax.set_xlabel("family member (continuation order)")
        if k == 0:
            ax.set_ylabel("|T - T_JPL| / T_JPL")
    h, l = axes[3].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "jpl_period_error.png")
    plt.close(fig)


def fig_nrho():
    n = json.loads((RES / "nrho.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.2), gridspec_kw={"width_ratios": [0.9, 1.6, 1.6]})
    for uname, color, label in (("jpl", BLUE, "JPL constants"), ("folta", ORANGE, "Folta et al. 2022 constants")):
        o = n["orbits"][uname]["9:2"][0]
        u = n["orbits"][uname]["constants"]
        X0 = np.array(o["X0"])
        X = propagate(X0, o["period_nd"], u["mu"], t_eval=np.linspace(0, o["period_nd"], 3000),
                      rtol=1e-13, atol=1e-13).X
        xyz = (X[:, :3] - [1 - u["mu"], 0, 0]) * u["LU_km"] / 1e3
        lab = f"{label}: r_p = {o['perilune_radius_km']:.0f} km"
        for ax, (i, j) in zip(axes, ((0, 2), (1, 2), (0, 1))):
            ax.plot(xyz[:, i], xyz[:, j], color=color, lw=1.4, label=lab)
    names = "xyz"
    for ax, (i, j) in zip(axes, ((0, 2), (1, 2), (0, 1))):
        ax.add_patch(plt.Circle((0, 0), R_MOON_KM / 1e3, color="#8f8e89", zorder=3))
        ax.set_aspect("equal")
        ax.set_xlabel(f"{names[i]} from Moon [10³ km]")
        ax.set_ylabel(f"{names[j]} [10³ km]")
    axes[0].set_title("x-z (rotating frame)")
    axes[1].set_title("y-z")
    axes[2].set_title("x-y")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5)
    fig.suptitle("9:2 synodic-resonant L2 southern NRHO, period 6.562 d (CR3BP)", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "nrho_9_2.png")
    plt.close(fig)


def fig_manifolds():
    m = np.load(RES / "manifolds.npz")
    s = json.loads((RES / "manifolds.json").read_text())
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    for key, color, label in (("L1_unstable", BLUE, "L1 Lyapunov, unstable manifold (forward)"),
                              ("L2_stable", ORANGE, "L2 Lyapunov, stable manifold (backward)")):
        arcs = m[f"{key}_arcs"]
        for j, arc in enumerate(arcs):
            a0.plot((arc[:, 0] - (1 - MU)) * LU_KM / 1e3, arc[:, 1] * LU_KM / 1e3, color=color,
                    lw=0.35, alpha=0.6)
        a0.plot([], [], color=color, lw=2, label=label)
        sec = m[f"{key}_section"]
        a1.plot(sec[:, 1] * LU_KM / 1e3, sec[:, 4] * VU_KMS, "o", ms=2.5, color=color, label=label)
    for w in (1, 2):
        o = m[f"orbit_L{w}"]
        a0.plot((o[:, 0] - (1 - MU)) * LU_KM / 1e3, o[:, 1] * LU_KM / 1e3, color=INK, lw=1.4,
                label="Lyapunov orbits" if w == 1 else None)
    if "heteroclinic_L1" in m:
        for k in ("heteroclinic_L1", "heteroclinic_L2"):
            h = m[k]
            a0.plot((h[:, 0] - (1 - MU)) * LU_KM / 1e3, h[:, 1] * LU_KM / 1e3, color=GREEN, lw=1.8,
                    label="heteroclinic connection" if k.endswith("L1") else None)
        hs = s["heteroclinic"]["section_state"]
        a1.plot(hs[1] * LU_KM / 1e3, hs[4] * VU_KMS, "*", ms=13, color=GREEN,
                markeredgecolor="white", label="heteroclinic connection", zorder=5)
    a0.add_patch(plt.Circle((0, 0), R_MOON_KM / 1e3, color="#8f8e89", zorder=4))
    a0.axvline(0, color=INK2, lw=0.8, ls="--")
    a0.text(1.5, 52, "section x = 1 - μ", color=INK2, fontsize=8)
    a0.set_aspect("equal")
    a0.set_xlim(-80, 80)
    a0.set_ylim(-65, 65)
    a0.set_xlabel("x from Moon [10³ km]")
    a0.set_ylabel("y [10³ km]")
    a0.set_title(f"Manifold tubes at C = {s['C']}")
    a0.legend(fontsize=7, loc="upper left")
    a1.set_xlabel("y at x = 1 - μ [10³ km]")
    a1.set_ylabel("ẏ [km/s]")
    a1.set_title("Poincaré section x = 1 - μ, y < 0")
    a1.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "manifolds.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_families, fig_overlay, fig_errors, fig_nrho, fig_manifolds):
        f()
        print("made", f.__name__)


if __name__ == "__main__":
    main()
