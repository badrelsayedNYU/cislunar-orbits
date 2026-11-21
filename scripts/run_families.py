"""Compute the Earth-Moon L1/L2 Lyapunov and halo families and compare them with the JPL catalog.

    python scripts/run_families.py                 # everything (~ several minutes)
    python scripts/run_families.py --only lyapunov

Writes data/results/<family>.npz (orbits + diagnostics) and data/results/<family>.json
(summary numbers read by scripts/make_results.py).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr3bp.continuation import (Family, continue_family, find_tangent_bifurcation,  # noqa: E402
                                lyapunov_linear_guess)
from cr3bp.correction import V_from_state  # noqa: E402
from cr3bp.dynamics import propagate  # noqa: E402
from cr3bp.jpl import FOLD_DC, compare_family, load_family, pointwise_compare  # noqa: E402
from cr3bp.stability import analyze  # noqa: E402
from cr3bp.units import MU, nd_to_km  # noqa: E402

OUT = ROOT / "data" / "results"


def perilune_apolune(X0, T, rtol):
    """Minimum and maximum distance from the Moon over one period (dense sampling + refinement)."""
    arc = propagate(X0, T, MU, dense=True, rtol=rtol, atol=rtol)
    t = np.linspace(0, T, 4001)
    X = arc.sol(t).T
    r = np.linalg.norm(X[:, :3] - np.array([1 - MU, 0, 0]), axis=1)
    from scipy.optimize import minimize_scalar

    def rmoon(tt):
        return np.linalg.norm(arc.sol(tt)[:3] - np.array([1 - MU, 0, 0]))

    i, j = np.argmin(r), np.argmax(r)
    lo = minimize_scalar(rmoon, bounds=(t[max(i - 1, 0)], t[min(i + 1, len(t) - 1)]),
                         method="bounded", options={"xatol": 1e-12}).fun
    hi = -minimize_scalar(lambda tt: -rmoon(tt), bounds=(t[max(j - 1, 0)], t[min(j + 1, len(t) - 1)]),
                          method="bounded", options={"xatol": 1e-12}).fun
    return lo, hi


CLOSURE_REFINE = 5e-11   # re-correct at tighter tolerance above this full-period closure
TIGHT = 1e-13


def refine(fam: Family):
    """Re-correct members whose full-period closure exceeds CLOSURE_REFINE at rtol = atol = 1e-13.

    The minimum-norm Newton step from the stored solution moves the orbit by ~1e-12, so the
    member stays at the same place on the family.
    """
    from cr3bp.correction import correct, state_from_V
    from cr3bp.dynamics import jacobi
    from cr3bp.stability import monodromy

    n_refined = 0
    for o in fam.orbits:
        _, XT = monodromy(o.X0, o.T, MU, o.rtol, o.rtol)
        if np.linalg.norm(XT - o.X0) <= CLOSURE_REFINE or o.rtol <= TIGHT:
            continue
        c = correct(V_from_state(o.X0, o.T, fam.planar), MU, fam.planar, rtol=TIGHT, atol=TIGHT)
        o.X0 = state_from_V(c.V, fam.planar)
        o.T = 2 * c.V[-1]
        o.C = float(jacobi(o.X0, MU))
        o.residual = c.residual
        o.rtol = TIGHT
        n_refined += 1
    fam.log.append(f"{n_refined} members re-corrected at rtol = atol = {TIGHT:g} "
                   f"(closure > {CLOSURE_REFINE:g} at the default tolerance)")
    return n_refined


def postprocess(fam: Family, cat_names):
    refine(fam)
    a = fam.arrays()
    n = len(fam.orbits)
    keys = ["closure", "nu", "s_z", "pair_defect", "det"]
    diag = {k: np.empty(n) for k in keys}
    s1 = np.empty(n, complex)
    s2 = np.empty(n, complex)
    Xhalf = np.empty((n, 6))
    rp = np.empty(n)
    ra = np.empty(n)
    for i, o in enumerate(fam.orbits):
        d = analyze(o.X0, o.T, MU, rtol=o.rtol)
        for k in keys:
            diag[k][i] = d[k]
        s1[i], s2[i] = d["s1"], d["s2"]
        Xhalf[i] = propagate(o.X0, o.T / 2, MU, rtol=o.rtol, atol=o.rtol).Xf
        rp[i], ra[i] = perilune_apolune(o.X0, o.T, o.rtol)
    data = dict(a, Xhalf=Xhalf, s1=s1, s2=s2, r_perilune=rp, r_apolune=ra, **diag)
    summary = {
        "family": fam.name,
        "members": n,
        "C_range": [float(a["C"].min()), float(a["C"].max())],
        "T_range": [float(a["T"].min()), float(a["T"].max())],
        "max_closure": float(diag["closure"].max()),
        "max_half_period_residual": float(a["residual"].max()),
        "max_pair_defect": float(diag["pair_defect"].max()),
        "max_det_error": float(np.max(np.abs(diag["det"] - 1))),
        "rtol_counts": {f"{r:g}": int(np.sum(a["rtol"] == r)) for r in sorted(set(a["rtol"]))},
        "min_perilune_km": float(nd_to_km(rp.min())),
        "continuation_log": fam.log,
        "switched_to_pac_at": fam.switched_to_pac_at,
        "methods": {m: int(np.sum(a["method"] == m)) for m in sorted(set(a["method"]))},
        "comparisons": {},
    }
    for cname in cat_names:
        cat = load_family(cname)
        cmp = compare_family(cat, a["X0"], Xhalf, a["C"], a["T"], diag["nu"])
        ok = np.isfinite(cmp["T_relerr"])
        away = ok & ~cmp["near_fold"]
        data[f"T_cat_{cname}"] = cmp["T_cat"]
        data[f"nu_cat_{cname}"] = cmp["nu_cat"]
        data[f"near_fold_{cname}"] = cmp["near_fold"]
        # pointwise check at 60 catalog orbits spread over the part of the catalog that lies
        # within the computed family: mapped to an interior member and within 0.01 LU in state
        # space (about twice the largest spacing between consecutive computed members)
        inside = np.where((cmp["catalog_position"] > 0) & (cmp["catalog_position"] < n - 1)
                          & (cmp["catalog_ic_distance"] < 0.01))[0]
        idx = inside[np.linspace(0, len(inside) - 1, 60).astype(int)]
        pw, pw_failed = pointwise_compare(cat, a["X0"], Xhalf, a["T"], fam.planar, idx)
        pwT = np.array([abs(r["T"] - r["T_cat"]) / r["T_cat"] for r in pw])
        pwC = np.array([abs(r["C"] - r["C_cat"]) for r in pw])
        pwN = np.array([abs(r["nu"] - r["nu_cat"]) / r["nu_cat"] for r in pw])
        pwX = np.array([r["ic_difference"] for r in pw])
        summary["comparisons"][cname] = {
            "n_compared_excluding_folds": int(away.sum()),
            "C_folds": [float(c) for c in cmp["C_folds"]],
            "fold_exclusion_dC": FOLD_DC,
            "T_relerr_max_excluding_folds": float(np.max(cmp["T_relerr"][away])),
            "nu_relerr_max_excluding_folds": float(np.max(cmp["nu_relerr"][away])),
            "catalog_orbits_within_family": int(len(inside)),
            "pointwise": {
                "n": len(pw),
                "failed": pw_failed,
                "T_relerr_max": float(pwT.max()), "T_relerr_median": float(np.median(pwT)),
                "C_abserr_max": float(pwC.max()), "C_abserr_median": float(np.median(pwC)),
                "nu_relerr_max": float(pwN.max()), "nu_relerr_median": float(np.median(pwN)),
                "ic_difference_max": float(pwX.max()),
                "C_range": [min(r["C_cat"] for r in pw), max(r["C_cat"] for r in pw)],
                "fixed_variables": sorted({r["fixed_variable"] for r in pw}),
            },
            "catalog_count": len(cat),
            "catalog_fetched_utc": cat.fetched_utc,
            "n_compared": int(ok.sum()),
            "C_compared": [float(a["C"][ok].min()), float(a["C"][ok].max())] if ok.any() else None,
            "T_relerr_max": float(np.nanmax(cmp["T_relerr"])),
            "T_relerr_median": float(np.nanmedian(cmp["T_relerr"])),
            "nu_relerr_max": float(np.nanmax(cmp["nu_relerr"])),
            "nu_relerr_median": float(np.nanmedian(cmp["nu_relerr"])),
            "catalog_ic_distance_median": float(np.median(cmp["catalog_ic_distance"])),
        }
    return data, summary


def save(name, data, summary):
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", **data)
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=1, default=float))


def run_lyapunov(which):
    name = f"lyapunov_L{which}"
    cat = load_family(name)
    t0 = time.time()
    X0, T = lyapunov_linear_guess(MU, which, 1e-3)   # Ax = 1e-3 LU (390 km)
    fam = Family(name, MU, planar=True)
    step = -2e-3 if which == 1 else 2e-3             # grow away from the libration point in x0
    continue_family(fam, V_from_state(X0, T, True), n_max=2000, natural_step=step / 4,
                    natural_step_max=abs(step), ds=5e-3, ds_max=5e-3,
                    stop=lambda o, V: o.C < cat.jacobi.min() - 0.005, verbose=True)
    data, summary = postprocess(fam, [name])
    summary["runtime_s"] = time.time() - t0

    # halo bifurcation: out-of-plane pair passes through +1 (s_z = 2)
    sz = data["s_z"]
    idx = np.where(np.diff(np.sign(sz - 2.0)) != 0)[0]
    bif = []
    for i in idx:
        Va = V_from_state(fam.orbits[i].X0, fam.orbits[i].T, True)
        Vb = V_from_state(fam.orbits[i + 1].X0, fam.orbits[i + 1].T, True)
        Vbif, resid = find_tangent_bifurcation(MU, Va, Vb)
        from cr3bp.correction import state_from_V
        from cr3bp.dynamics import jacobi
        Xb = state_from_V(Vbif, True)
        d = analyze(Xb, 2 * Vbif[-1], MU)
        bif.append({"index": int(i), "X0": Xb.tolist(), "T": float(2 * Vbif[-1]),
                    "C": float(jacobi(Xb, MU)), "s_z_minus_2": float(resid), "nu": d["nu"],
                    "closure": d["closure"]})
    summary["bifurcations"] = bif
    save(name, data, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "continuation_log"}, indent=1,
                     default=float))
    return fam, data, summary


def moon_distance_min(V, planar=False, n=1500):
    """Closest approach to the Moon over a half period (by symmetry, over the whole orbit)."""
    from cr3bp.correction import state_from_V

    X = state_from_V(V, planar)
    a = propagate(X, V[-1], MU, t_eval=np.linspace(0, V[-1], n))
    return float(np.min(np.linalg.norm(a.X[:, :3] - np.array([1 - MU, 0, 0]), axis=1)))


R_TIGHT = 0.05   # LU (~19,500 km): orbits passing closer to the Moon use rtol = atol = 1e-13


def run_halo(which):
    """Northern halo branch from the Lyapunov bifurcation; southern branch by z -> -z."""
    from cr3bp.correction import natural_constraint
    from cr3bp.units import R_MOON_KM, km_to_nd

    lyap = json.loads((OUT / f"lyapunov_L{which}.json").read_text())
    b = lyap["bifurcations"][0]
    t0 = time.time()
    V = V_from_state(np.array(b["X0"]), b["T"], False)
    z_step = 1e-4
    V[1] = z_step                                   # step off the planar family in +z (north)
    fam = Family(f"halo_L{which}_N", MU, planar=False)
    r_stop = km_to_nd(R_MOON_KM)
    continue_family(
        fam, V, n_max=3000, natural_index="auto", natural_step=1e-3, natural_step_max=4e-3,
        max_natural=10, start_extra=natural_constraint(1, z_step), ds=4e-3, ds_max=4e-3,
        rtol_fn=lambda V: 1e-13 if moon_distance_min(V) < R_TIGHT else 1e-12,
        stop=lambda o, V: moon_distance_min(V) < r_stop, verbose=True)
    fam.log.append(f"stopped when the perilune radius fell below the lunar radius "
                   f"({R_MOON_KM} km)")
    north, sn = postprocess(fam, [f"halo_L{which}_N"])
    sn["runtime_s"] = time.time() - t0
    sn["bifurcation_C"] = b["C"]
    save(fam.name, north, sn)

    # southern branch: (x, y, z, vx, vy, vz) -> (x, y, -z, vx, vy, -vz) maps solutions to solutions
    south_fam = Family(f"halo_L{which}_S", MU, planar=False, log=list(fam.log))
    mirror = np.diag([1, 1, -1, 1, 1, -1.0])
    from cr3bp.continuation import Orbit
    for o in fam.orbits:
        south_fam.orbits.append(Orbit(X0=mirror @ o.X0, T=o.T, C=o.C, residual=o.residual,
                                      rtol=o.rtol, method=o.method))
    south, ss = postprocess(south_fam, [f"halo_L{which}_S"])
    ss["derived_from"] = fam.name + " by z -> -z symmetry (each mirrored orbit re-integrated)"
    ss["bifurcation_C"] = b["C"]
    save(south_fam.name, south, ss)
    for summ in (sn, ss):
        print(json.dumps({k: v for k, v in summ.items() if k != "continuation_log"}, indent=1,
                         default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["lyapunov", "halo"], default=None)
    args = ap.parse_args()
    if args.only in (None, "lyapunov"):
        for which in (1, 2):
            run_lyapunov(which)
    if args.only in (None, "halo"):
        for which in (1, 2):
            run_halo(which)


if __name__ == "__main__":
    main()
