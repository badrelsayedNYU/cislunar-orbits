"""Read the cached JPL Three-Body Periodic Orbits catalog (data/jpl/*.json).

Nothing here touches the network; scripts/fetch_jpl.py fills the cache.
"""

from __future__ import annotations

import functools
import json
import pathlib
from dataclasses import dataclass

import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "jpl"

FAMILIES = ("lyapunov_L1", "lyapunov_L2", "halo_L1_N", "halo_L1_S", "halo_L2_N", "halo_L2_S")


@dataclass(frozen=True)
class CatalogFamily:
    name: str
    states: np.ndarray      # (N, 6) initial conditions [x, y, z, vx, vy, vz], nondimensional
    jacobi: np.ndarray      # (N,)
    period: np.ndarray      # (N,) nondimensional
    stability: np.ndarray   # (N,) JPL stability index
    system: dict            # raw "system" block of the API response
    fetched_utc: str
    request_url: str

    def __len__(self) -> int:
        return len(self.jacobi)


@functools.lru_cache(maxsize=None)
def load_family(name: str) -> CatalogFamily:
    if name not in FAMILIES:
        raise KeyError(f"unknown family {name!r}; choose from {FAMILIES}")
    raw = json.loads((DATA_DIR / f"{name}.json").read_text())
    resp = raw["response"]
    fields = resp["fields"]
    table = np.array(resp["data"], dtype=float)
    col = {f: table[:, i] for i, f in enumerate(fields)}
    states = np.column_stack([col[k] for k in ("x", "y", "z", "vx", "vy", "vz")])
    return CatalogFamily(
        name=name,
        states=states,
        jacobi=col["jacobi"],
        period=col["period"],
        stability=col["stability"],
        system=resp["system"],
        fetched_utc=raw["fetched_utc"],
        request_url=raw["request_url"],
    )


@functools.lru_cache(maxsize=None)
def system() -> dict:
    """Earth-Moon system constants as floats, taken from the cached API response.

    All six cached families must agree; this is checked in the tests.
    """
    s = load_family("halo_L2_S").system
    out = {
        "name": s["name"],
        "mu": float(s["mass_ratio"]),
        "lunit_km": float(s["lunit"]),
        "tunit_s": float(s["tunit"]),
        "radius_moon_km": float(s["radius_secondary"]),
    }
    for k in ("L1", "L2", "L3", "L4", "L5"):
        out[k] = np.array([float(v) for v in s[k]])
    return out


# ---------------------------------------------------------------------------------------------
# Comparison of a computed family with the catalog at matching Jacobi constant
# ---------------------------------------------------------------------------------------------

def _monotone_runs(C):
    """Index ranges [i0, i1] over which C is strictly monotone along the family."""
    d = np.sign(np.diff(C))
    runs, start = [], 0
    for i in range(1, len(d)):
        if d[i] != d[start]:
            runs.append((start, i))
            start = i
    runs.append((start, len(C) - 1))
    return runs


FOLD_DC = 1e-4   # orbits with |C - C_fold| < FOLD_DC are flagged as fold neighbours


def compare_family(cat: CatalogFamily, X0, Xhalf, C, T, nu, *, min_run=5):
    """Interpolate the catalog's period and stability index at each computed orbit's Jacobi constant.

    The JPL catalog is sorted by Jacobi constant, not by position along the family, and C is not
    monotone along every family (the L2 halo family turns back near the NRHOs), so period(C)
    can have two branches. The computed family (ordered by continuation) is used only to place
    each catalog orbit along the family: a catalog orbit is assigned the arclength position of the
    nearest computed initial condition (the catalog may start an orbit at either x-z crossing, so
    both X0 and X(T/2) are tried). Within each run of the computed family over which C is
    monotone, the catalog orbits that fall in that run are interpolated in C with a cubic spline.
    Computed orbits whose C lies outside the catalog's range on their run are skipped.

    Returns a dict of arrays aligned with the computed family (NaN where not compared).
    """
    from scipy.interpolate import CubicSpline

    n = len(C)
    # nearest computed member for every catalog orbit
    pos = np.empty(len(cat))
    dmin = np.empty(len(cat))
    for j0 in range(0, len(cat), 512):
        S = cat.states[j0:j0 + 512]
        d0 = np.linalg.norm(S[:, None, :] - X0[None, :, :], axis=2)
        d1 = np.linalg.norm(S[:, None, :] - Xhalf[None, :, :], axis=2)
        d = np.minimum(d0, d1)
        k = np.argmin(d, axis=1)
        pos[j0:j0 + 512] = k
        dmin[j0:j0 + 512] = d[np.arange(len(k)), k]

    T_cat = np.full(n, np.nan)
    nu_cat = np.full(n, np.nan)
    runs = _monotone_runs(C)
    # turning points of C along the family (interior run boundaries): dT/dC is unbounded there,
    # so interpolation in C is ill-conditioned nearby
    C_folds = np.array([C[i1] for (_, i1) in runs[:-1]])
    near_fold = np.zeros(n, bool)
    if len(C_folds):
        near_fold = np.min(np.abs(C[:, None] - C_folds[None, :]), axis=1) < FOLD_DC
    for i0, i1 in runs:
        if i1 - i0 + 1 < min_run:
            continue
        # catalog orbits beyond either end of the computed family all map to the end members;
        # leave them out so they cannot mix in a different branch
        sel = (pos >= i0) & (pos <= i1) & (pos > 0) & (pos < n - 1)
        if sel.sum() < 4:
            continue
        Cc, Tc, Nc = cat.jacobi[sel], cat.period[sel], cat.stability[sel]
        o = np.argsort(Cc)
        Cc, Tc, Nc = Cc[o], Tc[o], Nc[o]
        keep = np.r_[True, np.diff(Cc) > 0]
        Cc, Tc, Nc = Cc[keep], Tc[keep], Nc[keep]
        if len(Cc) < 4:
            continue
        sT, sN = CubicSpline(Cc, Tc), CubicSpline(Cc, Nc)
        # skip the run's two end members (fold neighborhoods) and anything outside the data
        for i in range(i0 + 1, i1):
            if Cc[0] <= C[i] <= Cc[-1]:
                T_cat[i] = sT(C[i])
                nu_cat[i] = sN(C[i])
    return {
        "T_cat": T_cat,
        "nu_cat": nu_cat,
        "T_relerr": np.abs(T - T_cat) / T_cat,
        "nu_relerr": np.abs(nu - nu_cat) / nu_cat,
        "catalog_ic_distance": dmin,
        "catalog_position": pos,
        "near_fold": near_fold,
        "C_folds": C_folds,
    }


def pointwise_compare(cat: CatalogFamily, X0_fam, Xhalf_fam, T_fam, planar, idx, *, rtol=1e-13):
    """Direct comparison with individual catalog orbits, without interpolation.

    For catalog orbit j: start from the nearest computed family member (represented from whichever
    perpendicular x-z crossing, X0 or X(T/2), is nearer to the catalog IC), pick the free variable
    (x0, z0 or vy0) with the largest component of the local family tangent, fix it to the catalog
    value, and correct. The result is our orbit at the catalog's location on the family; its
    period, Jacobi constant and stability index are compared with the catalog's. Only one
    coordinate of the catalog IC is used (as the target of the fixed variable), and the method
    works at folds in C. Returns (list of result dicts, number of failed corrections).
    """
    from .correction import (CorrectionError, V_from_state, correct, family_tangent,
                             natural_constraint, state_from_V)
    from .dynamics import jacobi
    from .stability import analyze
    from .units import MU

    out, failed = [], 0
    for j in idx:
        Xc = cat.states[j]
        d0 = np.linalg.norm(X0_fam - Xc, axis=1)
        d1 = np.linalg.norm(Xhalf_fam - Xc, axis=1)
        if d0.min() <= d1.min():
            k_near = int(np.argmin(d0))
            Xs = X0_fam[k_near]
        else:
            k_near = int(np.argmin(d1))
            Xs = Xhalf_fam[k_near]
        try:
            V = V_from_state(Xs, T_fam[k_near], planar)
            c0 = correct(V, MU, planar, rtol=rtol, atol=rtol)
            tan = family_tangent(c0.DF)
            k = int(np.argmax(np.abs(tan[:-1])))
            target = V_from_state(Xc, cat.period[j], planar)[k]
            Vg = c0.V + (target - c0.V[k]) / tan[k] * tan
            c = correct(Vg, MU, planar, extra=natural_constraint(k, target), rtol=rtol, atol=rtol)
        except (CorrectionError, np.linalg.LinAlgError, RuntimeError):
            failed += 1
            continue
        X = state_from_V(c.V, planar)
        T = 2 * c.V[-1]
        d = analyze(X, T, MU, rtol=rtol)
        out.append({
            "catalog_index": int(j),
            "fixed_variable": ["x0", "vy0"][k] if planar else ["x0", "z0", "vy0"][k],
            "T": T, "T_cat": float(cat.period[j]),
            "C": float(jacobi(X, MU)), "C_cat": float(cat.jacobi[j]),
            "nu": d["nu"], "nu_cat": float(cat.stability[j]),
            "ic_difference": float(np.linalg.norm(X - Xc)),
            "closure": d["closure"],
        })
    return out, failed
