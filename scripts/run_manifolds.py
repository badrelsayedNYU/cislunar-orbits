"""Invariant manifolds of L1 and L2 Lyapunov orbits at a common Jacobi constant, cut at x = 1 - mu.

    python scripts/run_manifolds.py      (needs data/results/lyapunov_L*.npz)

* L1 Lyapunov: unstable manifold, the half that enters the Moon region, propagated forward.
* L2 Lyapunov: stable manifold, the half that comes from the Moon region, propagated backward.
Both are cut at the Poincare section x = 1 - mu (the Moon's x position), y < 0, where their
intersections in (y, vy) are planar heteroclinic connections from the L1 to the L2 orbit.

Checks: Jacobi drift of every trajectory (integration error) and the Jacobi offset introduced by
the step itself; sensitivity of the section curves to the step size d. Stretch goal: refine one
heteroclinic connection. Writes data/results/manifolds.json and data/results/manifolds.npz.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr3bp.correction import V_from_state, correct, jacobi_constraint, state_from_V  # noqa: E402
from cr3bp.manifolds import branch_sign, propagate_branch, seeds  # noqa: E402
from cr3bp.stability import analyze  # noqa: E402
from cr3bp.units import LU_KM, MU, VU_KMS, nd_to_days  # noqa: E402

OUT = ROOT / "data" / "results"
C_M = 3.15            # common Jacobi constant (between C(L2) = 3.1722 and the halo bifurcation)
N_SEEDS = 300
D_KM = 50.0           # default step-off distance
D_STUDY_KM = [5, 10, 25, 50, 100, 200, 500]
X_SECTION = 1 - MU
T_MAX = 8.0           # TU (~35 days) per trajectory


def lyapunov_at(which, C):
    d = np.load(OUT / f"lyapunov_L{which}.npz")
    k = int(np.argmin(np.abs(d["C"] - C)))
    c = correct(V_from_state(d["X0"][k], d["T"][k], True), MU, True,
                extra=jacobi_constraint(C, MU, True))
    return state_from_V(c.V, True), 2 * c.V[-1]


BRANCHES = {
    "L1_unstable": dict(which=1, kind="unstable", backward=False, toward_x_greater=True),
    "L2_stable": dict(which=2, kind="stable", backward=True, toward_x_greater=False),
}


def run_branch(spec, X0, T, d_km, n=N_SEEDS, keep_arcs=False):
    s = seeds(X0, T, MU, n, spec["kind"])
    d = d_km / LU_KM
    sign = branch_sign(s.X_orbit[0], s.v[0], d, MU, backward=spec["backward"],
                       toward_x_greater=spec["toward_x_greater"])
    sec, phase, drift, dC, impacts, arcs = [], [], [], [], {"moon": 0, "earth": 0}, []
    for k in range(n):
        a = propagate_branch(s.X_orbit[k] + sign * d * s.v[k], MU, backward=spec["backward"],
                             x_section=X_SECTION, t_max=T_MAX)
        dC.append(a.C0 - C_M)
        drift.append(a.C_drift)
        if a.impact:
            impacts[a.impact] += 1
        elif a.hit and a.X_section[1] < 0:
            sec.append(a.X_section)
            phase.append(s.t[k] / T)
        if keep_arcs:
            arcs.append(a.X[:, :3])
    return {
        "sign": sign, "lambda": s.lam, "section": np.array(sec), "phase": np.array(phase),
        "drift": np.array(drift), "dC_step": np.array(dC), "impacts": impacts, "arcs": arcs,
        "seeds": s,
    }


def curve_distance(A, B):
    """For each point of A (y, vy), the nearest point of polyline B: returns (|dy| km, |dvy| m/s).

    Nearest is measured in nondimensional (y, vy); B is ordered by phase.
    """
    out = []
    for p in A:
        best = (np.inf, 0.0, 0.0)
        for i in range(len(B) - 1):
            a, b = B[i], B[i + 1]
            ab = b - a
            if np.dot(ab, ab) == 0:
                continue
            u = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
            q = a + u * ab
            dist = np.linalg.norm(p - q)
            if dist < best[0]:
                best = (dist, abs(p[0] - q[0]), abs(p[1] - q[1]))
        out.append(best[1:])
    out = np.array(out)
    return out[:, 0] * LU_KM, out[:, 1] * VU_KMS * 1000


def polyline_intersections(P, Q):
    """Intersections of two polylines in the plane: list of (i, s, j, u, point)."""
    hits = []
    for i in range(len(P) - 1):
        p, r = P[i], P[i + 1] - P[i]
        for j in range(len(Q) - 1):
            q, s_ = Q[j], Q[j + 1] - Q[j]
            den = r[0] * s_[1] - r[1] * s_[0]
            if den == 0:
                continue
            w = q - p
            t = (w[0] * s_[1] - w[1] * s_[0]) / den
            u = (w[0] * r[1] - w[1] * r[0]) / den
            if 0 <= t <= 1 and 0 <= u <= 1:
                hits.append((i, t, j, u, p + t * r))
    return hits


def heteroclinic(orbits, res, d_km=D_KM):
    """Refine one L1 -> L2 connection: solve for phases (tau1, tau2) whose manifold trajectories
    meet at the section with equal (y, vy). Equal C (to the step offset) then fixes vx."""
    from scipy.optimize import fsolve

    from cr3bp.dynamics import propagate

    def section_state(which, tau):
        spec = BRANCHES["L1_unstable" if which == 1 else "L2_stable"]
        X0, T = orbits[which]
        r = res["L1_unstable" if which == 1 else "L2_stable"]
        arc = propagate(X0, tau * T, MU, stm=True) if tau > 0 else None
        Xo = arc.Xf if arc else X0
        Phi = arc.Phi if arc else np.eye(6)
        v0 = r["seeds"].v[0]
        v = Phi @ v0
        v /= np.linalg.norm(v[:3])
        a = propagate_branch(Xo + r["sign"] * d_km / LU_KM * v, MU, backward=spec["backward"],
                             x_section=X_SECTION, t_max=T_MAX)
        if not a.hit or a.impact:
            raise RuntimeError("no section crossing")
        return a

    # candidate intersections between the two section curves
    P = res["L1_unstable"]
    Q = res["L2_stable"]
    cands = []
    sp = P["section"][:, [1, 4]]
    sq = Q["section"][:, [1, 4]]
    for a_seg, a_ph in _segments_with_phase(sp, P["phase"]):
        for b_seg, b_ph in _segments_with_phase(sq, Q["phase"]):
            for i, t, j, u, pt in polyline_intersections(a_seg, b_seg):
                cands.append((a_ph[i] + t * (a_ph[i + 1] - a_ph[i]),
                              b_ph[j] + u * (b_ph[j + 1] - b_ph[j]), pt))
    if not cands:
        return {"found": False, "candidates": 0}

    def F(tau):
        a1 = section_state(1, tau[0] % 1.0)
        a2 = section_state(2, tau[1] % 1.0)
        return np.array([a1.X_section[1] - a2.X_section[1], a1.X_section[4] - a2.X_section[4]])

    tau0 = np.array(cands[0][:2])
    tau, info, ier, msg = fsolve(F, tau0, full_output=True, xtol=1e-13)
    a1 = section_state(1, tau[0] % 1.0)
    a2 = section_state(2, tau[1] % 1.0)
    dX = a1.X_section - a2.X_section
    return {
        "found": bool(ier == 1), "candidates": len(cands), "message": msg,
        "tau_L1": float(tau[0] % 1.0), "tau_L2": float(tau[1] % 1.0),
        "section_state": a1.X_section.tolist(),
        "position_mismatch_km": float(np.linalg.norm(dX[:3]) * LU_KM),
        "velocity_mismatch_mms": float(np.linalg.norm(dX[3:]) * VU_KMS * 1e6),
        "time_L1_to_section_days": float(nd_to_days(a1.t_section)),
        "time_section_to_L2_days": float(nd_to_days(-a2.t_section)),
        "C_L1_side": a1.C0, "C_L2_side": a2.C0,
        "arc_L1": a1.X[:, :3].tolist(), "arc_L2": a2.X[:, :3].tolist(),
    }


def _segments_with_phase(pts, phase, gap=3.0):
    if len(phase) < 2:
        return []
    cuts = np.where(np.diff(phase) * N_SEEDS > gap)[0] + 1
    return [(p, ph) for p, ph in zip(np.split(pts, cuts), np.split(phase, cuts)) if len(p) > 1]


def main():
    t0 = time.time()
    orbits = {w: lyapunov_at(w, C_M) for w in (1, 2)}
    summary = {"C": C_M, "x_section": X_SECTION, "n_seeds": N_SEEDS, "d_km": D_KM,
               "t_max_days": float(nd_to_days(T_MAX)), "orbits": {}, "branches": {}}
    for w, (X0, T) in orbits.items():
        a = analyze(X0, T, MU)
        summary["orbits"][f"L{w}"] = {"X0": X0.tolist(), "T": T, "T_days": float(nd_to_days(T)),
                                      "nu": a["nu"], "closure": a["closure"]}
    res = {}
    arrays = {}
    for name, spec in BRANCHES.items():
        X0, T = orbits[spec["which"]]
        r = run_branch(spec, X0, T, D_KM, keep_arcs=True)
        res[name] = r
        summary["branches"][name] = {
            "eigenvalue": r["lambda"], "section_points": int(len(r["section"])),
            "impacts": r["impacts"],
            "jacobi_drift_max": float(r["drift"].max()),
            "jacobi_drift_median": float(np.median(r["drift"])),
            "jacobi_step_offset_max": float(np.abs(r["dC_step"]).max()),
        }
        arrays[f"{name}_section"] = r["section"]
        arrays[f"{name}_phase"] = r["phase"]
        arrays[f"{name}_arcs"] = np.array(r["arcs"][::3])   # every third arc for figures
        print(name, summary["branches"][name])

    # step-size sensitivity: compare section curves for each d against d = 50 km
    study = {}
    for name, spec in BRANCHES.items():
        X0, T = orbits[spec["which"]]
        ref = res[name]
        ref_segs = _segments_with_phase(ref["section"][:, [1, 4]], ref["phase"])
        study[name] = {}
        for dk in D_STUDY_KM:
            r = ref if dk == D_KM else run_branch(spec, X0, T, dk, n=N_SEEDS)
            pts = r["section"][:, [1, 4]]
            if len(pts) == 0:
                study[name][str(dk)] = None
                continue
            # distance from each point to the nearest reference segment
            dy = np.full(len(pts), np.inf)
            dv = np.full(len(pts), np.inf)
            for seg, _ in ref_segs:
                a, b = curve_distance(pts, seg)
                better = a + b / 1000 < dy + dv / 1000
                dy = np.where(better, a, dy)
                dv = np.where(better, b, dv)
            study[name][str(dk)] = {
                "section_points": int(len(pts)), "impacts": r["impacts"],
                "dy_km_median": float(np.median(dy)), "dy_km_p95": float(np.percentile(dy, 95)),
                "dvy_mps_median": float(np.median(dv)), "dvy_mps_p95": float(np.percentile(dv, 95)),
                "jacobi_step_offset_max": float(np.abs(r["dC_step"]).max()),
                "jacobi_drift_max": float(r["drift"].max()),
            }
            print(name, dk, study[name][str(dk)])
    summary["step_size_study"] = study

    het = heteroclinic(orbits, res)
    summary["heteroclinic"] = {k: v for k, v in het.items() if not k.startswith("arc_")}
    if het.get("found"):
        arrays["heteroclinic_L1"] = np.array(het["arc_L1"])
        arrays["heteroclinic_L2"] = np.array(het["arc_L2"])
    print("heteroclinic", summary["heteroclinic"])
    for w, (X0, T) in orbits.items():
        from cr3bp.dynamics import propagate
        arrays[f"orbit_L{w}"] = propagate(X0, T, MU, t_eval=np.linspace(0, T, 400)).X[:, :3]
    summary["runtime_s"] = time.time() - t0
    np.savez_compressed(OUT / "manifolds.npz", **arrays)
    (OUT / "manifolds.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
