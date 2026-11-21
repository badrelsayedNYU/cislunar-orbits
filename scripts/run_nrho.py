"""The 9:2 lunar-synodic-resonant L2 southern NRHO, plus cross-checks against published CR3BP data.

    python scripts/run_nrho.py      (needs data/results/halo_L2_S.npz from run_families.py)

1. Solve the L2 southern halo member whose period is (p/q) x 29.530589 days for the 9:2 (Gateway),
   4:1 and 3:1 synodic resonances. The period in days must be converted to nondimensional time,
   and that conversion depends on the characteristic time unit, so each orbit is solved twice:
     - "jpl":   JPL catalog constants (mu, LU = 389,703.26 km, TU = 382,981.29 s), the project's
                default and the constants used for every catalog comparison;
     - "folta": Folta et al. (2022), NASA/TP-20220014814, Table 2 (mu = 1.215058535056245e-2,
                l = 384,400 km, t = 375,190.2588926273 s), the conventional values in the
                published CR3BP NRHO literature.
   The period is the fixed variable in the correction (T/2 constrained); the starting guess is
   the bracketing member of the computed family.
2. Folta et al. (2022) Table 3: published period, Jacobi constant and initial state of selected
   L1/L2 Lyapunov and halo orbits. Our orbit is corrected at the published period (starting from
   our own family, never from the published state) and its Jacobi constant and state compared.

Writes data/results/nrho.json and data/results/folta_table3.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr3bp.correction import V_from_state, correct, natural_constraint, state_from_V  # noqa: E402
from cr3bp.dynamics import apses, jacobi, propagate  # noqa: E402
from cr3bp.stability import analyze  # noqa: E402
from cr3bp import units  # noqa: E402

OUT = ROOT / "data" / "results"
RTOL = 1e-13
SYNODIC = units.SYNODIC_MONTH_DAYS

UNIT_SETS = {
    "jpl": {"mu": units.MU, "LU_km": units.LU_KM, "TU_s": units.TU_S,
            "source": "JPL Three-Body Periodic Orbits API (cached response, data/jpl/)"},
    "folta": {"mu": 1.215058535056245e-2, "LU_km": 384400.0, "TU_s": 375190.2588926273,
              "source": "Folta et al. (2022), NASA/TP-20220014814, Table 2"},
}
RESONANCES = {"9:2": 2 / 9, "4:1": 1 / 4, "3:1": 1 / 3}

# Folta et al. (2022) Table 3 rows for the families computed here (Earth-Moon CR3BP, 6 decimals).
FOLTA_TABLE3 = [
    ("lyapunov_L1", 3.071168, 3.107961, [0.807303, 0, 0, 0, 0.298948, 0]),
    ("halo_L1_N", 2.760344, 3.151498, [0.823969, 0, 0.053194, 0, 0.163217, 0]),
    ("halo_L1_N", 1.868528, 3.003577, [0.906618, 0, 0.203669, 0, 0.169171, 0]),
    ("lyapunov_L2", 3.727062, 3.072614, [1.062267, 0, 0, 0, 0.470321, 0]),
    ("halo_L2_N", 3.366323, 3.128090, [1.174193, 0, 0.076230, 0, -0.182432, 0]),
    ("halo_L2_N", 2.269175, 3.015746, [1.075397, 0, 0.202158, 0, -0.192618, 0]),
]


def solve_fixed_period(V_guess, T, mu, planar=False):
    c = correct(V_guess, mu, planar, extra=natural_constraint(len(V_guess) - 1, T / 2),
                rtol=RTOL, atol=RTOL)
    return c


def describe(X0, T, mu, LU_km, TU_s, R_moon_km):
    rp, ra, tp, ta = apses(X0, T, mu, rtol=RTOL)
    d = analyze(X0, T, mu, rtol=RTOL)
    arc = propagate(X0, T, mu, t_eval=np.linspace(0, T, 4001), rtol=RTOL, atol=RTOL)
    Xp = propagate(X0, tp, mu, rtol=RTOL, atol=RTOL).Xf if tp > 0 else X0
    v_peri = np.linalg.norm(Xp[3:]) * LU_km / TU_s
    return {
        "X0": X0.tolist(),
        "period_nd": T,
        "period_days": T * TU_s / 86400,
        "jacobi": float(jacobi(X0, mu)),
        "perilune_radius_km": rp * LU_km,
        "perilune_altitude_km": rp * LU_km - R_moon_km,
        "apolune_radius_km": ra * LU_km,
        "perilune_speed_rotating_kms": v_peri,
        "z_min_km": float(arc.X[:, 2].min() * LU_km),
        "z_max_km": float(arc.X[:, 2].max() * LU_km),
        "stability_index": d["nu"],
        "broucke_s": [str(np.round(d["s1"], 10)), str(np.round(d["s2"], 10))],
        "eigenvalue_moduli": sorted(float(abs(l)) for l in d["eigenvalues"]),
        "closure": d["closure"],
        "pair_defect": d["pair_defect"],
    }


def nrho():
    fam = np.load(OUT / "halo_L2_S.npz")
    X0f, Tf = fam["X0"], fam["T"]
    out = {"synodic_month_days": SYNODIC, "integration_rtol_atol": RTOL,
           "radius_moon_km": units.R_MOON_KM, "family": "halo_L2_S", "orbits": {}}
    for uname, u in UNIT_SETS.items():
        out["orbits"][uname] = {"constants": u}
        for rname, frac in RESONANCES.items():
            T_days = frac * SYNODIC
            T = T_days * 86400 / u["TU_s"]
            idx = np.where(np.diff(np.sign(Tf - T)) != 0)[0]
            sols = []
            for i in idx:
                # guess: member i, period nudged by the predictor of the fixed-period constraint
                c = solve_fixed_period(V_from_state(X0f[i], Tf[i], False), T, u["mu"])
                X0 = state_from_V(c.V, False)
                s = describe(X0, T, u["mu"], u["LU_km"], u["TU_s"], units.R_MOON_KM)
                s["target_period_days"] = T_days
                s["family_bracket"] = [int(i), int(i + 1)]
                sols.append(s)
            out["orbits"][uname][rname] = sols
            for s in sols:
                print(f"{uname:5s} {rname}: T = {s['period_days']:.4f} d ({s['period_nd']:.6f} TU), "
                      f"r_p = {s['perilune_radius_km']:.1f} km (alt {s['perilune_altitude_km']:.1f}), "
                      f"r_a = {s['apolune_radius_km']:.1f} km, C = {s['jacobi']:.6f}, "
                      f"nu = {s['stability_index']:.4f}, closure = {s['closure']:.1e}")
    (OUT / "nrho.json").write_text(json.dumps(out, indent=1))


def folta_table3():
    rows = []
    mu = units.MU
    for name, T_pub, C_pub, X_pub in FOLTA_TABLE3:
        planar = name.startswith("lyapunov")
        fam = np.load(OUT / f"{name}.npz")
        X_pub = np.array(X_pub, dtype=float)
        # nearest member of our family (either x-z crossing) to the published state picks the
        # branch where a period occurs more than once; the published state is not used otherwise
        reps = [(fam["X0"], "X0"), (fam["Xhalf"], "Xhalf")]
        best = min(((np.linalg.norm(R - X_pub, axis=1).min(), int(np.argmin(np.linalg.norm(R - X_pub, axis=1))), R)
                    for R, _ in reps), key=lambda t: t[0])
        _, k, R = best
        c = solve_fixed_period(V_from_state(R[k], fam["T"][k], planar), T_pub, mu, planar)
        X = state_from_V(c.V, planar)
        C = float(jacobi(X, mu))
        rows.append({"family": name, "T_published": T_pub, "C_published": C_pub,
                     "X0_published": X_pub.tolist(), "C_ours": C, "X0_ours": X.tolist(),
                     "C_diff": C - C_pub, "max_state_diff": float(np.max(np.abs(X - X_pub)))})
        print(f"{name:12s} T = {T_pub}: C ours {C:.7f} vs published {C_pub:.6f} "
              f"(diff {C - C_pub:+.1e}); max |state diff| {rows[-1]['max_state_diff']:.1e}")
    (OUT / "folta_table3.json").write_text(json.dumps({
        "source": "Folta et al. (2022), NASA/TP-20220014814, Table 3 (values to 6 decimals)",
        "mu_used": mu, "rows": rows}, indent=1))


if __name__ == "__main__":
    nrho()
    folta_table3()
