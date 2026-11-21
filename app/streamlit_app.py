"""Streamlit viewer for the computed Earth-Moon CR3BP families.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr3bp.dynamics import apses, propagate  # noqa: E402
from cr3bp.manifolds import propagate_branch, seeds  # noqa: E402
from cr3bp.units import LU_KM, MU, R_MOON_KM, nd_to_days  # noqa: E402

RES = ROOT / "data" / "results"
FAMILIES = {
    "L1 Lyapunov": "lyapunov_L1", "L2 Lyapunov": "lyapunov_L2",
    "L1 halo, northern": "halo_L1_N", "L1 halo, southern": "halo_L1_S",
    "L2 halo, northern": "halo_L2_N", "L2 halo, southern": "halo_L2_S",
}
BLUE, ORANGE, AQUA, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


@st.cache_data
def load(name):
    d = np.load(RES / f"{name}.npz")
    return {k: d[k] for k in d.files}


@st.cache_data
def trajectory(X0, T, rtol, n=800):
    return propagate(np.array(X0), T, MU, t_eval=np.linspace(0, T, n), rtol=rtol, atol=rtol).X


@st.cache_data
def manifold_arcs(X0, T, n_seeds, t_max, d_km):
    out = {}
    for kind, backward in (("unstable", False), ("stable", True)):
        s = seeds(np.array(X0), T, MU, n_seeds, kind)
        arcs = []
        for k in range(n_seeds):
            for sign in (+1, -1):
                a = propagate_branch(s.X_orbit[k] + sign * d_km / LU_KM * s.v[k], MU,
                                     backward=backward, t_max=t_max, n_out=300)
                arcs.append(a.X[:, :3])
        out[kind] = arcs
    return out


def moon_mesh(scale):
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    r = R_MOON_KM / scale
    return go.Surface(x=r * np.cos(u) * np.sin(v), y=r * np.sin(u) * np.sin(v),
                      z=r * np.cos(v), colorscale=[[0, "#8f8e89"], [1, "#8f8e89"]],
                      showscale=False, hoverinfo="skip", name="Moon")


def km(X):
    return (X[:, :3] - [1 - MU, 0, 0]) * LU_KM / 1e3


st.set_page_config(page_title="Cislunar periodic orbits", layout="wide")
st.title("Earth-Moon periodic orbits and manifolds (CR3BP)")
st.caption("Families computed by this repository's differential corrector and continuation, "
           "validated against the JPL Three-Body Periodic Orbit catalog. Coordinates: rotating "
           "frame, centred on the Moon, in thousands of km.")

tab_fam, tab_nrho = st.tabs(["Families", "9:2 NRHO"])

with tab_fam:
    c1, c2 = st.columns([1, 3])
    with c1:
        fam_label = st.selectbox("Family", list(FAMILIES))
        name = FAMILIES[fam_label]
        d = load(name)
        n = len(d["T"])
        i = st.slider("Member (continuation order)", 0, n - 1, n // 2)
        show_neighbors = st.checkbox("Show other family members", value=True)
        show_man = st.checkbox("Compute invariant manifolds", value=False)
        if show_man:
            n_seeds = st.slider("Seeds along the orbit", 10, 60, 24, step=2)
            t_max_d = st.slider("Propagation time [days]", 5, 40, 20)
            d_km = st.select_slider("Step-off distance [km]", [10, 25, 50, 100, 200], value=50)

        X0, T, C, nu = d["X0"][i], float(d["T"][i]), float(d["C"][i]), float(d["nu"][i])
        rp, ra = float(d["r_perilune"][i]) * LU_KM, float(d["r_apolune"][i]) * LU_KM
        st.metric("Period", f"{nd_to_days(T):.4f} d", f"{T:.6f} TU", delta_color="off")
        st.metric("Jacobi constant", f"{C:.6f}")
        st.metric("Stability index ν", f"{nu:.4g}")
        st.metric("Perilune / apolune radius", f"{rp:,.0f} / {ra:,.0f} km")
        Tc = d.get(f"T_cat_{name}")
        if Tc is not None and np.isfinite(Tc[i]):
            st.caption(f"JPL catalog at this C (interpolated): T = {Tc[i]:.9f} TU, relative "
                       f"difference {abs(T - Tc[i]) / Tc[i]:.1e}; ν = "
                       f"{d[f'nu_cat_{name}'][i]:.6g}.")
        st.caption(f"Closure ‖X(T)−X₀‖ = {d['closure'][i]:.1e}; integration rtol = atol = "
                   f"{d['rtol'][i]:g}.")
        st.caption("Initial state [x, y, z, vx, vy, vz] (nondim.): "
                   + ", ".join(f"{v:.10f}" for v in X0))

    with c2:
        fig = go.Figure()
        fig.add_trace(moon_mesh(1e3))
        if show_neighbors:
            for k in np.linspace(0, n - 1, 16).astype(int):
                Y = km(trajectory(tuple(d["X0"][k]), float(d["T"][k]), float(d["rtol"][k]), 300))
                fig.add_trace(go.Scatter3d(x=Y[:, 0], y=Y[:, 1], z=Y[:, 2], mode="lines",
                                           line=dict(color="#c3c2b7", width=1.5),
                                           hoverinfo="skip", showlegend=False))
        if show_man:
            if nu < 1.001:
                st.info("This orbit is linearly stable (ν ≈ 1): no stable/unstable manifolds.")
            else:
                with st.spinner("Propagating manifold trajectories..."):
                    arcs = manifold_arcs(tuple(X0), T, n_seeds, t_max_d / nd_to_days(1.0), d_km)
                for kind, color in (("unstable", BLUE), ("stable", ORANGE)):
                    for j, a in enumerate(arcs[kind]):
                        Y = km(np.c_[a, np.zeros_like(a)])
                        fig.add_trace(go.Scatter3d(
                            x=Y[:, 0], y=Y[:, 1], z=Y[:, 2], mode="lines",
                            line=dict(color=color, width=1), opacity=0.5, hoverinfo="skip",
                            name=f"{kind} manifold", showlegend=j == 0, legendgroup=kind))
        Y = km(trajectory(tuple(X0), T, float(d["rtol"][i])))
        fig.add_trace(go.Scatter3d(x=Y[:, 0], y=Y[:, 1], z=Y[:, 2], mode="lines",
                                   line=dict(color="#0b0b0b", width=5), name="selected orbit",
                                   hovertemplate="x %{x:.1f}, y %{y:.1f}, z %{z:.1f} (10³ km)"))
        fig.update_layout(height=680, margin=dict(l=0, r=0, t=10, b=0),
                          scene=dict(aspectmode="data", xaxis_title="x [10³ km]",
                                     yaxis_title="y [10³ km]", zaxis_title="z [10³ km]"),
                          legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, width="stretch")

        # period and stability vs C for the family, selected member marked
        g1, g2 = st.columns(2)
        for col, key, title, logy in ((g1, "T", "Period [TU]", False),
                                      (g2, "nu", "Stability index ν", True)):
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=d["C"], y=d[key], mode="lines", line=dict(color=BLUE),
                                    name="this work"))
            ck = f"{'T' if key == 'T' else 'nu'}_cat_{name}"
            if ck in d:
                f2.add_trace(go.Scatter(x=d["C"], y=d[ck], mode="markers",
                                        marker=dict(color=ORANGE, size=4),
                                        name="JPL (interpolated)"))
            f2.add_trace(go.Scatter(x=[C], y=[d[key][i]], mode="markers",
                                    marker=dict(color="#0b0b0b", size=10), name="selected"))
            f2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title=title,
                             xaxis_title="Jacobi constant", yaxis_type="log" if logy else None)
            col.plotly_chart(f2, width="stretch")

with tab_nrho:
    nr = json.loads((RES / "nrho.json").read_text())
    st.markdown("The L2 southern halo member with period (2/9) × 29.530589 d = 6.5624 d (the "
                "Gateway orbit's 9:2 lunar synodic resonance), solved in the CR3BP with two "
                "sets of characteristic quantities. Both are CR3BP results.")
    cols = st.columns(2)
    fig = go.Figure()
    fig.add_trace(moon_mesh(1e3))
    for col, (uname, color, label) in zip(cols, (("jpl", BLUE, "JPL catalog constants"),
                                                 ("folta", ORANGE, "Folta et al. (2022) constants"))):
        o = nr["orbits"][uname]["9:2"][0]
        u = nr["orbits"][uname]["constants"]
        with col:
            st.subheader(label)
            st.write(f"μ = {u['mu']:.12f}, LU = {u['LU_km']:,.2f} km, TU = {u['TU_s']:,.2f} s")
            st.metric("Perilune radius (altitude)",
                      f"{o['perilune_radius_km']:,.1f} km ({o['perilune_altitude_km']:,.1f} km)")
            st.metric("Apolune radius", f"{o['apolune_radius_km']:,.1f} km")
            st.metric("Jacobi constant / ν", f"{o['jacobi']:.6f} / {o['stability_index']:.4f}")
        X = propagate(np.array(o["X0"]), o["period_nd"], u["mu"],
                      t_eval=np.linspace(0, o["period_nd"], 1500), rtol=1e-13, atol=1e-13).X
        Y = (X[:, :3] - [1 - u["mu"], 0, 0]) * u["LU_km"] / 1e3
        fig.add_trace(go.Scatter3d(x=Y[:, 0], y=Y[:, 1], z=Y[:, 2], mode="lines",
                                   line=dict(color=color, width=5), name=label))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(aspectmode="data", xaxis_title="x [10³ km]",
                                 yaxis_title="y [10³ km]", zaxis_title="z [10³ km]"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Published: about 3150 km (Zimovan, Howell & Davis 2017, CR3BP, read from a "
               "figure); 3196–3557 km, mean 3366 km (Lee 2019, ephemeris model). See RESULTS.md.")
