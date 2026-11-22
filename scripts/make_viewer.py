"""Write docs/index.html: a static interactive viewer for GitHub Pages.

Reads only data/results (the same files as RESULTS.md and the figures), re-integrates a subset of
members for plotting, and embeds them with plotly. No server is needed.

    python scripts/make_viewer.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr3bp.dynamics import propagate  # noqa: E402
from cr3bp.units import LU_KM, MU, R_MOON_KM, nd_to_days  # noqa: E402

RES = ROOT / "data" / "results"
OUT = ROOT / "docs" / "index.html"
FAMILIES = {
    "L1 Lyapunov": "lyapunov_L1", "L2 Lyapunov": "lyapunov_L2",
    "L1 halo, northern": "halo_L1_N", "L1 halo, southern": "halo_L1_S",
    "L2 halo, northern": "halo_L2_N", "L2 halo, southern": "halo_L2_S",
}
N_MEMBERS, N_PTS = 24, 400
BLUE, ORANGE = "#2a78d6", "#eb6834"
DEFAULT = "L2 halo, southern"     # shown first: the family containing the NRHO


def km(X, mu=MU, lu=LU_KM):
    return np.round((X[:, :3] - [1 - mu, 0, 0]) * lu / 1e3, 2)


def moon(visible=True):
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    r = R_MOON_KM / 1e3
    return go.Surface(x=r * np.cos(u) * np.sin(v), y=r * np.sin(u) * np.sin(v), z=r * np.cos(v),
                      colorscale=[[0, "#8f8e89"], [1, "#8f8e89"]], showscale=False,
                      hoverinfo="skip", name="Moon", visible=visible)


def scene(eye):
    return dict(aspectmode="data", camera=dict(eye=eye), xaxis_title="x [10³ km]", yaxis_title="y [10³ km]",
                zaxis_title="z [10³ km]")


def families_figure():
    fig = go.Figure([moon()])
    owner = [None]                      # family label of each trace (None = always shown)
    for k, (label, name) in enumerate(FAMILIES.items()):
        d = np.load(RES / f"{name}.npz")
        n = len(d["T"])
        idx = np.unique(np.linspace(0, n - 1, N_MEMBERS).astype(int))
        colors = sample_colorscale("Blues", list(np.linspace(0.35, 1.0, len(idx))))
        for j, i in zip(range(len(idx)), idx):
            T = float(d["T"][i])
            X = propagate(d["X0"][i], T, MU, t_eval=np.linspace(0, T, N_PTS),
                          rtol=float(d["rtol"][i]), atol=float(d["rtol"][i])).X
            Y = km(X)
            info = (f"{label}, member {i} of {n}<br>T = {nd_to_days(T):.4f} d, "
                    f"C = {float(d['C'][i]):.6f}, ν = {float(d['nu'][i]):.4g}<br>"
                    f"perilune {float(d['r_perilune'][i]) * LU_KM:,.0f} km, "
                    f"apolune {float(d['r_apolune'][i]) * LU_KM:,.0f} km")
            fig.add_trace(go.Scatter3d(
                x=Y[:, 0], y=Y[:, 1], z=Y[:, 2], mode="lines", visible=label == DEFAULT,
                line=dict(color=colors[j], width=3), showlegend=False,
                hovertemplate=info + "<extra></extra>"))
            owner.append(label)
    buttons = [dict(label=label, method="update",
                    args=[{"visible": [o is None or o == label for o in owner]}])
               for label in FAMILIES]
    fig.update_layout(
        height=640, margin=dict(l=0, r=0, t=40, b=0), scene=scene(dict(x=1.6, y=-1.6, z=0.9)),
        updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.06, yanchor="top",
                          showactive=True, active=list(FAMILIES).index(DEFAULT))])
    return fig


def nrho_figure(nr):
    fig = go.Figure([moon()])
    rows = []
    for uname, color, label in (("jpl", BLUE, "JPL catalog constants"),
                                ("folta", ORANGE, "Folta et al. (2022) constants")):
        o = nr["orbits"][uname]["9:2"][0]
        u = nr["orbits"][uname]["constants"]
        X = propagate(np.array(o["X0"]), o["period_nd"], u["mu"],
                      t_eval=np.linspace(0, o["period_nd"], 1500), rtol=1e-13, atol=1e-13).X
        Y = km(X, u["mu"], u["LU_km"])
        fig.add_trace(go.Scatter3d(x=Y[:, 0], y=Y[:, 1], z=Y[:, 2], mode="lines", name=label,
                                   line=dict(color=color, width=5), hoverinfo="skip"))
        rows.append((label, u, o))
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0), scene=scene(dict(x=2.6, y=1.6, z=0.9)),
                      legend=dict(orientation="h", y=1.0))
    return fig, rows


def main():
    nr = json.loads((RES / "nrho.json").read_text())
    f1 = families_figure()
    f2, rows = nrho_figure(nr)
    table = "".join(
        f"<tr><td>{label}</td><td>{u['mu']:.12f}</td><td>{u['LU_km']:,.2f}</td>"
        f"<td>{o['perilune_radius_km']:,.1f}</td><td>{o['apolune_radius_km']:,.0f}</td>"
        f"<td>{o['jacobi']:.6f}</td><td>{o['stability_index']:.4f}</td></tr>"
        for label, u, o in rows)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cislunar periodic orbits</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 0 auto;
        padding: 16px; color: #1c1c1a; background: #fff; line-height: 1.5; }}
 h1 {{ font-size: 1.6rem; margin-bottom: .2rem; }} h2 {{ margin-top: 2.5rem; }}
 .muted {{ color: #52514e; }} table {{ border-collapse: collapse; font-size: .9rem; }}
 td, th {{ border-bottom: 1px solid #ddd; padding: 4px 10px; text-align: right; }}
 td:first-child, th:first-child {{ text-align: left; }}
 .scroll {{ overflow-x: auto; }}
</style></head><body>
<h1>Earth-Moon periodic orbits (CR3BP)</h1>
<p class="muted">Orbits computed by the differential corrector and continuation in
<a href="https://github.com/badrelsayedNYU/cislunar-orbits">badrelsayedNYU/cislunar-orbits</a> and
validated against the JPL Three-Body Periodic Orbit catalog (see
<a href="https://github.com/badrelsayedNYU/cislunar-orbits/blob/main/RESULTS.md">RESULTS.md</a>).
Rotating frame centred on the Moon, thousands of km. Drag to rotate; hover over an orbit for its
period, Jacobi constant, stability index and apses.</p>
<h2>Families</h2>
<p class="muted">Pick a family with the menu. {N_MEMBERS} evenly spaced members of each family are
shown, darker along the continuation.</p>
{f1.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False})}
<h2>9:2 synodic-resonant NRHO</h2>
<p class="muted">The L2 southern halo member with period (2/9) × 29.530589 d = 6.5624 d, solved in
the CR3BP with two sets of characteristic quantities. Both are CR3BP results; the perilune radius
depends on which set is used. Published: about 3150 km (Zimovan, Howell &amp; Davis 2017, CR3BP,
read from a figure); 3196–3557 km, mean 3366 km (Lee 2019, ephemeris model).</p>
<div class="scroll"><table>
<tr><th>Constants</th><th>μ</th><th>LU [km]</th><th>Perilune [km]</th><th>Apolune [km]</th>
<th>C</th><th>ν</th></tr>{table}</table></div>
{f2.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}
<p class="muted">Generated by <code>scripts/make_viewer.py</code> from the repository's results.
For live member selection and manifold computation, run the Streamlit app locally:
<code>streamlit run app/streamlit_app.py</code>.</p>
</body></html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
