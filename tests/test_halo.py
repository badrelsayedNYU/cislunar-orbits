"""Milestone 3: halo families from the Lyapunov bifurcation, southern branch by symmetry."""

import numpy as np
import pytest

from cr3bp.continuation import (Family, continue_family, find_tangent_bifurcation,
                                lyapunov_linear_guess)
from cr3bp.correction import V_from_state, natural_constraint, state_from_V
from cr3bp.dynamics import propagate
from cr3bp.jpl import _monotone_runs, load_family, pointwise_compare
from cr3bp.stability import analyze
from cr3bp.units import MU


@pytest.fixture(scope="module")
def l2_halo():
    """Bifurcation from a short L2 Lyapunov run, then 30 northern halo members."""
    X0, T = lyapunov_linear_guess(MU, 2, 1e-3)
    lyap = Family("L2", MU, True)
    continue_family(lyap, V_from_state(X0, T, True), n_max=20, natural_step=5e-4,
                    natural_step_max=2e-3)
    sz = np.array([analyze(o.X0, o.T, MU)["s_z"] for o in lyap.orbits])
    i = np.where(np.diff(np.sign(sz - 2)) != 0)[0][0]
    Vb, _ = find_tangent_bifurcation(
        MU, V_from_state(lyap.orbits[i].X0, lyap.orbits[i].T, True),
        V_from_state(lyap.orbits[i + 1].X0, lyap.orbits[i + 1].T, True))
    Xb = state_from_V(Vb, True)
    V = V_from_state(Xb, 2 * Vb[-1], False)
    V[1] = 1e-4
    halo = Family("halo", MU, False)
    continue_family(halo, V, n_max=30, natural_index="auto", natural_step=1e-3,
                    natural_step_max=4e-3, max_natural=10,
                    start_extra=natural_constraint(1, 1e-4), ds=4e-3, ds_max=8e-3)
    return halo


def test_halo_leaves_the_plane_and_is_periodic(l2_halo):
    z0 = np.array([o.X0[2] for o in l2_halo.orbits])
    assert z0[0] == pytest.approx(1e-4)
    assert z0[-1] > 0.05                      # well out of plane (> 19,000 km)
    for o in l2_halo.orbits[::6]:
        d = analyze(o.X0, o.T, MU)
        assert d["closure"] < 1e-10
        assert d["pair_defect"] < 1e-7


def test_pseudo_arclength_used_after_step_off(l2_halo):
    methods = [o.method for o in l2_halo.orbits]
    assert methods[:10] == ["natural"] * 10
    assert set(methods[10:]) == {"pseudo-arclength"}
    assert l2_halo.switched_to_pac_at == 10


def test_northern_halo_matches_catalog_pointwise(l2_halo):
    a = l2_halo.arrays()
    Xh = np.array([propagate(x, t / 2, MU).Xf for x, t in zip(a["X0"], a["T"])])
    cat = load_family("halo_L2_N")
    # catalog orbits in the C range covered by the 30 members (just below the bifurcation)
    sel = np.where((cat.jacobi > a["C"].min() + 1e-3) & (cat.jacobi < a["C"].max() - 1e-3)
                   & (cat.period > 3.2))[0]
    idx = sel[np.linspace(0, len(sel) - 1, 6).astype(int)]
    res, failed = pointwise_compare(cat, a["X0"], Xh, a["T"], False, idx)
    assert failed == 0 and len(res) == 6
    for r in res:
        assert r["T"] == pytest.approx(r["T_cat"], rel=1e-9)
        assert r["C"] == pytest.approx(r["C_cat"], abs=1e-10)
        assert r["nu"] == pytest.approx(r["nu_cat"], rel=1e-6)


def test_southern_branch_by_mirror_symmetry(l2_halo):
    """(x, y, z, vx, vy, vz) -> (x, y, -z, vx, vy, -vz): mirrored orbits are periodic with the
    same period, Jacobi constant and stability, and match JPL's southern catalog."""
    S = np.diag([1, 1, -1, 1, 1, -1.0])
    o = l2_halo.orbits[-1]
    dn = analyze(o.X0, o.T, MU)
    ds = analyze(S @ o.X0, o.T, MU)
    assert ds["closure"] < 1e-10
    assert ds["nu"] == pytest.approx(dn["nu"], rel=1e-9)
    cs, cn = load_family("halo_L2_S"), load_family("halo_L2_N")
    j = np.argmin(np.linalg.norm(cn.states - o.X0, axis=1))
    np.testing.assert_allclose(cs.states[j], S @ cn.states[j], atol=1e-12)


def test_monotone_runs_split_at_turning_points():
    C = np.array([3.0, 2.9, 2.8, 2.85, 2.95, 2.9])
    assert _monotone_runs(C) == [(0, 2), (2, 4), (4, 5)]
