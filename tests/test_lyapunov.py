"""Milestone 2: planar Lyapunov families, differential correction, continuation, halo bifurcation."""

import numpy as np
import pytest

from cr3bp.continuation import (Family, continue_family, find_tangent_bifurcation,
                                lyapunov_linear_guess)
from cr3bp.correction import (V_from_state, correct, jacobi_constraint, natural_constraint,
                              state_from_V)
from cr3bp.dynamics import jacobi, propagate
from cr3bp.jpl import compare_family, load_family
from cr3bp.stability import analyze
from cr3bp.units import MU


@pytest.mark.parametrize("which", [1, 2])
def test_linear_guess_corrects_to_small_lyapunov(which):
    X0, T_lin = lyapunov_linear_guess(MU, which, 1e-4)
    c = correct(V_from_state(X0, T_lin, True), MU, True,
                extra=natural_constraint(0, X0[0]))
    T = 2 * c.V[-1]
    assert c.residual < 1e-12
    # a 39 km orbit is nearly linear: the period is within 0.1% of 2*pi/omega
    assert T == pytest.approx(T_lin, rel=1e-3)
    d = analyze(state_from_V(c.V, True), T, MU)
    assert d["closure"] < 1e-10


def test_fixed_jacobi_correction_matches_catalog():
    """Correct our own orbit to a catalog orbit's Jacobi constant; the period must agree."""
    cat = load_family("lyapunov_L1")
    X0, T = lyapunov_linear_guess(MU, 1, 1e-3)
    fam = Family("L1", MU, True)
    continue_family(fam, V_from_state(X0, T, True), n_max=30, natural_step=-2e-3,
                    natural_step_max=2e-3)
    o = fam.orbits[-1]
    # nearest catalog orbit in C (used only for its C value and its period as the reference)
    j = np.argmin(np.abs(cat.jacobi - o.C))
    c = correct(V_from_state(o.X0, o.T, True), MU, True,
                extra=jacobi_constraint(cat.jacobi[j], MU, True))
    X = state_from_V(c.V, True)
    assert jacobi(X, MU) == pytest.approx(cat.jacobi[j], abs=1e-12)
    assert 2 * c.V[-1] == pytest.approx(cat.period[j], rel=1e-9)
    assert analyze(X, 2 * c.V[-1], MU)["nu"] == pytest.approx(cat.stability[j], rel=1e-6)


@pytest.fixture(scope="module")
def l2_family():
    X0, T = lyapunov_linear_guess(MU, 2, 1e-3)
    fam = Family("L2", MU, True)
    continue_family(fam, V_from_state(X0, T, True), n_max=40, natural_step=5e-4,
                    natural_step_max=2e-3)
    return fam


def test_family_members_are_periodic_and_symplectic(l2_family):
    for o in l2_family.orbits[::5]:
        d = analyze(o.X0, o.T, MU)
        assert d["closure"] < 1e-10
        assert d["pair_defect"] < 1e-7
        assert abs(d["det"] - 1) < 1e-8


def test_family_matches_catalog(l2_family):
    a = l2_family.arrays()
    Xh = np.array([propagate(x, t / 2, MU).Xf for x, t in zip(a["X0"], a["T"])])
    nu = np.array([analyze(x, t, MU)["nu"] for x, t in zip(a["X0"], a["T"])])
    cmp = compare_family(load_family("lyapunov_L2"), a["X0"], Xh, a["C"], a["T"], nu)
    ok = np.isfinite(cmp["T_relerr"])
    assert ok.sum() >= 20
    assert np.nanmax(cmp["T_relerr"]) < 1e-9
    assert np.nanmax(cmp["nu_relerr"]) < 1e-5


def test_halo_bifurcation_on_L2_lyapunov(l2_family):
    """s_z = 2 is crossed once in the first members; the bifurcating orbit has the period of the
    longest-period member of JPL's L2 halo family (its planar end)."""
    sz = np.array([analyze(o.X0, o.T, MU)["s_z"] for o in l2_family.orbits])
    i = np.where(np.diff(np.sign(sz - 2)) != 0)[0]
    assert len(i) >= 1
    i = i[0]
    Va = V_from_state(l2_family.orbits[i].X0, l2_family.orbits[i].T, True)
    Vb = V_from_state(l2_family.orbits[i + 1].X0, l2_family.orbits[i + 1].T, True)
    Vbif, resid = find_tangent_bifurcation(MU, Va, Vb)
    assert abs(resid) < 1e-9
    halo = load_family("halo_L2_N")
    assert 2 * Vbif[-1] == pytest.approx(halo.period.max(), rel=1e-6)
    C_bif = jacobi(state_from_V(Vbif, True), MU)
    assert C_bif == pytest.approx(3.1521, abs=1e-4)
