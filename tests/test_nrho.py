"""Milestone 4: the 9:2 synodic-resonant L2 southern NRHO and published CR3BP cross-checks.

Uses the committed family in data/results/ only for initial guesses.
"""

import pathlib

import numpy as np
import pytest

from cr3bp import units
from cr3bp.correction import V_from_state, correct, natural_constraint, state_from_V
from cr3bp.dynamics import apses, jacobi
from cr3bp.stability import analyze

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "data" / "results"
FOLTA = {"mu": 1.215058535056245e-2, "LU_km": 384400.0, "TU_s": 375190.2588926273}


def _solve(T, mu):
    fam = np.load(RESULTS / "halo_L2_S.npz")
    i = np.where(np.diff(np.sign(fam["T"] - T)) != 0)[0]
    assert len(i) == 1                       # one 9:2 member on the computed family
    i = i[0]
    c = correct(V_from_state(fam["X0"][i], fam["T"][i], False), mu, False,
                extra=natural_constraint(3, T / 2), rtol=1e-13, atol=1e-13)
    return state_from_V(c.V, False)


@pytest.fixture(scope="module")
def nrho_jpl():
    T = 2 / 9 * units.SYNODIC_MONTH_DAYS * 86400 / units.TU_S
    return _solve(T, units.MU), T


def test_nrho_is_periodic_southern_and_nearly_stable(nrho_jpl):
    X0, T = nrho_jpl
    d = analyze(X0, T, units.MU, rtol=1e-13)
    assert d["closure"] < 1e-10
    assert units.nd_to_days(T) == pytest.approx(6.5624, abs=1e-4)
    assert 1.0 <= d["nu"] < 2.0                  # NRHOs are nearly stable
    assert X0[2] < 0                             # southern: apolune below the Earth-Moon plane


def test_nrho_geometry_jpl_units(nrho_jpl):
    X0, T = nrho_jpl
    rp, ra, _, _ = apses(X0, T, units.MU, rtol=1e-13)
    assert units.nd_to_km(rp) == pytest.approx(2936, abs=5)
    assert units.nd_to_km(ra) == pytest.approx(71407, abs=20)


def test_unit_choice_moves_the_9_2_perilune():
    """Same 6.5624-day period, Folta et al. (2022) Table 2 constants: perilune ~3249 km, ~313 km
    higher than with JPL's constants, and within Lee (2019)'s ephemeris range 3196-3557 km."""
    T = 2 / 9 * units.SYNODIC_MONTH_DAYS * 86400 / FOLTA["TU_s"]
    X0 = _solve(T, FOLTA["mu"])
    rp, _, _, _ = apses(X0, T, FOLTA["mu"], rtol=1e-13)
    rp_km = rp * FOLTA["LU_km"]
    assert rp_km == pytest.approx(3249, abs=5)
    assert 3196 <= rp_km <= 3557


@pytest.mark.parametrize("name,T_pub,C_pub,X_pub", [
    ("halo_L2_N", 3.366323, 3.128090, [1.174193, 0, 0.076230, 0, -0.182432, 0]),
    ("halo_L1_N", 1.868528, 3.003577, [0.906618, 0, 0.203669, 0, 0.169171, 0]),
])
def test_folta_table3_halo_orbits(name, T_pub, C_pub, X_pub):
    """Correct our orbit at the published period; C and the state agree to the table's 6 decimals."""
    fam = np.load(RESULTS / f"{name}.npz")
    X_pub = np.array(X_pub)
    R = fam["X0"] if np.linalg.norm(fam["X0"] - X_pub, axis=1).min() < \
        np.linalg.norm(fam["Xhalf"] - X_pub, axis=1).min() else fam["Xhalf"]
    k = int(np.argmin(np.linalg.norm(R - X_pub, axis=1)))
    c = correct(V_from_state(R[k], fam["T"][k], False), units.MU, False,
                extra=natural_constraint(3, T_pub / 2))
    X = state_from_V(c.V, False)
    assert jacobi(X, units.MU) == pytest.approx(C_pub, abs=1e-6)
    np.testing.assert_allclose(X, X_pub, atol=1.5e-6)
