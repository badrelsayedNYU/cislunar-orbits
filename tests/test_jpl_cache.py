"""Milestone 0: the cached JPL catalog is complete, self-consistent, and drives the unit system."""

import numpy as np
import pytest

from cr3bp import units
from cr3bp.jpl import FAMILIES, load_family, system


@pytest.mark.parametrize("name", FAMILIES)
def test_cache_present_and_well_formed(name):
    fam = load_family(name)
    assert len(fam) > 1000
    assert fam.states.shape == (len(fam), 6)
    assert fam.fetched_utc.endswith("Z")
    assert "ssd-api.jpl.nasa.gov/periodic_orbits.api" in fam.request_url
    # Every catalog orbit starts with a perpendicular x-z plane crossing: y0 = vx0 = vz0 = 0.
    # The stored values are zero only to ~1e-8 (halo vz0 reaches 1.6e-8), which bounds how
    # precisely the catalog itself can be reproduced.
    assert np.max(np.abs(fam.states[:, 1])) < 1e-20
    assert np.max(np.abs(fam.states[:, 3])) < 1e-7
    assert np.max(np.abs(fam.states[:, 5])) < 1e-7
    assert np.all(fam.stability >= 1.0)


def test_all_families_share_system_constants():
    ref = load_family(FAMILIES[0]).system
    for name in FAMILIES[1:]:
        assert load_family(name).system == ref


def test_planar_families_are_planar():
    for name in ("lyapunov_L1", "lyapunov_L2"):
        assert np.max(np.abs(load_family(name).states[:, 2])) < 1e-20


def test_southern_branch_mirrors_northern():
    for lp in ("L1", "L2"):
        n, s = load_family(f"halo_{lp}_N"), load_family(f"halo_{lp}_S")
        assert len(n) == len(s)
        np.testing.assert_allclose(s.states[:, 2], -n.states[:, 2], rtol=0, atol=1e-12)
        np.testing.assert_allclose(s.period, n.period, rtol=1e-12)


def test_units_from_catalog():
    s = system()
    assert units.MU == s["mu"] == pytest.approx(0.01215058560962404, rel=1e-15)
    assert units.LU_KM == pytest.approx(389703.2648, rel=1e-9)
    assert units.TU_S == pytest.approx(382981.2891, rel=1e-9)
    # TU = sqrt(LU^3 / (GM_Earth + GM_Moon)). With DE-series GMs (398600.435, 4902.800 km^3/s^2)
    # this reproduces JPL's TU to ~1e-6. Note LU = 389,703 km is not the 384,400 km mean distance,
    # so 2*pi*TU = 27.85 d rather than the 27.32 d sidereal month.
    gm = 398600.435 + 4902.800
    assert units.TU_S == pytest.approx(np.sqrt(units.LU_KM**3 / gm), rel=5e-6)
    assert 2 * np.pi * units.TU_DAYS == pytest.approx(27.85, abs=0.01)


def test_unit_round_trips():
    x = np.array([0.0, 1e-3, 1.0, 7.5])
    np.testing.assert_allclose(units.km_to_nd(units.nd_to_km(x)), x, rtol=1e-15)
    np.testing.assert_allclose(units.days_to_nd(units.nd_to_days(x)), x, rtol=1e-15)
    np.testing.assert_allclose(units.kms_to_nd(units.nd_to_kms(x)), x, rtol=1e-15)
    assert units.nd_to_kms(1.0) == pytest.approx(units.LU_KM / units.TU_S)
