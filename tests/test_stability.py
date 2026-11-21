"""Stability index definition, Broucke parameters and symplectic structure of the monodromy matrix."""

import numpy as np
import pytest

from cr3bp.jpl import load_family
from cr3bp.stability import analyze, broucke, eigen_pairs, monodromy, stability_index
from cr3bp.units import MU


def _clean_catalog_members(name, k=6):
    """Catalog orbits spread over the family whose stored IC closes to < 1e-9 (precise entries)."""
    fam = load_family(name)
    out = []
    for i in np.linspace(0, len(fam) - 1, 4 * k).astype(int):
        M, XT = monodromy(fam.states[i], fam.period[i], MU, 1e-13, 1e-13)
        if np.linalg.norm(XT - fam.states[i]) < 1e-9:
            out.append((M, fam.stability[i]))
        if len(out) == k:
            break
    assert len(out) == k
    return out


@pytest.mark.parametrize("name", ["lyapunov_L1", "lyapunov_L2", "halo_L1_N", "halo_L2_S"])
def test_stability_index_definition_reproduces_jpl(name):
    """nu = (|lam_max| + 1/|lam_max|)/2 from JPL's own ICs matches JPL's stability column.

    Agreement is ~1e-12 for most entries; some L2 Lyapunov entries are tabulated to only ~2e-6
    even where the stored IC closes to < 1e-9, hence rel = 1e-5.
    """
    for M, nu_jpl in _clean_catalog_members(name):
        assert stability_index(M) == pytest.approx(nu_jpl, rel=1e-5)


def test_broucke_parameters_match_eigenvalues():
    fam = load_family("halo_L2_S")
    for i in (100, 600, 1200):
        M, _ = monodromy(fam.states[i], fam.period[i], MU)
        s = sorted(np.real_if_close(broucke(M)), key=lambda v: abs(v))
        lam, _ = eigen_pairs(M)
        # nontrivial pairs: remove the two eigenvalues closest to 1
        nontriv = lam[np.argsort(np.abs(lam - 1))[2:]]
        s_eig = sorted({round((l + 1 / l).real, 6) for l in nontriv}, key=abs)
        np.testing.assert_allclose(sorted(np.real(s), key=abs), s_eig, rtol=1e-5)


def test_monodromy_is_symplectic_and_has_trivial_pair():
    fam = load_family("lyapunov_L1")
    i = len(fam) // 3
    d = analyze(fam.states[i], fam.period[i], MU, rtol=1e-13)
    assert d["det"] == pytest.approx(1.0, abs=1e-8)
    assert d["pair_defect"] < 1e-7
    lam = d["eigenvalues"]
    # two eigenvalues at +1 (periodicity and the Jacobi integral)
    assert np.sum(np.abs(lam - 1) < 1e-4) == 2
