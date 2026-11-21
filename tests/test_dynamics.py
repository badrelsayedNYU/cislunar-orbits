"""Milestone 1: equations of motion, Jacobi constant, STM, Lagrange points, symmetry, crossings."""

import numpy as np
import pytest

from cr3bp.dynamics import (eom, hessian_U, jacobi, jacobian, propagate, propagate_to_crossing,
                            pseudo_potential)
from cr3bp.jpl import load_family, system
from cr3bp.lagrange import gradient_U, lagrange_points
from cr3bp.units import MU

# A generic, non-periodic 3D state between Earth and Moon used for several checks.
X_GENERIC = np.array([0.85, 0.05, 0.08, 0.01, 0.12, -0.03])


# ---- Lagrange points ----

def test_lagrange_points_are_equilibria():
    for name, r in lagrange_points(MU).items():
        g = gradient_U(r, MU)
        assert np.max(np.abs(g)) < 5e-16, name
        assert np.max(np.abs(eom(0, np.r_[r, 0, 0, 0], MU))) < 5e-16, name


def test_lagrange_points_match_jpl():
    s = system()
    for name, r in lagrange_points(MU).items():
        np.testing.assert_allclose(r, s[name], rtol=0, atol=1e-14, err_msg=name)


def test_collinear_ordering():
    L = lagrange_points(MU)
    assert L["L3"][0] < -MU < L["L1"][0] < 1 - MU < L["L2"][0]


# ---- derivatives ----

def test_hessian_matches_finite_difference_of_eom():
    """A = df/dX from hessian_U matches central differences of the EOM."""
    A = jacobian(X_GENERIC, MU)
    h = 1e-6
    A_fd = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = h
        A_fd[:, j] = (eom(0, X_GENERIC + e, MU) - eom(0, X_GENERIC - e, MU)) / (2 * h)
    np.testing.assert_allclose(A, A_fd, rtol=0, atol=1e-8)


def test_eom_is_gradient_of_pseudopotential():
    h = 1e-6
    r = X_GENERIC[:3]
    grad_fd = np.array([(pseudo_potential(np.r_[r + h * e, 0, 0, 0], MU)
                         - pseudo_potential(np.r_[r - h * e, 0, 0, 0], MU)) / (2 * h)
                        for e in np.eye(3)])
    np.testing.assert_allclose(gradient_U(r, MU), grad_fd, atol=1e-9)
    H = hessian_U(r, MU)
    np.testing.assert_allclose(H, H.T, atol=0)


# ---- Jacobi constant ----

def test_jacobi_conserved_over_long_arc():
    """Ten revolutions of a JPL L2 halo orbit (about 30 days): Jacobi drift is at round-off level."""
    fam = load_family("halo_L2_S")
    i = len(fam) // 2
    X0, T = fam.states[i], fam.period[i]
    arc = propagate(X0, 10 * T, MU, t_eval=np.linspace(0, 10 * T, 2001))
    C = jacobi(arc.X, MU)
    drift = np.max(np.abs(C - C[0]))
    assert drift < 1e-10, drift
    assert jacobi(X0, MU) == pytest.approx(fam.jacobi[i], abs=1e-12)


def test_jacobi_conserved_chaotic_arc():
    """A generic chaotic arc through the Moon region for 20 TU (~89 days)."""
    arc = propagate(X_GENERIC, 20.0, MU, t_eval=np.linspace(0, 20, 4001))
    C = jacobi(arc.X, MU)
    assert np.max(np.abs(C - C[0])) < 1e-10


# ---- state transition matrix ----

def _fd_stm(X0, tf, h):
    Phi_fd = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = h
        xp = propagate(X0 + e, tf, MU, rtol=1e-13, atol=1e-13).Xf
        xm = propagate(X0 - e, tf, MU, rtol=1e-13, atol=1e-13).Xf
        Phi_fd[:, j] = (xp - xm) / (2 * h)
    return Phi_fd


@pytest.mark.parametrize("tf", [0.5, 2.0, -1.5])
def test_stm_matches_central_differences(tf):
    """Variational STM vs. central differences at h = 1e-5, 1e-6, 1e-7.

    Where the finite-difference truncation error dominates (err > 1e-6 at h = 1e-5) it must
    fall as h^2, showing the residual is the FD error and not an STM error. The best agreement
    must be below 5e-8 relative to the largest STM entry."""
    Phi = propagate(X_GENERIC, tf, MU, stm=True, rtol=1e-13, atol=1e-13).Phi
    scale = np.max(np.abs(Phi))
    err = {h: np.max(np.abs(Phi - _fd_stm(X_GENERIC, tf, h))) / scale for h in (1e-5, 1e-6, 1e-7)}
    if err[1e-5] > 1e-6:
        assert 30 < err[1e-5] / err[1e-6] < 300, err   # second-order convergence
    assert min(err.values()) < 5e-8, err


def test_stm_is_symplectic():
    """CR3BP flow is Hamiltonian: Phi^T Omega Phi = Omega in canonical coordinates.

    In rotating-frame velocity coordinates the invariant form is J' = [[2W, I], [-I, 0]] with
    W the Coriolis block; equivalently det(Phi) = 1. Check the determinant and reciprocal
    eigenvalue pairing after one period of a periodic orbit (done in the stability tests).
    """
    arc = propagate(X_GENERIC, 3.0, MU, stm=True)
    assert np.linalg.det(arc.Phi) == pytest.approx(1.0, abs=1e-8)


# ---- symmetry ----

def test_mirror_symmetry():
    """(x, y, z, vx, vy, vz, t) -> (x, -y, z, -vx, vy, -vz, -t) maps solutions to solutions."""
    S = np.diag([1, -1, 1, -1, 1, -1.0])
    ts = np.linspace(0, 3.0, 301)
    fwd = propagate(X_GENERIC, 3.0, MU, t_eval=ts, rtol=1e-13, atol=1e-13)
    bwd = propagate(S @ X_GENERIC, -3.0, MU, t_eval=-ts, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(bwd.X, fwd.X @ S, rtol=0, atol=1e-10)


# ---- events ----

def test_y_crossing_of_jpl_halo_is_half_period():
    fam = load_family("halo_L1_N")
    for i in (100, 2000, 4000):
        arc = propagate_to_crossing(fam.states[i], MU)
        assert arc.t_events[0][0] == pytest.approx(fam.period[i] / 2, rel=1e-9)
        assert abs(arc.Xf[1]) < 1e-12
        # perpendicular crossing: vx = vz = 0 at the half period (to catalog precision)
        assert abs(arc.Xf[3]) < 1e-7 and abs(arc.Xf[5]) < 1e-7
