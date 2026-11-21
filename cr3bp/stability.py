"""Monodromy matrix, eigenstructure and stability indices of periodic orbits.

Stability index (JPL catalog): the API documentation defines it only as "a function of the
eigenvalues of the monodromy matrix" with stability requiring a value <= 1. We use

    nu = (|lambda_max| + 1/|lambda_max|) / 2,

lambda_max the eigenvalue of largest modulus. This gives nu = 1 for orbits whose eigenvalues all
lie on the unit circle (JPL reports exactly 1 for those) and nu = (lambda + 1/lambda)/2 for a real
unstable pair. tests/test_stability.py confirms the definition reproduces the catalog's values by
integrating JPL's own initial conditions.

Broucke stability parameters: for a symplectic 6x6 monodromy with the trivial pair at +1, the
two nontrivial reciprocal pairs (lambda, 1/lambda) have s_i = lambda_i + 1/lambda_i equal to the
roots of s^2 + alpha s + (beta - 2) = 0 with alpha = 2 - tr(M), beta = (alpha^2 + 2 - tr(M^2))/2.
These come from traces only, so no eigenvector pairing is needed; a pair passes through +1 when
some s_i = 2 (tangent bifurcation, e.g. halo from Lyapunov).
"""

from __future__ import annotations

import numpy as np

from .dynamics import ATOL, RTOL, propagate


def monodromy(X0, T, mu, rtol=RTOL, atol=ATOL):
    arc = propagate(X0, T, mu, stm=True, rtol=rtol, atol=atol)
    return arc.Phi, arc.Xf


def stability_index(M) -> float:
    lam = np.linalg.eigvals(M)
    a = np.max(np.abs(lam))
    return 0.5 * (a + 1.0 / a)


def broucke(M):
    """(s1, s2) = lambda + 1/lambda for the two nontrivial pairs (may be complex)."""
    tr1 = np.trace(M)
    tr2 = np.trace(M @ M)
    alpha = 2.0 - tr1
    beta = 0.5 * (alpha**2 + 2.0 - tr2)
    disc = np.lib.scimath.sqrt(alpha**2 - 4.0 * beta + 8.0)
    s1 = (-alpha + disc) / 2.0
    s2 = (-alpha - disc) / 2.0
    return s1, s2


def eigen_pairs(M):
    """Eigenvalues sorted by modulus and the worst reciprocal-pair defect max |lam_i lam_j - 1|.

    Each eigenvalue is paired with the eigenvalue closest to its reciprocal; for a symplectic
    matrix every product is exactly 1.
    """
    lam = np.linalg.eigvals(M)
    lam = lam[np.argsort(np.abs(lam))]
    defects = [np.min(np.abs(l * lam - 1.0)) for l in lam]
    return lam, float(np.max(defects))


def out_of_plane_trace(M) -> float:
    """For a planar orbit the (z, vz) block decouples; its trace is s for the out-of-plane pair."""
    return float(M[2, 2] + M[5, 5])


def analyze(X0, T, mu, rtol=RTOL, atol=None):
    """Monodromy-based diagnostics for one periodic orbit."""
    atol = rtol if atol is None else atol
    M, XT = monodromy(X0, T, mu, rtol, atol)
    lam, defect = eigen_pairs(M)
    s1, s2 = broucke(M)
    return {
        "closure": float(np.linalg.norm(XT - X0)),
        "nu": stability_index(M),
        "s1": complex(s1),
        "s2": complex(s2),
        "s_z": out_of_plane_trace(M),
        "eigenvalues": lam,
        "pair_defect": defect,
        "det": float(np.linalg.det(M)),
        "M": M,
    }
