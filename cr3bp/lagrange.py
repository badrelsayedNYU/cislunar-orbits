"""Libration points: collinear L1-L3 by Newton iteration on the quintic, L4/L5 in closed form."""

from __future__ import annotations

import numpy as np


def _collinear_quintic(gamma, mu, which):
    """Quintic in gamma = distance from the nearer primary (Szebehely; Koon, Lo, Marsden & Ross).

    L1: gamma from the Moon toward Earth;  L2: gamma from the Moon away from Earth;
    L3: gamma from Earth away from the Moon.
    Returns (f, f').
    """
    g = gamma
    if which == 1:
        c = [1.0, -(3 - mu), 3 - 2 * mu, -mu, 2 * mu, -mu]
    elif which == 2:
        c = [1.0, 3 - mu, 3 - 2 * mu, -mu, -2 * mu, -mu]
    elif which == 3:
        c = [1.0, 2 + mu, 1 + 2 * mu, -(1 - mu), -2 * (1 - mu), -(1 - mu)]
    else:
        raise ValueError(which)
    f = np.polyval(c, g)
    fp = np.polyval(np.polyder(c), g)
    return f, fp


def collinear_gamma(mu, which, tol=1e-15, maxiter=50):
    """Distance gamma of L1/L2 from the Moon, or of L3 from Earth (nondimensional)."""
    if which in (1, 2):
        g = (mu / 3) ** (1 / 3)               # Hill-sphere first guess
    else:
        g = 1 - 7 * mu / 12
    for _ in range(maxiter):
        f, fp = _collinear_quintic(g, mu, which)
        dg = f / fp
        g -= dg
        if abs(dg) < tol * max(1.0, abs(g)):
            return g
    raise RuntimeError(f"Newton iteration for L{which} did not converge")


def lagrange_points(mu) -> dict[str, np.ndarray]:
    """Positions of L1..L5 in the barycentric rotating frame."""
    g1 = collinear_gamma(mu, 1)
    g2 = collinear_gamma(mu, 2)
    g3 = collinear_gamma(mu, 3)
    return {
        "L1": np.array([1 - mu - g1, 0.0, 0.0]),
        "L2": np.array([1 - mu + g2, 0.0, 0.0]),
        "L3": np.array([-mu - g3, 0.0, 0.0]),
        "L4": np.array([0.5 - mu, np.sqrt(3) / 2, 0.0]),
        "L5": np.array([0.5 - mu, -np.sqrt(3) / 2, 0.0]),
    }


def gradient_U(r, mu):
    """Gradient of the pseudo-potential; zero at every libration point."""
    x, y, z = r
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    a1, a2 = (1 - mu) / r1**3, mu / r2**3
    return np.array([x - a1 * (x + mu) - a2 * (x - 1 + mu), y - (a1 + a2) * y, -(a1 + a2) * z])
