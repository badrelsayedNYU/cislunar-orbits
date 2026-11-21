"""Stable and unstable invariant manifolds of unstable periodic orbits.

Procedure (Folta et al. 2022, sec. 6.13; Koon, Lo, Marsden & Ross):
  1. Monodromy M = Phi(T, 0). Its real eigenvalues lambda_u > 1 and lambda_s = 1/lambda_u give
     the unstable and stable directions v_u(0), v_s(0).
  2. Map them around the orbit with the STM, v(t) = Phi(t, 0) v(0), and normalise so that the
     position part has unit length. A step of size d then moves the position by exactly d.
  3. X_manifold(t) = X_po(t) +/- d v(t). Unstable manifolds are propagated forward in time,
     stable manifolds backward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import ATOL, METHOD, RTOL, eom, eom_stm, jacobi
from .stability import monodromy


def eigendirections(M):
    """Unit (position-normalised) unstable and stable eigenvectors of the monodromy matrix."""
    lam, V = np.linalg.eig(M)
    real = np.abs(lam.imag) < 1e-9 * np.abs(lam)
    lr = np.where(real, lam.real, np.nan)
    iu = int(np.nanargmax(np.abs(lr)))
    is_ = int(np.nanargmin(np.abs(lr)))
    vu, vs = V[:, iu].real, V[:, is_].real
    return lam[iu].real, vu / np.linalg.norm(vu[:3]), lam[is_].real, vs / np.linalg.norm(vs[:3])


@dataclass
class ManifoldSeeds:
    t: np.ndarray            # phase times along the orbit
    X_orbit: np.ndarray      # (n, 6) orbit states
    v: np.ndarray            # (n, 6) eigendirection at each phase, position part unit length
    lam: float


def seeds(X0, T, mu, n, kind="unstable", rtol=RTOL, atol=ATOL) -> ManifoldSeeds:
    """Orbit states and mapped eigendirections at n phases t_k = k T / n."""
    M, _ = monodromy(X0, T, mu, rtol, atol)
    lu, vu, ls, vs = eigendirections(M)
    v0, lam = (vu, lu) if kind == "unstable" else (vs, ls)
    t = np.linspace(0, T, n, endpoint=False)
    Y0 = np.concatenate([X0, np.eye(6).ravel()])
    sol = solve_ivp(eom_stm, (0, T), Y0, method=METHOD, rtol=rtol, atol=atol, t_eval=t,
                    args=(mu,))
    Xo = sol.y[:6].T
    v = np.empty((n, 6))
    for k in range(n):
        Phi = sol.y[6:, k].reshape(6, 6)
        w = Phi @ v0
        v[k] = w / np.linalg.norm(w[:3])
    return ManifoldSeeds(t=t, X_orbit=Xo, v=v, lam=lam)


R_EARTH_KM = 6378.1   # equatorial radius (IAU/WGS-84 value), used only to flag impacts


def collision_event(center_x, radius):
    def ev(t, X, mu):
        return np.sqrt((X[0] - center_x) ** 2 + X[1] ** 2 + X[2] ** 2) - radius
    ev.terminal = True
    ev.direction = -1
    return ev


def section_event(x_section):
    def ev(t, X, mu):
        return X[0] - x_section
    ev.terminal = True
    ev.direction = 0
    return ev


@dataclass
class ManifoldArc:
    t: np.ndarray
    X: np.ndarray
    hit: bool                # reached the section
    impact: str | None       # "moon" / "earth" if the arc hit a primary's surface first
    X_section: np.ndarray | None
    t_section: float | None
    C0: float                # Jacobi constant of the perturbed initial state
    C_drift: float           # max |C(t) - C0| along the arc (integration error)


def propagate_branch(X_start, mu, *, backward, x_section=None, t_max=8.0, n_out=600,
                     rtol=RTOL, atol=ATOL, r_moon=None, r_earth=None) -> ManifoldArc:
    """Propagate one manifold trajectory until the section, an impact, or |t| = t_max.

    r_moon / r_earth (nondimensional) stop the arc at the surface of the primary; the CR3BP
    treats the primaries as points, so without this an arc can pass through the Moon.
    """
    from .units import LU_KM, R_MOON_KM

    r_moon = R_MOON_KM / LU_KM if r_moon is None else r_moon
    r_earth = R_EARTH_KM / LU_KM if r_earth is None else r_earth
    tf = -t_max if backward else t_max
    events = [section_event(x_section) if x_section is not None else (lambda t, X, mu: 1.0),
              collision_event(1 - mu, r_moon), collision_event(-mu, r_earth)]
    if x_section is None:
        events[0].terminal = False
    sol = solve_ivp(eom, (0, tf), X_start, method=METHOD, rtol=rtol, atol=atol, args=(mu,),
                    events=events, dense_output=True)
    t_end = sol.t[-1]
    ts = np.linspace(0, t_end, n_out)
    X = sol.sol(ts).T
    C = jacobi(X, mu)
    C0 = float(jacobi(X_start, mu))
    hit = x_section is not None and len(sol.t_events[0]) > 0
    impact = "moon" if len(sol.t_events[1]) else ("earth" if len(sol.t_events[2]) else None)
    return ManifoldArc(t=ts, X=X, hit=hit, impact=impact,
                       X_section=sol.y_events[0][0] if hit else None,
                       t_section=float(sol.t_events[0][0]) if hit else None,
                       C0=C0, C_drift=float(np.max(np.abs(C - C0))))


def branch_sign(seed_X, seed_v, d, mu, *, backward, toward_x_greater, t_probe=2.0,
                rtol=RTOL, atol=ATOL):
    """+1 or -1: the sign of the step whose trajectory heads toward larger x (or smaller x).

    Decided once per orbit from the phase-0 seed: the two half-manifolds leave in opposite
    directions, one into the Moon region and one away from it.
    """
    out = {}
    for s in (+1, -1):
        a = propagate_branch(seed_X + s * d * seed_v, mu, backward=backward, t_max=t_probe,
                             n_out=50, rtol=rtol, atol=atol)
        out[s] = a.X[-1, 0]
    best = max(out, key=out.get) if toward_x_greater else min(out, key=out.get)
    return best
