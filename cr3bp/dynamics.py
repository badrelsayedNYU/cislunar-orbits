"""CR3BP equations of motion, Jacobi constant, variational equations and propagation.

Rotating frame centred at the barycentre, Earth at (-mu, 0, 0), Moon at (1 - mu, 0, 0),
nondimensional units (LU = Earth-Moon distance, TU = 1/n). State X = [x, y, z, vx, vy, vz].

    x'' - 2 y' = dU/dx,   y'' + 2 x' = dU/dy,   z'' = dU/dz,
    U = (x^2 + y^2)/2 + (1 - mu)/r1 + mu/r2,
    C = 2U - (vx^2 + vy^2 + vz^2)        (Jacobi constant)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.integrate import solve_ivp

RTOL = 1e-12
ATOL = 1e-12
METHOD = "DOP853"
T_SKIP = 1e-3   # event-free initial step in propagate_to_crossing (TU, ~1.1 h)


def eom(t, X, mu):
    x, y, z, vx, vy, vz = X[:6]
    dx1 = x + mu
    dx2 = x - 1.0 + mu
    r1 = sqrt(dx1 * dx1 + y * y + z * z)
    r2 = sqrt(dx2 * dx2 + y * y + z * z)
    a1 = (1.0 - mu) / (r1 * r1 * r1)
    a2 = mu / (r2 * r2 * r2)
    ax = 2.0 * vy + x - a1 * dx1 - a2 * dx2
    ay = -2.0 * vx + y - (a1 + a2) * y
    az = -(a1 + a2) * z
    return np.array([vx, vy, vz, ax, ay, az])


def pseudo_potential(X, mu):
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    return 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2


def jacobi(X, mu):
    """Jacobi constant C = 2U - v^2 for one state (6,) or many (..., 6)."""
    X = np.asarray(X, dtype=float)
    v2 = X[..., 3] ** 2 + X[..., 4] ** 2 + X[..., 5] ** 2
    return 2.0 * pseudo_potential(X, mu) - v2


def hessian_U(r, mu):
    """Second derivatives of the pseudo-potential, 3x3."""
    x, y, z = r[0], r[1], r[2]
    dx1, dx2 = x + mu, x - 1 + mu
    r1 = sqrt(dx1 * dx1 + y * y + z * z)
    r2 = sqrt(dx2 * dx2 + y * y + z * z)
    m1, m2 = 1 - mu, mu
    r13, r23 = r1**3, r2**3
    r15, r25 = r13 * r1 * r1, r23 * r2 * r2
    c = m1 / r13 + m2 / r23
    Uxx = 1 - c + 3 * (m1 * dx1 * dx1 / r15 + m2 * dx2 * dx2 / r25)
    Uyy = 1 - c + 3 * y * y * (m1 / r15 + m2 / r25)
    Uzz = -c + 3 * z * z * (m1 / r15 + m2 / r25)
    Uxy = 3 * y * (m1 * dx1 / r15 + m2 * dx2 / r25)
    Uxz = 3 * z * (m1 * dx1 / r15 + m2 * dx2 / r25)
    Uyz = 3 * y * z * (m1 / r15 + m2 / r25)
    return np.array([[Uxx, Uxy, Uxz], [Uxy, Uyy, Uyz], [Uxz, Uyz, Uzz]])


def jacobian(X, mu):
    """A = df/dX, the 6x6 linearization used in the variational equations."""
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = hessian_U(X, mu)
    A[3, 4] = 2.0
    A[4, 3] = -2.0
    return A


def eom_stm(t, Y, mu):
    """42-state system: state (6) followed by the row-major STM (36), dPhi/dt = A(t) Phi."""
    X = Y[:6]
    Phi = Y[6:].reshape(6, 6)
    dPhi = np.empty((6, 6))
    dPhi[0:3] = Phi[3:6]
    H = hessian_U(X, mu)
    dPhi[3:6] = H @ Phi[0:3]
    dPhi[3] += 2.0 * Phi[4]
    dPhi[4] -= 2.0 * Phi[3]
    out = np.empty(42)
    out[:6] = eom(t, X, mu)
    out[6:] = dPhi.ravel()
    return out


@dataclass
class Arc:
    t: np.ndarray            # (n,) output times
    X: np.ndarray            # (n, 6) states
    Phi: np.ndarray | None   # (6, 6) STM at the final time (if requested)
    t_events: list
    X_events: list
    sol: object              # dense output (if requested)

    @property
    def Xf(self):
        return self.X[-1]

    @property
    def tf(self):
        return self.t[-1]


def propagate(X0, tf, mu, *, stm=False, events=None, t_eval=None, dense=False,
              rtol=RTOL, atol=ATOL, max_step=np.inf) -> Arc:
    """Integrate from t = 0 to tf (tf may be negative) with DOP853."""
    X0 = np.asarray(X0, dtype=float)
    if stm:
        Y0 = np.concatenate([X0, np.eye(6).ravel()])
        f = eom_stm
    else:
        Y0 = X0
        f = eom
    res = solve_ivp(f, (0.0, tf), Y0, method=METHOD, rtol=rtol, atol=atol, args=(mu,),
                    events=events, t_eval=t_eval, dense_output=dense, max_step=max_step)
    if res.status == -1:
        raise RuntimeError(f"integration failed: {res.message}")
    Y = res.y.T
    Phi = Y[-1, 6:].reshape(6, 6) if stm else None
    t_ev = list(res.t_events) if events is not None else []
    X_ev = [y[:, :6] for y in res.y_events] if events is not None else []
    return Arc(t=res.t, X=Y[:, :6], Phi=Phi, t_events=t_ev, X_events=X_ev, sol=res.sol)


def y_crossing_event(direction=0.0, terminal=True):
    """Event function for crossing the x-z plane (y = 0)."""

    def ev(t, Y, mu):
        return Y[1]

    ev.terminal = terminal
    ev.direction = direction
    return ev


def propagate_to_crossing(X0, mu, *, stm=False, n_cross=1, t_max=20.0, direction=0.0,
                          rtol=RTOL, atol=ATOL) -> Arc:
    """Propagate until the n-th crossing of y = 0 (ignoring the start point at t = 0).

    The first crossing is found after a short initial step so a state starting on y = 0 is not
    counted. The final state lies on the plane to the event-location tolerance (~1e-14).
    """
    # Event detection would fire at t = 0 for a start point on the plane, so take a short
    # event-free step first and chain the two arcs (and their STMs).
    first = propagate(X0, T_SKIP, mu, stm=stm, rtol=rtol, atol=atol)
    ev = y_crossing_event(direction=direction, terminal=n_cross)
    rest = propagate(first.Xf, t_max - T_SKIP, mu, stm=stm, events=[ev], rtol=rtol, atol=atol)
    if len(rest.t_events[0]) < n_cross:
        raise RuntimeError(f"no y = 0 crossing #{n_cross} found within t = {t_max}")
    return Arc(
        t=np.concatenate([first.t, rest.t[1:] + T_SKIP]),
        X=np.vstack([first.X, rest.X[1:]]),
        Phi=rest.Phi @ first.Phi if stm else None,
        t_events=[rest.t_events[0] + T_SKIP],
        X_events=rest.X_events,
        sol=None,
    )


def apses(X0, T, mu, body="moon", rtol=RTOL, atol=None, n=4001):
    """(r_min, r_max, t_min, t_max) of the distance to the Moon (or Earth) over one period.

    Dense sampling of the DOP853 interpolant, then bounded scalar refinement of each extremum.
    """
    from scipy.optimize import minimize_scalar

    atol = rtol if atol is None else atol
    c = np.array([1 - mu, 0, 0]) if body == "moon" else np.array([-mu, 0, 0])
    arc = propagate(X0, T, mu, dense=True, rtol=rtol, atol=atol)
    t = np.linspace(0, T, n)
    r = np.linalg.norm(arc.sol(t)[:3].T - c, axis=1)

    def dist(tt):
        return np.linalg.norm(arc.sol(tt)[:3] - c)

    out = []
    for i, sgn in ((int(np.argmin(r)), 1.0), (int(np.argmax(r)), -1.0)):
        lo, hi = t[max(i - 1, 0)], t[min(i + 1, n - 1)]
        res = minimize_scalar(lambda tt: sgn * dist(tt), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-13})
        out.append((sgn * res.fun, res.x))
    return out[0][0], out[1][0], out[0][1], out[1][1]
