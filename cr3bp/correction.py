"""Single-shooting differential correction for x-z-plane-symmetric periodic orbits.

A symmetric periodic orbit starts perpendicular to the x-z plane, X0 = [x0, 0, z0, 0, vy0, 0],
and crosses it perpendicularly again at T/2. With the free-variable vector

    V = [x0, z0, vy0, T/2]            (spatial)     or   [x0, vy0, T/2]   (planar)

the half-period constraints are

    F(V) = [y(T/2), vx(T/2), vz(T/2)] = 0     (planar: [y, vx]),

with Jacobian DF = [ Phi(T/2)[rows, cols] | f(X(T/2))[rows] ]. F has one fewer equation than
unknowns: its one-dimensional null space is the tangent to the family. A square system is formed
by appending one extra scalar constraint g(V) = 0:

  * natural parameter:    V[k] - value
  * pseudo-arclength:     (V - V*) . tangent* - ds
  * fixed Jacobi constant C(X0) - C*

or, with no extra constraint, Newton takes the minimum-norm step (used to correct a rough guess).
Integration is to the fixed time T/2 (not event-terminated) so that T is an explicit unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import ATOL, RTOL, eom, jacobi, propagate
from .lagrange import gradient_U


class CorrectionError(RuntimeError):
    pass


def free_indices(planar: bool):
    """(state indices of the free variables, state rows of the constraints)."""
    return ([0, 4], [1, 3]) if planar else ([0, 2, 4], [1, 3, 5])


def state_from_V(V, planar: bool):
    if planar:
        x0, vy0 = V[0], V[1]
        return np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
    x0, z0, vy0 = V[0], V[1], V[2]
    return np.array([x0, 0.0, z0, 0.0, vy0, 0.0])


def V_from_state(X0, T, planar: bool):
    if planar:
        return np.array([X0[0], X0[4], T / 2])
    return np.array([X0[0], X0[2], X0[4], T / 2])


def half_period_map(V, mu, planar, rtol=RTOL, atol=ATOL):
    """F(V) and DF(V) for the perpendicular-crossing constraints."""
    cols, rows = free_indices(planar)
    X0 = state_from_V(V, planar)
    arc = propagate(X0, V[-1], mu, stm=True, rtol=rtol, atol=atol)
    Xf = arc.Xf
    f = eom(0.0, Xf, mu)
    F = Xf[rows]
    DF = np.column_stack([arc.Phi[np.ix_(rows, cols)], f[rows]])
    return F, DF


def family_tangent(DF, prev=None):
    """Unit null vector of DF (the family direction), oriented along prev if given."""
    _, _, Vt = np.linalg.svd(DF)
    t = Vt[-1]
    if prev is not None and np.dot(t, prev) < 0:
        t = -t
    return t


# ---- extra scalar constraints ----

def natural_constraint(k, value):
    def g(V):
        dg = np.zeros_like(V)
        dg[k] = 1.0
        return V[k] - value, dg
    return g


def pseudo_arclength_constraint(V_prev, tangent, ds):
    def g(V):
        return np.dot(V - V_prev, tangent) - ds, tangent
    return g


def jacobi_constraint(C_target, mu, planar):
    def g(V):
        X0 = state_from_V(V, planar)
        gradU = gradient_U(X0[:3], mu)
        dg = np.zeros_like(V)
        if planar:
            dg[0] = 2 * gradU[0]
            dg[1] = -2 * X0[4]
        else:
            dg[0] = 2 * gradU[0]
            dg[1] = 2 * gradU[2]
            dg[2] = -2 * X0[4]
        return jacobi(X0, mu) - C_target, dg
    return g


@dataclass
class Correction:
    V: np.ndarray
    DF: np.ndarray           # constraint Jacobian (without the extra row) at the solution
    residual: float          # ||F|| (plus |g|) at the solution
    iterations: int
    history: list = field(default_factory=list)


def correct(V0, mu, planar, extra=None, tol=1e-12, maxiter=20, rtol=RTOL, atol=ATOL,
            max_step=None, tol_floor=1e-10) -> Correction:
    """Newton iteration on F(V) = 0 (and g(V) = 0 if an extra constraint is given).

    Converged when ||F|| < tol, or when ||F|| < tol_floor and a Newton step no longer reduces it
    by half (the residual has reached the integration-noise floor). The full-period closure
    ||X(T) - X0|| is checked separately for every stored orbit.
    """
    V = np.array(V0, dtype=float)
    history = []
    best = None
    for it in range(maxiter + 1):
        F, DF = half_period_map(V, mu, planar, rtol, atol)
        if extra is not None:
            g, dg = extra(V)
            Fa = np.append(F, g)
            DFa = np.vstack([DF, dg])
        else:
            Fa, DFa = F, DF
        err = float(np.linalg.norm(Fa))
        history.append(err)
        if not np.isfinite(err):
            raise CorrectionError("non-finite residual")
        if err < tol:
            return Correction(V=V, DF=DF, residual=err, iterations=it, history=history)
        if best is not None and best[2] < tol_floor and err > 0.5 * best[2]:
            # stalled at the noise floor: return the best iterate
            return Correction(V=best[0], DF=best[1], residual=best[2], iterations=it,
                              history=history)
        if best is None or err < best[2]:
            best = (V.copy(), DF, err)
        if it == maxiter:
            break
        if extra is not None:
            dV = np.linalg.solve(DFa, -Fa)
        else:
            dV = np.linalg.lstsq(DFa, -Fa, rcond=None)[0]
        if max_step is not None:
            n = np.linalg.norm(dV)
            if n > max_step:
                dV *= max_step / n
        V = V + dV
        if V[-1] <= 0:
            raise CorrectionError("half period became non-positive")
    raise CorrectionError(f"no convergence after {maxiter} iterations; history {history}")
