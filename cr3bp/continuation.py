"""Family continuation: natural-parameter first, pseudo-arclength when natural steps fail.

Each family member is stored as an Orbit with its initial state, period, Jacobi constant and
correction diagnostics. Monodromy/stability are computed afterwards (stability.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .correction import (CorrectionError, V_from_state, correct, family_tangent,
                         natural_constraint, pseudo_arclength_constraint, state_from_V)
from .dynamics import jacobi
from .lagrange import lagrange_points


@dataclass
class Orbit:
    X0: np.ndarray
    T: float
    C: float
    residual: float
    rtol: float
    method: str          # "natural" or "pseudo-arclength"


@dataclass
class Family:
    name: str
    mu: float
    planar: bool
    orbits: list = field(default_factory=list)
    log: list = field(default_factory=list)
    switched_to_pac_at: int | None = None

    def arrays(self):
        X0 = np.array([o.X0 for o in self.orbits])
        T = np.array([o.T for o in self.orbits])
        C = np.array([o.C for o in self.orbits])
        res = np.array([o.residual for o in self.orbits])
        rtol = np.array([o.rtol for o in self.orbits])
        method = np.array([o.method for o in self.orbits])
        return dict(X0=X0, T=T, C=C, residual=res, rtol=rtol, method=method)


def lyapunov_linear_guess(mu, which: int, Ax: float):
    """Planar Lyapunov initial guess from the linearization about L1/L2.

    xi'' - 2 eta' = Uxx xi,  eta'' + 2 xi' = Uyy eta.  The center frequency omega satisfies
    omega^4 - (Uxx + Uyy - 4) omega^2 + Uxx Uyy = 0 (smaller root of the in-plane pair), and
    xi = -Ax cos(omega t), eta = beta Ax sin(omega t), beta = (omega^2 + Uxx) / (2 omega).
    Starting at the Earth-side (L1) or far-side (L2) x-axis crossing matches the JPL catalog.
    Returns (X0, T).
    """
    from .dynamics import hessian_U

    L = lagrange_points(mu)[f"L{which}"]
    H = hessian_U(L, mu)
    Uxx, Uyy = H[0, 0], H[1, 1]
    # lambda^4 + (4 - Uxx - Uyy) lambda^2 + Uxx Uyy = 0 with lambda = i omega
    b = 4 - Uxx - Uyy
    lam2 = (-b + np.array([1, -1]) * np.sqrt(b * b - 4 * Uxx * Uyy)) / 2
    omega = np.sqrt(-lam2[lam2 < 0][0])
    beta = (omega**2 + Uxx) / (2 * omega)
    sgn = -1.0 if which == 1 else 1.0          # start on the Earth side of L1, far side of L2
    xi0 = sgn * Ax
    X0 = np.array([L[0] + xi0, 0, 0, 0, -beta * omega * xi0, 0])
    return X0, 2 * np.pi / omega


def continue_family(family: Family, V_start, *, n_max, ds=1e-3, ds_min=1e-6, ds_max=0.02,
                    natural_index=0, natural_step=None, natural_step_max=None, start_with_natural=True,
                    stop=None, rtol_fn=None, tangent0=None, tol=1e-12, verbose=False,
                    max_natural=None, start_extra=None):
    """Grow `family` from the corrected free-variable vector V_start.

    Natural-parameter steps change V[natural_index] by natural_step with a secant/tangent
    predictor. If a natural step fails to converge after halving down to ds_min, the method
    switches permanently to pseudo-arclength continuation (recorded in family.switched_to_pac_at).
    stop(orbit, V) -> bool ends the continuation; rtol_fn(V) -> tolerance chooses rtol = atol.

    natural_index="auto" fixes, at every step, the free variable (excluding T/2) with the largest
    component in the family tangent, i.e. the one changing fastest along the family there.
    natural_step is then the signed change of that variable along the current tangent direction.
    max_natural switches to pseudo-arclength after that many members even if natural steps work.
    start_extra is an optional extra constraint for the first correction (e.g. fixing z0 when
    stepping off a bifurcation, where the minimum-norm step could fall back onto the planar family).
    """
    mu, planar = family.mu, family.planar
    rtol_fn = rtol_fn or (lambda V: 1e-12)

    def add(V, method, residual, rt):
        X0 = state_from_V(V, planar)
        o = Orbit(X0=X0, T=2 * V[-1], C=float(jacobi(X0, mu)), residual=residual, rtol=rt,
                  method=method)
        family.orbits.append(o)
        return o

    rt = rtol_fn(V_start)
    c0 = correct(V_start, mu, planar, extra=start_extra, tol=tol, rtol=rt, atol=rt)
    V = c0.V
    tangent = family_tangent(c0.DF, tangent0)
    auto = natural_index == "auto"
    k0 = int(np.argmax(np.abs(tangent[:-1]))) if auto else natural_index
    if tangent0 is None and natural_step is not None and np.sign(tangent[k0]) != np.sign(natural_step):
        tangent = -tangent
    add(V, "natural" if start_with_natural else "pseudo-arclength", c0.residual, rt)

    use_natural = start_with_natural and natural_step is not None
    dn = natural_step
    h = ds
    while len(family.orbits) < n_max:
        rt = rtol_fn(V)
        try:
            if use_natural:
                k = int(np.argmax(np.abs(tangent[:-1]))) if auto else natural_index
                # tangent predictor scaled so the natural parameter advances by dn
                # (for "auto", dn is a magnitude taken along the current tangent direction)
                step_k = abs(dn) * np.sign(tangent[k]) if auto else dn
                Vg = V + (step_k / tangent[k]) * tangent
                extra = natural_constraint(k, V[k] + step_k)
                cr = correct(Vg, mu, planar, extra=extra, tol=tol, rtol=rt, atol=rt)
            else:
                Vg = V + h * tangent
                extra = pseudo_arclength_constraint(V, tangent, h)
                cr = correct(Vg, mu, planar, extra=extra, tol=tol, rtol=rt, atol=rt)
        except (CorrectionError, np.linalg.LinAlgError, RuntimeError) as exc:
            if use_natural:
                dn /= 2
                if abs(dn) < ds_min:
                    use_natural = False
                    family.switched_to_pac_at = len(family.orbits)
                    family.log.append(f"natural-parameter steps failed at member "
                                      f"{len(family.orbits)}; switched to pseudo-arclength")
                    if verbose:
                        print(family.log[-1])
            else:
                h /= 2
                if h < ds_min:
                    family.log.append(f"pseudo-arclength step below ds_min at member "
                                      f"{len(family.orbits)}: {exc}")
                    break
            continue
        V_new = cr.V
        tangent = family_tangent(cr.DF, tangent)
        o = add(V_new, "natural" if use_natural else "pseudo-arclength", cr.residual, rt)
        V = V_new
        if verbose and len(family.orbits) % 25 == 0:
            print(f"  {family.name} #{len(family.orbits)}: C = {o.C:.6f}, T = {o.T:.5f}, "
                  f"iters = {cr.iterations}, step = {dn if use_natural else h:.2e}")
        # step-size control
        if use_natural:
            if cr.iterations <= 3:
                dn = np.sign(dn) * min(abs(dn) * 1.5, natural_step_max or abs(natural_step) * 8)
        else:
            if cr.iterations <= 3:
                h = min(h * 1.5, ds_max)
            elif cr.iterations >= 6:
                h = max(h / 2, ds_min)
        if stop is not None and stop(o, V):
            break
        if use_natural and max_natural is not None and len(family.orbits) >= max_natural:
            use_natural = False
            family.switched_to_pac_at = len(family.orbits)
            family.log.append(f"natural-parameter continuation used for the first {max_natural} "
                              f"members; pseudo-arclength from member {max_natural}")
    return family


def find_tangent_bifurcation(mu, V_a, V_b, *, natural_index=0, rtol=1e-12, xtol=1e-13):
    """Locate the planar-family member whose out-of-plane pair sits at +1 (s_z = 2).

    V_a and V_b are planar free-variable vectors bracketing the sign change of s_z - 2.
    Each evaluation corrects the planar orbit at fixed V[natural_index] (Brent's method on
    that parameter). Returns (V_bif, s_z - 2 at V_bif).
    """
    from scipy.optimize import brentq

    from .correction import correct
    from .stability import monodromy, out_of_plane_trace

    cache = {"V": V_a}

    def f(p):
        guess = cache["V"].copy()
        # linear interpolation between the bracketing members as the initial guess
        w = (p - V_a[natural_index]) / (V_b[natural_index] - V_a[natural_index])
        guess = (1 - w) * V_a + w * V_b
        c = correct(guess, mu, True, extra=natural_constraint(natural_index, p), rtol=rtol,
                    atol=rtol)
        cache["V"] = c.V
        M, _ = monodromy(state_from_V(c.V, True), 2 * c.V[-1], mu, rtol, rtol)
        return out_of_plane_trace(M) - 2.0

    p = brentq(f, V_a[natural_index], V_b[natural_index], xtol=xtol, rtol=4 * np.finfo(float).eps)
    val = f(p)
    return cache["V"], val
