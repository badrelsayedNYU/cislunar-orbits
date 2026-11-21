"""Read the cached JPL Three-Body Periodic Orbits catalog (data/jpl/*.json).

Nothing here touches the network; scripts/fetch_jpl.py fills the cache.
"""

from __future__ import annotations

import functools
import json
import pathlib
from dataclasses import dataclass

import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "jpl"

FAMILIES = ("lyapunov_L1", "lyapunov_L2", "halo_L1_N", "halo_L1_S", "halo_L2_N", "halo_L2_S")


@dataclass(frozen=True)
class CatalogFamily:
    name: str
    states: np.ndarray      # (N, 6) initial conditions [x, y, z, vx, vy, vz], nondimensional
    jacobi: np.ndarray      # (N,)
    period: np.ndarray      # (N,) nondimensional
    stability: np.ndarray   # (N,) JPL stability index
    system: dict            # raw "system" block of the API response
    fetched_utc: str
    request_url: str

    def __len__(self) -> int:
        return len(self.jacobi)


@functools.lru_cache(maxsize=None)
def load_family(name: str) -> CatalogFamily:
    if name not in FAMILIES:
        raise KeyError(f"unknown family {name!r}; choose from {FAMILIES}")
    raw = json.loads((DATA_DIR / f"{name}.json").read_text())
    resp = raw["response"]
    fields = resp["fields"]
    table = np.array(resp["data"], dtype=float)
    col = {f: table[:, i] for i, f in enumerate(fields)}
    states = np.column_stack([col[k] for k in ("x", "y", "z", "vx", "vy", "vz")])
    return CatalogFamily(
        name=name,
        states=states,
        jacobi=col["jacobi"],
        period=col["period"],
        stability=col["stability"],
        system=resp["system"],
        fetched_utc=raw["fetched_utc"],
        request_url=raw["request_url"],
    )


@functools.lru_cache(maxsize=None)
def system() -> dict:
    """Earth-Moon system constants as floats, taken from the cached API response.

    All six cached families must agree; this is checked in the tests.
    """
    s = load_family("halo_L2_S").system
    out = {
        "name": s["name"],
        "mu": float(s["mass_ratio"]),
        "lunit_km": float(s["lunit"]),
        "tunit_s": float(s["tunit"]),
        "radius_moon_km": float(s["radius_secondary"]),
    }
    for k in ("L1", "L2", "L3", "L4", "L5"):
        out[k] = np.array([float(v) for v in s[k]])
    return out
