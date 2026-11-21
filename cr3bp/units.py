"""Nondimensional <-> dimensional conversions using the JPL catalog's own Earth-Moon units.

Length unit LU = Earth-Moon distance, time unit TU = 1 / (mean motion of the primaries),
both read from the cached API response so that every comparison uses JPL's constants.
"""

from __future__ import annotations

from .jpl import system

_S = system()

MU: float = _S["mu"]                 # m_Moon / (m_Earth + m_Moon)
LU_KM: float = _S["lunit_km"]        # km per length unit
TU_S: float = _S["tunit_s"]          # seconds per time unit
TU_DAYS: float = TU_S / 86400.0
VU_KMS: float = LU_KM / TU_S         # km/s per velocity unit
R_MOON_KM: float = _S["radius_moon_km"]

SYNODIC_MONTH_DAYS = 29.530589       # mean lunar synodic month, as given in the project spec


def km_to_nd(km):
    return km / LU_KM


def nd_to_km(lu):
    return lu * LU_KM


def days_to_nd(days):
    return days / TU_DAYS


def nd_to_days(tu):
    return tu * TU_DAYS


def nd_to_kms(vu):
    return vu * VU_KMS


def kms_to_nd(kms):
    return kms / VU_KMS
