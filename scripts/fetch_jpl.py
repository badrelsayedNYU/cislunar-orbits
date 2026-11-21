"""Download the Earth-Moon Lyapunov and halo families from the JPL Three-Body Periodic Orbits API.

The raw JSON responses are written to data/jpl/ unchanged, wrapped with the request URL and the
UTC fetch time. Tests and every other script read only from this cache, never from the network.

    python scripts/fetch_jpl.py            # fetch all six families
    python scripts/fetch_jpl.py --force    # overwrite existing cache files

API documentation: https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import ssl
import urllib.parse
import urllib.request

API = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "jpl"

# (cache name, query parameters)
QUERIES = [
    ("lyapunov_L1", {"sys": "earth-moon", "family": "lyapunov", "libr": "1"}),
    ("lyapunov_L2", {"sys": "earth-moon", "family": "lyapunov", "libr": "2"}),
    ("halo_L1_N", {"sys": "earth-moon", "family": "halo", "libr": "1", "branch": "N"}),
    ("halo_L1_S", {"sys": "earth-moon", "family": "halo", "libr": "1", "branch": "S"}),
    ("halo_L2_N", {"sys": "earth-moon", "family": "halo", "libr": "2", "branch": "N"}),
    ("halo_L2_S", {"sys": "earth-moon", "family": "halo", "libr": "2", "branch": "S"}),
]


def _ssl_context() -> ssl.SSLContext:
    # python.org macOS builds ship without a CA bundle; use certifi's if available.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(params: dict) -> tuple[str, dict]:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "cislunar-orbits/0.1 (research script)"})
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp:
        payload = json.load(resp)
    if payload.get("signature", {}).get("version") != "1.0":
        raise RuntimeError(f"Unexpected API version in {url}: {payload.get('signature')}")
    if "warning" in payload or int(payload.get("count", 0)) == 0:
        raise RuntimeError(f"No orbits returned for {url}: {payload}")
    return url, payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="overwrite existing cache files")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, params in QUERIES:
        path = OUT / f"{name}.json"
        if path.exists() and not args.force:
            print(f"{path.name}: cached, skipping (use --force to refresh)")
            continue
        url, payload = fetch(params)
        wrapped = {
            "fetched_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "request_url": url,
            "response": payload,
        }
        path.write_text(json.dumps(wrapped, indent=1))
        print(f"{path.name}: {payload['count']} orbits")


if __name__ == "__main__":
    main()
