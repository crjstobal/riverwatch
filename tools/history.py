"""Download the long gage-height record of the crossing and of the upstream gage, resampled to 15 minutes.

usage: history.py [start=2017-04-12] [end=2026-10-25]    writes data/history/<site>.npz (t: UTC seconds, v: ft)
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/history"
SITES = ["08101300", "08101000"]
IV = "https://waterservices.usgs.gov/nwis/iv/"


def fetch(site, start, end):
    url = f"{IV}?format=json&sites={site}&parameterCd=00065&startDT={start}&endDT={end}&siteStatus=all"
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                d = json.load(r)
            break
        except Exception as e:
            print("retry", site, start, e); time.sleep(5 * (i + 1))
    else:
        return []
    blocks = [b for s in d["value"]["timeSeries"] for b in s["values"] if b["value"]]
    if not blocks:
        return []
    # primary sensor when several are published; otherwise the longest series
    prim = [b for b in blocks if "primary" in (b["method"][0].get("methodDescription") or "").lower()]
    block = prim[0] if prim else max(blocks, key=lambda b: len(b["value"]))
    out = []
    for v in block["value"]:
        val = float(v["value"])
        if val > -100:
            out.append((datetime.fromisoformat(v["dateTime"]).timestamp(), val))
    return out


def resample(pts, step=900):
    if not pts:
        return np.array([]), np.array([])
    t = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts])
    o = np.argsort(t); t, v = t[o], v[o]
    grid = np.arange(np.ceil(t[0] / step) * step, t[-1], step)
    j = np.searchsorted(t, grid)
    j = np.clip(j, 1, len(t) - 1)
    near = np.minimum(np.abs(t[j] - grid), np.abs(t[j - 1] - grid))
    vals = np.interp(grid, t, v)
    vals[near > 3600] = np.nan  # gaps longer than an hour stay gaps
    return grid, vals


def main(start="2017-04-12", end="2026-10-25"):
    OUT.mkdir(parents=True, exist_ok=True)
    for site in SITES:
        pts = []
        d0 = datetime.fromisoformat(start)
        while d0 < datetime.fromisoformat(end):
            d1 = min(d0 + timedelta(days=180), datetime.fromisoformat(end))
            chunk = fetch(site, d0.date().isoformat(), d1.date().isoformat())
            pts += chunk
            print(site, d0.date(), len(chunk))
            d0 = d1
        g, v = resample(pts)
        np.savez_compressed(OUT / f"{site}.npz", t=g, v=v)
        print(site, "samples", len(g), "valid", int(np.isfinite(v).sum()))


if __name__ == "__main__":
    main(*sys.argv[1:])
