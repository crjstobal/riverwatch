"""USGS data access: HIVIS camera frames (NIMS API, public S3 bucket) and NWIS instantaneous gage values.

Every function takes an `at` timestamp (UTC seconds). Nothing after `at` is ever returned, so a replay of
a past event sees exactly what a live system would have seen at that moment.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

NIMS = "https://api.waterdata.usgs.gov/nims/v0"
NWIS_IV = "https://waterservices.usgs.gov/nwis/iv/"
CACHE = Path("/tmp/riverwatch-cache")


def _get(url: str, raw: bool = False, tries: int = 4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RiverWatch/1.0 (OpenCV AI Competition 2026)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                b = r.read()
            return b if raw else json.loads(b)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def frame_time(name: str) -> float:
    return datetime.strptime(name.rsplit("___", 1)[1][:-4], "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=timezone.utc).timestamp()


@lru_cache(maxsize=64)
def camera(cam_id: str) -> dict:
    return _get(f"{NIMS}/cameras?camId={urllib.parse.quote(cam_id)}")[0]


def frames_before(cam_id: str, at: float, n: int = 2, window_h: float = 3.0) -> list[str]:
    """Names of the n most recent frames at or before `at` (newest first)."""
    q = urllib.parse.urlencode({"camId": cam_id, "after": iso(at - window_h * 3600), "before": iso(at),
                                "limit": 50, "recent": "true"})
    names = _get(f"{NIMS}/listFiles?{q}")
    names = [x for x in names if frame_time(x) <= at]
    return sorted(names, key=frame_time, reverse=True)[:n]


def load_frame(cam_id: str, name: str, local_dir: Path | None = None) -> np.ndarray | None:
    """720 px frame. Uses a local copy if present (evaluation data), else the USGS public S3 bucket."""
    if local_dir is not None and (local_dir / name).exists():
        return cv2.imread(str(local_dir / name))
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists():
        p.write_bytes(_get(camera(cam_id)["smallDir"] + name, raw=True))
    return cv2.imread(str(p))


def gage_sensors(site: str, at: float, hours: float = 6.0) -> dict[str, list[tuple[float, float]]]:
    """Gage height (ft) per sensor (method description) for the `hours` before `at`."""
    start, end = iso(at - hours * 3600), iso(at)
    try:
        d = _get(f"{NWIS_IV}?format=json&sites={site}&parameterCd=00065&startDT={start}&endDT={end}")
    except Exception:
        return {}
    out = {}
    for s in d["value"]["timeSeries"]:
        for b in s["values"]:
            name = (b["method"][0].get("methodDescription") or "sensor").strip("[]")
            pts = []
            for v in b["value"]:
                t = datetime.fromisoformat(v["dateTime"]).timestamp()
                val = float(v["value"])
                if t <= at and val > -100:
                    pts.append((t, val))
            out[name] = sorted(pts)
    return out


def gage_series(site: str, at: float, hours: float = 6.0) -> list[tuple[float, float]]:
    """Primary-sensor gage height (ft); the only series used for decisions. Empty if the primary is silent.

    Sites with a single sensor publish it without the word "primary"; then that sensor is the primary.
    """
    sensors = gage_sensors(site, at, hours)
    primary = [v for k, v in sensors.items() if "primary" in k.lower()]
    if primary:
        return primary[0]
    if len(sensors) == 1:
        return next(iter(sensors.values()))
    return []


def trend(series: list[tuple[float, float]], minutes: float = 60.0) -> float | None:
    """Rate of change in ft/h over the last `minutes` (least squares)."""
    if len(series) < 2:
        return None
    t_end = series[-1][0]
    pts = [(t, v) for t, v in series if t >= t_end - minutes * 60]
    if len(pts) < 2:
        return None
    t = np.array([p[0] for p in pts]) / 3600.0
    v = np.array([p[1] for p in pts])
    return float(np.polyfit(t - t[0], v, 1)[0])
