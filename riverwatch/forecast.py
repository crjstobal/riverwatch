"""Short-term forecast for the crossing: probability that the stage reaches the close-crossing level within
the next 60 minutes, from the recent stage at the crossing and at the gage upstream.

Physics behind the features: a flood wave from upstream reaches the crossing about 80 minutes later (median over
36 events since 2017), so the upstream rise over the last 1 to 4 hours says what is coming; the local rate of rise
covers rain that falls on the crossing's own catchment, where the upstream gage sees nothing.

The model is a small gradient-boosted tree ensemble exported to JSON and evaluated with numpy (no scikit-learn in
the Lambda package), combined with a physical rule on the upstream rise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STEP = 900  # 15-minute grid

FEATURES = ["depth_over_deck", "rise_15m", "rise_60m", "above_24h_min", "up_rise_60m", "up_rise_120m",
            "up_rise_240m", "up_above_24h_min"]


def lagged(series: np.ndarray, i: int, k: int) -> float:
    j = i - k
    return series[i] - series[j] if j >= 0 else np.nan


def features_at(d: np.ndarray, u: np.ndarray, i: int, deck: float) -> np.ndarray:
    """Features at grid index i using only samples up to i (d: crossing, u: upstream, 15-min grids)."""
    w = slice(max(0, i - 96), i + 1)
    return np.array([
        d[i] - deck,
        lagged(d, i, 1), lagged(d, i, 4), d[i] - np.nanmin(d[w]),
        lagged(u, i, 4), lagged(u, i, 8), lagged(u, i, 16), u[i] - np.nanmin(u[w]),
    ])


def transform(X: np.ndarray) -> np.ndarray:
    """Positive parts and a saturating transform: rises matter, falls do not, and big rises saturate."""
    Z = X.copy()
    Z[:, 1:] = np.sign(Z[:, 1:]) * np.log1p(np.abs(Z[:, 1:]))
    rises = np.maximum(Z[:, [1, 2, 4, 5, 6]], 0)
    return np.c_[Z, rises, Z[:, 0] * (Z[:, 0] > -1)]


UPSTREAM_RULE_FT = 1.0  # upstream rise over 2 hours that alone triggers the warning (physical rule)


class Model:
    """Gradient-boosted trees exported from scikit-learn to JSON, evaluated with numpy, plus the upstream rule."""

    def __init__(self, params: dict):
        self.p = params
        self.base = params["baseline"]
        self.trees = [{k: np.array(v) for k, v in tr.items()} for tr in params["trees"]]
        self.threshold = params.get("threshold", 0.5)

    @classmethod
    def load(cls, path: Path) -> "Model":
        return cls(json.loads(Path(path).read_text()))

    def raw(self, x: np.ndarray) -> float:
        total = self.base
        for tr in self.trees:
            n = 0
            while not tr["is_leaf"][n]:
                v = x[tr["feature"][n]]
                go_left = (v <= tr["threshold"][n]) if np.isfinite(v) else bool(tr["missing_left"][n])
                n = int(tr["left"][n] if go_left else tr["right"][n])
            total += tr["value"][n]
        return float(total)

    def prob(self, x: np.ndarray) -> float:
        return 1.0 / (1.0 + np.exp(-self.raw(x)))

    def warn(self, x: np.ndarray) -> tuple[bool, str]:
        p = self.prob(x)
        if x[5] >= UPSTREAM_RULE_FT:
            return True, f"upstream gage rose {x[5]:.1f} ft in 2 h"
        if p >= self.threshold:
            return True, f"model probability {p:.2f}"
        return False, ""


def from_series(model: Model, down: list[tuple[float, float]], up: list[tuple[float, float]], deck: float,
                at: float) -> dict | None:
    """Forecast from raw (t, value) series as returned by the gage tools.

    Full mode needs both gages up to date. If the crossing series is missing (silent gage, night), it degrades to the
    upstream rule alone and says so. None if the upstream gage is missing too.
    """
    ut = np.array([p[0] for p in up]) if up else np.array([])
    up_ok = len(up) >= 5 and ut[-1] >= at - 3600
    down_ok = len(down) >= 5 and down[-1][0] >= at - 3600
    if not up_ok:
        return None
    grid = np.arange(at - 24 * 3600, at + 1, STEP)
    u = np.interp(grid, ut, np.array([p[1] for p in up]))
    if not down_ok:
        rise = float(u[-1] - u[-9])
        warn = rise >= UPSTREAM_RULE_FT
        return {"mode": "upstream_only", "p_close_within_60min": None, "warning": warn,
                "why": f"upstream gage rose {rise:.1f} ft in 2 h" if warn else "",
                "note": "crossing gage series unavailable: only the upstream rule can warn; local rain is not covered",
                "inputs": {"up_rise_120m": round(rise, 2)}}
    d = np.interp(grid, np.array([p[0] for p in down]), np.array([p[1] for p in down]))
    x = features_at(d, u, len(grid) - 1, deck)
    if not np.isfinite(x).all():
        return None
    warn, why = model.warn(x)
    return {"mode": "full", "p_close_within_60min": round(model.prob(x), 3), "warning": warn, "why": why,
            "inputs": {k: round(float(v), 2) for k, v in zip(FEATURES, x)}}
