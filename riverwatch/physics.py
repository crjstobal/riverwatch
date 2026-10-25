"""Physical plausibility for a stage time series.

A river cannot rise or fall arbitrarily fast. Readings that imply an impossible rate are held back
until the next frames confirm them, and isolated spikes are removed with a short causal median.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class Plausibility:
    def __init__(self, max_rise_ft_per_h: float = 6.0, window: int = 3):
        self.max_rate = max_rise_ft_per_h / 3600.0
        self.buf: deque[tuple[float, float]] = deque(maxlen=window)
        self.last: float | None = None
        self.last_t: float | None = None
        self.rejected = 0

    def reset(self):
        self.buf.clear()
        self.last = self.last_t = None

    def check(self, t: float, value: float) -> tuple[bool, float | None]:
        """Is this reading physically consistent with the accepted series? Returns (ok, implied rate ft/h)."""
        if self.last is None:
            return True, None
        dt = max(t - self.last_t, 60.0)
        rate = (value - self.last) / dt
        return abs(rate) <= self.max_rate * 1.0 or dt > 6 * 3600, rate * 3600

    def update(self, t: float, value: float) -> float | None:
        """Feed one raw reading. Returns the accepted stage, or None while the reading is held back.

        A held reading is not an estimate: the caller (the agent) must confirm it with another source.
        """
        if self.last_t is not None and t - self.last_t > 3600:
            self.reset()  # after a gap (night, outage) the old state says nothing about the river now
        self.buf.append((t, value))
        candidate = float(np.median([v for _, v in self.buf]))
        ok, _ = self.check(t, candidate)
        if not ok:
            self.rejected += 1
            return None
        self.last, self.last_t = candidate, t
        return round(candidate, 3)
