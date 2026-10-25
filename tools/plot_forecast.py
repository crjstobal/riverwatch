"""Figures for the forecast evaluation: lead time per closure event, and the 2026-10-02 flood in detail.

usage: plot_forecast.py      writes data/eval/forecast_leads.png and data/eval/forecast_oct2.png
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import train_forecast as tf  # noqa: E402
from riverwatch import forecast  # noqa: E402


def leads_for(alarm, t, d, ev):
    out = []
    for i in ev:
        a = max(0, i - 24); on = np.where(alarm[a:i])[0]
        if not len(on):
            out.append(0.0); continue
        idx = a + on; st = idx[0]
        for k in range(1, len(idx)):
            if idx[k] - idx[k - 1] > 4:
                st = idx[k]
        out.append((t[i] - t[st]) / 60)
    return np.array(out)


def main():
    Z = np.load(ROOT / "data/eval/forecast_cv.npz")
    t, d, u, pg = Z["t"], Z["d"], Z["u"], np.nan_to_num(Z["p_gb"])
    up120 = np.r_[np.full(8, np.nan), u[8:] - u[:-8]]
    ev = tf.events(t, d)
    shipped = (pg >= 0.5) | (np.nan_to_num(up120) >= forecast.UPSTREAM_RULE_FT)
    local = np.r_[np.nan, d[1:] - d[:-1]]
    trend = np.isfinite(local) & (d + 4 * np.maximum(np.nan_to_num(local), 0) >= tf.CLOSE)
    L1, L2 = leads_for(shipped, t, d, ev), leads_for(trend, t, d, ev)
    labels = [datetime.fromtimestamp(t[i], timezone.utc).strftime("%Y-%m-%d") for i in ev]
    peaks = [float(np.nanmax(d[i:i + 96])) for i in ev]

    fig, ax = plt.subplots(figsize=(13, 4.6))
    x = np.arange(len(ev))
    ax.bar(x - 0.2, L2, 0.4, color="#c9d3db", label="local trend extrapolation")
    ax.bar(x + 0.2, L1, 0.4, color="#2f86c9", label="RiverWatch forecast (model + upstream rule)")
    ax.set_xticks(x, [f"{l}\n{p:.0f} ft" for l, p in zip(labels, peaks)], rotation=90, fontsize=7)
    ax.set_ylabel("warning lead time (min)")
    ax.set_ylim(0, 380)
    ax.set_title("Warning before each of the 46 crossing closures since 2017 (leave-one-year-out; gage threshold alone = 0 min)")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(ROOT / "data/eval/forecast_leads.png", dpi=130); plt.close(fig)

    # Oct 2 detail (CDT)
    t0 = datetime(2026, 10, 2, 11, 0, tzinfo=timezone.utc).timestamp(); t1 = t0 + 8 * 3600
    m = (t >= t0) & (t <= t1)
    T = [datetime.fromtimestamp(x, timezone.utc) - timedelta(hours=5) for x in t[m]]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(T, d[m], color="#1f4e79", lw=2, label="crossing gage (stage)")
    ax.plot(T, u[m], color="#7aa6c8", lw=1.5, label="upstream gage, 12 km")
    ax.axhline(5.0, color="#c2511a", ls="--", lw=1, label="close crossing (1 ft over the deck)")
    ax2 = ax.twinx()
    ax2.fill_between(T, 0, pg[m], color="#e8a33d", alpha=0.25, step="mid", label="model probability")
    warn = shipped[m]
    first = np.argmax(warn) if warn.any() else None
    cross = np.argmax(d[m] >= 5.0)
    if first is not None:
        ax.axvline(T[first], color="#e07a1f", lw=2)
        ax.annotate(f"early warning {T[first]:%H:%M}", (T[first], 9), xytext=(-150, 0), textcoords="offset points",
                    color="#e07a1f", fontsize=10, arrowprops=dict(arrowstyle="->", color="#e07a1f"))
    ax.axvline(T[cross], color="#b00020", lw=1.5)
    ax.annotate(f"crossing closes {T[cross]:%H:%M}", (T[cross], 7), xytext=(15, 0), textcoords="offset points",
                color="#b00020", fontsize=10)
    ax.set_ylabel("stage (ft)"); ax2.set_ylabel("P(close within 60 min)"); ax2.set_ylim(0, 1)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)  # stage lines in front of the shading
    ax.set_xlabel("2026-10-02, local time (CDT)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "data/eval/forecast_oct2.png", dpi=130); plt.close(fig)
    print("ok", int((L1 > 0).sum()), "warned")


if __name__ == "__main__":
    main()
