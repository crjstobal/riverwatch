"""Evaluate camera-derived stage against the official USGS gage, with blocked cross-validation by day.

1. Every frame is measured once (quality gate, ECC registration, flow mask, waterline).
2. For each calendar day, the waterline -> stage calibration is fitted on the OTHER days only
   (daytime usable frames, robust isotonic fit) and applied to that day's frames. No leakage.
3. Two outputs are scored:
     vision only     : the per-frame estimate
     vision + physics: a plausibility filter (median of the last 3 usable readings, rise-rate cap)
                       of the kind the agent applies before acting
4. Metrics: MAE / RMSE / bias above the resolvable floor, share within 0.5 and 1 ft, flood-threshold
   detection (precision / recall) and what the quality gate rejected and why.

usage: evaluate.py CAM_ID
"""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from riverwatch import vision, physics  # noqa: E402


def ts(p):
    return datetime.strptime(p.stem.split("___")[1], "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=timezone.utc).timestamp()


def load_gauge(d):
    g = json.loads((d / "gauge.json").read_text())
    gt = np.array([datetime.fromisoformat(t).timestamp() for t, _ in g])
    gv = np.array([v for _, v in g])
    o = np.argsort(gt)
    return gt[o], gv[o]


def measure_all(cam, cfg, imgs, it, stage, day):
    """One perception pass over every frame; cached in data/eval/<cam>/frames.json."""
    out = ROOT / "data/eval" / cam
    cache = out / "frames.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out.mkdir(parents=True, exist_ok=True)
    base = np.nanquantile(stage, 0.05)
    dry_idx = np.where(day & (stage < base + 0.1))[0]
    prof = vision.build_profile(cam, cv2.imread(str(imgs[dry_idx[len(dry_idx) // 2]])), cfg["channel"])
    prof.save(ROOT / "profiles" / cam)
    frames, prev, prev_t = [], None, None
    for k, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None:
            continue
        use_prev = prev if prev_t is not None and it[k] - prev_t <= 1200 else None
        r, _ = vision.read_stage(img, use_prev, prof)
        frames.append({"t": float(it[k]), "file": p.name, "day": bool(day[k]),
                       "truth": None if np.isnan(stage[k]) else round(float(stage[k]), 3),
                       "usable": r.usable, "reasons": r.reasons, "waterline": r.waterline,
                       "columns": r.columns_with_water, "ecc": r.ecc, "brightness": r.quality["brightness"]})
        prev, prev_t = img, it[k]
        if k % 500 == 0:
            print(f"  measured {k}/{len(imgs)}", file=sys.stderr)
    cache.write_text(json.dumps(frames))
    return frames


def score(rows, key, floor, flood):
    ok = [r for r in rows if r[key] is not None and r["truth"] is not None]
    above = [r for r in ok if r["truth"] > floor + 0.5]
    err = np.array([r[key] - r["truth"] for r in above])
    tp = int(sum(r[key] >= flood and r["truth"] >= flood for r in ok))
    fp = int(sum(r[key] >= flood and r["truth"] < flood for r in ok))
    fn = int(sum(r[key] < flood and r["truth"] >= flood for r in ok))
    f = lambda v: None if v is None else round(float(v), 3)
    return {
        "frames_scored": len(ok), "frames_above_floor": len(above),
        "mae_ft": f(np.abs(err).mean()) if len(err) else None,
        "rmse_ft": f(np.sqrt((err ** 2).mean())) if len(err) else None,
        "bias_ft": f(err.mean()) if len(err) else None,
        "within_0_5ft": f((np.abs(err) <= 0.5).mean()) if len(err) else None,
        "within_1ft": f((np.abs(err) <= 1.0).mean()) if len(err) else None,
        "flood": {"threshold_ft": flood, "tp": tp, "fp": fp, "fn": fn,
                  "precision": f(tp / (tp + fp)) if tp + fp else None,
                  "recall": f(tp / (tp + fn)) if tp + fn else None},
    }


def main(cam):
    cfg = json.loads((ROOT / "cameras.json").read_text())[cam]
    d = ROOT / "data/cams" / cam
    gt, gv = load_gauge(d)
    imgs = sorted((d / "img").glob("*.jpg"))
    it = np.array([ts(p) for p in imgs])
    j = np.clip(np.searchsorted(gt, it), 1, len(gt) - 1)
    near = np.minimum(np.abs(gt[j] - it), np.abs(gt[j - 1] - it))
    stage = np.where(near < 600, np.interp(it, gt, gv), np.nan)
    local_h = ((it / 3600) + float(cfg.get("utc_offset", -5))) % 24
    day = (local_h >= 9) & (local_h <= 17)

    frames = measure_all(cam, cfg, imgs, it, stage, day)
    dkey = lambda r: datetime.fromtimestamp(r["t"] + 3600 * float(cfg.get("utc_offset", -5)), timezone.utc).date().isoformat()
    days = sorted({dkey(r) for r in frames})

    # blocked cross-validation by day
    for r in frames:
        r["est"] = None
    tables = {}
    for dd in days:
        train = [r for r in frames if dkey(r) != dd and r["usable"] and r["truth"] is not None]
        if not train:
            continue
        floor = float(np.quantile([r["truth"] for r in train], 0.05))
        # only frames where the camera saw flowing water and the gage is clearly above base flow
        pairs = [(r["waterline"], r["truth"]) for r in train if r["waterline"] is not None and r["truth"] > floor + 0.7]
        if len(pairs) < 10:
            continue
        calib = vision.fit_calibration(pairs, floor)
        tables[dd] = calib
        for r in frames:
            if dkey(r) == dd and r["usable"]:
                if r["waterline"] is None:
                    r["est"] = calib["floor_ft"]
                elif vision.above_range(r["waterline"], calib):
                    r["est"], r["above_range"] = None, True  # held for confirmation, not scored as a reading
                else:
                    r["est"] = round(vision.stage_from_height(r["waterline"], calib), 3)

    # physics plausibility filter, causal (uses only past readings)
    phys = physics.Plausibility(max_rise_ft_per_h=float(cfg.get("max_rise_ft_per_h", 8.0)))
    for r in frames:
        r["est_filtered"] = phys.update(r["t"], r["est"]) if r["est"] is not None else None

    floor = float(np.median([t["floor_ft"] for t in tables.values()]))
    flood = float(cfg["flood_ft"])
    reasons = Counter(x for r in frames for x in r["reasons"])
    summary = {
        "camera": cam, "label": cfg["label"], "period": [days[0], days[-1]],
        "frames": len(frames), "usable": int(sum(r["usable"] for r in frames)),
        "usable_daytime": int(sum(r["usable"] and r["day"] for r in frames)),
        "rejected_by_reason": dict(reasons.most_common()),
        "held_above_calibrated_range": int(sum(bool(r.get("above_range")) for r in frames)),
        "resolvable_floor_ft": round(floor, 2),
        "protocol": "blocked cross-validation by calendar day; calibration fitted on the other days only",
        "vision_only": score(frames, "est", floor, flood),
        "vision_plus_physics": score(frames, "est_filtered", floor, flood),
        "example_calibration": tables[days[len(days) // 2]] if tables else None,
    }
    out = ROOT / "data/eval" / cam
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "frames_scored.json").write_text(json.dumps(frames))
    print(json.dumps(summary, indent=1))
    plot(cam, cfg, frames, gt, gv, flood, floor, out)


def plot(cam, cfg, frames, gt, gv, flood, floor, out):
    T = lambda xs: [datetime.fromtimestamp(x, timezone.utc) for x in xs]
    t0, t1 = frames[0]["t"], frames[-1]["t"]
    m = (gt >= t0) & (gt <= t1)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(T(gt[m]), gv[m], color="#1f4e79", lw=1.3, label="USGS gage (official)", zorder=3)
    us = [r for r in frames if r["est"] is not None and r["est_filtered"] is not None]
    held = [r for r in frames if r["est"] is not None and r["est_filtered"] is None]
    ax.scatter(T([r["t"] for r in us]), [r["est"] for r in us], s=7, color="#e07a1f", alpha=0.75,
               label="Camera estimate (OpenCV 5), physically plausible", zorder=4)
    ax.scatter(T([r["t"] for r in held]), [min(r["est"], 13.5) for r in held], s=7, facecolors="none",
               edgecolors="#c9a27a", alpha=0.6, label="Camera estimate held by the physics check (implausible jump)")
    rej = [r for r in frames if not r["usable"]]
    ax.scatter(T([r["t"] for r in rej]), [floor - 0.6] * len(rej), s=6, color="#9a9a9a", marker="|",
               label="Frame rejected by quality gate")
    ax.axhline(flood, color="#b00020", ls="--", lw=0.8, label=f"Alert threshold {flood:g} ft")
    ax.set_ylabel("Gage height (ft)"); ax.set_ylim(floor - 0.9, 14)
    ax.set_title(cfg["label"] + " (held-out day by day)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "timeseries.png", dpi=130)


if __name__ == "__main__":
    main(sys.argv[1])
