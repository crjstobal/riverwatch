"""Replay a past flood through the real agent, on a simulated clock.

  * Calibration is refitted WITHOUT the replayed day (from data/eval/<cam>/frames.json), so the flood is unseen.
  * Every tool only sees data at or before the simulated `now` (frames, gage, upstream gage).
  * The agent decides when the next cycle runs (15 / 30 / 60 min), exactly as in production.
  * A simulated duty operator approves any pending request 10 minutes after it was made (marked as simulated).

usage: replay.py CAM_ID START_UTC END_UTC [name]
   e.g. replay.py TX_Cowhouse_Ck_at_W_Range_Rd_nr_Ft_Hood 2026-10-02T12:00:00Z 2026-10-03T02:00:00Z flood-2026-10-02
Writes data/replay/<cam>/<name>/{cycles.jsonl, frames/*.jpg, summary.json, gage.json}
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from riverwatch import agent, usgs, vision  # noqa: E402
from riverwatch.bedrock import BedrockModel  # noqa: E402

OPERATOR_DELAY_S = 600


def utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def calib_without(cam, cfg, day_iso):
    frames = json.loads((ROOT / "data/eval" / cam / "frames.json").read_text())
    off = float(cfg.get("utc_offset", -5))
    dkey = lambda r: datetime.fromtimestamp(r["t"] + 3600 * off, timezone.utc).date().isoformat()
    train = [r for r in frames if dkey(r) != day_iso and r["usable"] and r["truth"] is not None]
    floor = float(np.quantile([r["truth"] for r in train], 0.05))
    pairs = [(r["waterline"], r["truth"]) for r in train if r["waterline"] is not None and r["truth"] > floor + 0.7]
    return vision.fit_calibration(pairs, floor)


def local_series(path):
    g = json.loads(Path(path).read_text())
    return sorted((datetime.fromisoformat(t).timestamp(), v) for t, v in g)


def main(cam, start, end, name="replay"):
    cfg = json.loads((ROOT / "cameras.json").read_text())[cam]
    t0, t1 = utc(start), utc(end)
    off = float(cfg.get("utc_offset", -5))
    day_iso = datetime.fromtimestamp(t0 + 3600 * off, timezone.utc).date().isoformat()
    prof = vision.Profile.load(ROOT / "profiles" / cam)
    prof.calib = calib_without(cam, cfg, day_iso)
    print("calibration without", day_iso, prof.calib)

    gage = local_series(ROOT / "data/cams" / cam / "gauge.json")
    up_cache = {}
    for up in cfg.get("upstream", []):
        # cache keyed by the window, so a short test run never leaves a truncated series behind
        p = ROOT / "data" / f"gage_{up['site']}_{int(t0)}_{int(t1)}.json"
        if not p.exists():
            s = usgs.gage_series(up["site"], t1 + 3600, (t1 - t0) / 3600 + 30)
            p.write_text(json.dumps([[usgs.iso(t), v] for t, v in s]))
        up_cache[up["site"]] = local_series(p)

    def gage_override(ctx, hours):
        return [(t, v) for t, v in gage if ctx.at - hours * 3600 <= t <= ctx.at]

    def upstream_override(ctx, up, hours):
        return [(t, v) for t, v in up_cache[up["site"]] if ctx.at - hours * 3600 <= t <= ctx.at]

    out = ROOT / "data/replay" / cam / name
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "gage.json").write_text(json.dumps([[t, v] for t, v in gage if t0 - 3 * 3600 <= t <= t1]))
    (out / "upstream.json").write_text(json.dumps({k: [[t, v] for t, v in s if t0 - 3 * 3600 <= t <= t1] for k, s in up_cache.items()}))
    model = BedrockModel()
    state = agent.new_state()
    now, n = t0, 0
    log = open(out / "cycles.jsonl", "w")
    while now <= t1:
        # simulated duty operator
        p = state.get("pending")
        if p and now - p["requested_t"] >= OPERATOR_DELAY_S:
            state["approvals"].append({**p, "approved": usgs.iso(now), "by": "simulated operator"})
            state["alert"], state["pending"] = p["level"], None
        ctx = agent.Context(cam, cfg, prof, now, state, local_frames=ROOT / "data/cams" / cam / "img",
                            gage_override=gage_override, upstream_override=upstream_override)
        rec = agent.run_cycle(ctx, model)
        truth = [v for t, v in gage if t <= now]
        rec["truth_stage_ft"] = truth[-1] if truth else None
        rec["truth_level"] = agent.level_for(rec["truth_stage_ft"], cfg)
        rec["approvals"] = list(state["approvals"])
        if ctx.overlay is not None:
            fn = f"{n:03d}.jpg"
            cv2.imwrite(str(out / "frames" / fn), ctx.overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
            rec["frame"] = fn
        log.write(json.dumps(rec, default=str) + "\n"); log.flush()
        print(f"{rec['at']} truth={rec['truth_stage_ft']} alert={state['alert']} pending={bool(state['pending'])} "
              f"next={state['next_check_minutes']} calls={rec['model_calls']} | {rec['summary'][:110]}")
        now += 60 * int(state["next_check_minutes"])
        n += 1
    log.close()
    (out / "summary.json").write_text(json.dumps({"cam": cam, "start": start, "end": end, "cycles": n,
                                                  "calibration": prof.calib, "approvals": state["approvals"],
                                                  "issues": state["issues"]}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
