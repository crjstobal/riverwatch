"""RiverWatch agent: perception -> decision -> action loop around OpenCV 5 measurements.

The loop is vendor-neutral: `Model` is any object with `converse(system, messages, tools) -> dict` returning
a Bedrock-Converse-shaped message (see bedrock.py). Tools run here, in-process, against a `Context` that
carries a simulated clock, so the same code serves live ticks and historical replays.

What the visual evidence changes:
  * measure_camera says the frame is unusable -> the agent must fall back to the gage / upstream site
  * camera and gage disagree -> the agent weighs image quality and may flag a gage fault
  * camera shows a fast rise -> the agent tightens the sampling cadence and asks a human to close the crossing
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from pathlib import Path as _Path

from . import forecast, physics, usgs, vision

_FORECAST = _Path(__file__).resolve().parent.parent / "models" / "forecast.json"
FORECAST_MODEL = forecast.Model.load(_FORECAST) if _FORECAST.exists() else None
FORECAST_SKILL = "Held-out skill over 9.5 years (leave-one-year-out): warned before 32 of 46 closures, 28 of them at least 30 min ahead, median lead 38 min; about 3 warnings a year with the water staying below caution. It cannot see rain that falls between the gages and the crossing until the crossing itself starts rising."

MAX_MODEL_CALLS = 8
CADENCES = {15, 30, 60}

SYSTEM = """You are RiverWatch, a flood-watch agent for one river camera that overlooks a low-water road crossing.
Every few minutes you get control for one short decision cycle. Use the tools, then call finish.

Facts you can rely on:
- measure_camera runs OpenCV 5 on the latest camera frames and returns the river stage it sees, with image quality.
  If usable is false, the camera gives you NO information about the river: say why and use other sources.
  If above_range is true, the waterline is higher than anything seen during calibration: treat it as unconfirmed.
  If plausible is false, the implied rise rate is physically unlikely: re-check before acting on it.
- view_camera_frame shows you the annotated frame. Known camera failure: sunlit dry gravel can be marked as water
  when sun and shadows move; real flood water is a continuous brown or grey sheet across the channel.
- get_gage returns the official USGS gage (primary sensor) at the same site. Gages fail: the primary can be silent, and
  other sensors listed as not used for alerts may be stuck. A silent primary while the camera works is a gage issue.
  A gage whose value is changing (trend not 0, possibly_stuck false) is NOT stuck: if it disagrees with a camera that
  sees no water, suspect the camera first (dusk light, missed detection) and look at the frame.
- request_human_approval enforces guardrails in code (two confirmations to escalate, no lowering an alert while a
  recent reading still implies a higher level). If a request is blocked, follow the note.
- get_upstream returns the gage upstream; water there reaches this crossing later (about 80 min, median).
- forecast_crossing predicts whether the crossing closes within 60 minutes. If its warning is true while the public
  alert is clear, request caution as an early warning (say it is a forecast and why) and check every 15 min. A
  forecast never justifies close_crossing or danger by itself: those need real readings.
- Alert levels are defined by water depth over the crossing deck (see the context message).

Rules:
1. Never invent a stage. Quote the source (camera, gage, upstream) of every number you use.
2. If camera and gage disagree by more than 1 ft, look at the frame (view_camera_frame), decide which to trust using
   what you see, image quality and trends, and say so. A gage that is silent or stuck while the camera shows rising
   water is a sensor issue to flag.
3. Each reading comes with alert_level_implied, already computed from the thresholds: use it, do not recompute.
   When you request approval, request the level implied by the reading you trust most right now.
4. You may set the watch level (normal / elevated) yourself. Closing the crossing or a danger alert needs a human:
   call request_human_approval with the level, a one-paragraph reason and the evidence numbers.
5. Do not ask for approval again for the same level while one is pending or already approved; update the operator only
   if the situation changed materially.
6. Call forecast_crossing every cycle. Choose the next check interval: 15 min if the river is rising, above caution,
   the forecast warns, or sources disagree; 30 min while
   receding above caution; 60 min when low and stable.
7. Be brief. Finish with a two-sentence summary for the public status page."""

TOOLS = [
    {"toolSpec": {"name": "measure_camera", "description": "Run OpenCV 5 on the latest pair of camera frames (or an "
        "older pair with frames_back) and return stage, waterline, image quality and physical plausibility.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "frames_back": {"type": "integer", "description": "0 = newest pair, 1 = the pair before, up to 3", "minimum": 0, "maximum": 3}},
            "required": []}}}},
    {"toolSpec": {"name": "view_camera_frame", "description": "Look at the latest camera frame yourself, annotated by "
        "OpenCV (blue = what it measured as flowing water, yellow line = measured waterline). Use it when the camera "
        "reading is surprising, above range, implausible, or disagrees with the gage.",
        "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}}}},
    {"toolSpec": {"name": "get_gage", "description": "Official USGS gage height at this site for the last hours, with trend.",
        "inputSchema": {"json": {"type": "object", "properties": {"hours": {"type": "number", "minimum": 1, "maximum": 24}}, "required": []}}}},
    {"toolSpec": {"name": "get_upstream", "description": "Gage height and trend at the upstream station(s), with distance.",
        "inputSchema": {"json": {"type": "object", "properties": {"hours": {"type": "number", "minimum": 1, "maximum": 24}}, "required": []}}}},
    {"toolSpec": {"name": "forecast_crossing", "description": "Probability that the crossing closes (stage >= the "
        "close level) within the next 60 minutes, from the last 24 h at this gage and upstream (model trained on 9.5 "
        "years of history), with whether the early warning fires and why.",
        "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}}}},
    {"toolSpec": {"name": "set_watch_level", "description": "Set the internal watch level and the next check interval.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "level": {"type": "string", "enum": ["normal", "elevated"]},
            "next_check_minutes": {"type": "integer", "enum": [15, 30, 60]},
            "reason": {"type": "string"}}, "required": ["level", "next_check_minutes", "reason"]}}}},
    {"toolSpec": {"name": "request_human_approval", "description": "Ask the duty operator to approve a public alert. "
        "Required for close_crossing and danger, and to stand an alert down to clear.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "level": {"type": "string", "enum": ["caution", "close_crossing", "danger", "clear"]},
            "reason": {"type": "string"},
            "evidence": {"type": "string", "description": "numbers and their sources"}},
            "required": ["level", "reason", "evidence"]}}}},
    {"toolSpec": {"name": "flag_sensor_issue", "description": "Record a suspected fault in the camera or the gage for maintenance.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "sensor": {"type": "string", "enum": ["camera", "gage"]}, "detail": {"type": "string"}},
            "required": ["sensor", "detail"]}}}},
    {"toolSpec": {"name": "finish", "description": "End this cycle with a short public summary.",
        "inputSchema": {"json": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}}},
]


@dataclass
class Context:
    cam_id: str
    cfg: dict
    prof: vision.Profile
    at: float                                   # simulated or real clock, UTC seconds
    state: dict                                 # persisted between cycles (see new_state)
    local_frames: Path | None = None            # evaluation copy of the frames, else USGS S3
    gage_override: Callable | None = None       # scenario tests can replace data sources
    upstream_override: Callable | None = None
    camera_override: Callable | None = None
    trace: list = field(default_factory=list)
    overlay: np.ndarray | None = None           # last annotated frame, for the status page


def new_state() -> dict:
    return {"watch": "normal", "alert": "clear", "pending": None, "next_check_minutes": 60,
            "readings": [], "issues": [], "approvals": [], "summary": ""}


def level_for(stage: float | None, cfg: dict) -> str:
    lvl = "clear"
    for a in cfg["alerts"]:
        if stage is not None and stage >= a["stage_ft"]:
            lvl = a["level"]
    return lvl


# ---------------------------------------------------------------- tools

def _depth(stage, cfg):
    return None if stage is None else round(stage - cfg["crossing_deck_ft"], 2)


def tool_measure_camera(ctx: Context, frames_back: int = 0) -> dict:
    if ctx.camera_override:
        return ctx.camera_override(ctx, frames_back)
    names = usgs.frames_before(ctx.cam_id, ctx.at, n=2 + frames_back)
    if len(names) < 2 + frames_back:
        return {"usable": False, "reasons": ["camera has not delivered recent frames"], "frames_found": len(names)}
    cur_n, prev_n = names[frames_back], names[frames_back + 1]
    cur = usgs.load_frame(ctx.cam_id, cur_n, ctx.local_frames)
    prev = usgs.load_frame(ctx.cam_id, prev_n, ctx.local_frames)
    age_min = (ctx.at - usgs.frame_time(cur_n)) / 60
    if cur is None or prev is None:
        return {"usable": False, "reasons": ["frame could not be decoded"]}
    if usgs.frame_time(cur_n) - usgs.frame_time(prev_n) > 1200:
        return {"usable": False, "reasons": ["frames too far apart to measure flow"], "frame_age_min": round(age_min)}
    r, mask = vision.read_stage(cur, prev, ctx.prof)
    out = {"frame_time": usgs.iso(usgs.frame_time(cur_n)), "frame_age_min": round(age_min), "usable": r.usable,
           "reasons": r.reasons, "quality": {k: r.quality.get(k) for k in ("brightness", "saturation", "sharpness", "contrast")},
           "registration_ecc": r.ecc}
    if r.usable:
        stage = r.floor_ft if r.below_range else r.stage_ft
        out.update({"stage_ft": stage, "waterline": r.waterline, "columns_with_water": r.columns_with_water,
                    "below_range": r.below_range, "above_range": r.above_range,
                    "depth_over_crossing_ft": _depth(stage, ctx.cfg),
                    "alert_level_implied": None if r.above_range else level_for(stage, ctx.cfg),
                    "note": "no flowing water detected across the channel. This means stage at or below the camera floor "
                            "only if other sources agree; at dusk or with smooth water the detector can miss a flood" if r.below_range else
                            "waterline above calibrated range: unconfirmed" if r.above_range else ""})
        prev_ok = [x for x in ctx.state["readings"] if x["source"] == "camera" and ctx.at - x["t"] <= 3 * 3600]
        if prev_ok and stage is not None:
            p = physics.Plausibility(float(ctx.cfg.get("max_rise_ft_per_h", 8.0)))
            p.last, p.last_t = prev_ok[-1]["stage"], prev_ok[-1]["t"]
            ok, rate = p.check(usgs.frame_time(cur_n), stage)
            out["plausible"] = bool(ok and not r.above_range)
            out["implied_rate_ft_per_h"] = None if rate is None else round(rate, 2)
        else:
            out["plausible"] = not r.above_range
        if frames_back == 0 and out["plausible"] and stage is not None:
            ctx.state["readings"].append({"t": usgs.frame_time(cur_n), "stage": stage, "source": "camera"})
        if frames_back == 0:
            ctx.overlay = annotate(cur, mask, ctx.prof, out)
    elif frames_back == 0:
        ctx.overlay = annotate(cur, None, ctx.prof, out)
    return out


def _series_summary(series, cfg, site_label):
    if not series:
        return {"site": site_label, "available": False, "note": "no data in the window: gage silent or offline"}
    last_t, last_v = series[-1]
    tr = usgs.trend(series, 60)
    vals = [v for _, v in series]
    stuck = len(series) >= 6 and np.ptp(vals[-6:]) == 0 and (tr or 0) == 0
    return {"site": site_label, "available": True, "latest_ft": last_v, "latest_time": usgs.iso(last_t),
            "trend_ft_per_h": None if tr is None else round(tr, 2), "max_ft": max(vals), "min_ft": min(vals),
            "possibly_stuck": bool(stuck)}


def tool_view_camera_frame(ctx: Context) -> dict:
    if ctx.overlay is None:
        tool_measure_camera(ctx, 0)
    if ctx.overlay is None:
        return {"error": "no frame available"}
    return {"_image": ctx.overlay, "note": "annotated frame attached"}


def tool_get_gage(ctx: Context, hours: float = 3) -> dict:
    if ctx.gage_override:
        series, others = ctx.gage_override(ctx, hours), {}
    else:
        sensors = usgs.gage_sensors(ctx.cfg["site"], ctx.at, max(hours, 24))
        prim = [k for k in sensors if "primary" in k.lower()] or (list(sensors)[:1] if len(sensors) == 1 else [])
        series = [(t, v) for t, v in (sensors.get(prim[0], []) if prim else []) if t >= ctx.at - hours * 3600]
        others = {k: v for k, v in sensors.items() if not prim or k != prim[0]}
    out = _series_summary(series, ctx.cfg, ctx.cfg["site"])
    if not out.get("available"):
        out["note"] = ("primary sensor silent in the window: no official reading to use for alerts. "
                       "Report it with flag_sensor_issue(sensor='gage') unless already flagged in the last hours.")
    if others:
        out["other_sensors_not_used_for_alerts"] = {
            k: {"latest_ft": v[-1][1], "range_24h_ft": round(max(x for _, x in v) - min(x for _, x in v), 3)}
            for k, v in others.items() if v}
    if out.get("available"):
        out["depth_over_crossing_ft"] = _depth(out["latest_ft"], ctx.cfg)
        out["alert_level_implied"] = level_for(out["latest_ft"], ctx.cfg)
        ctx.state["readings"].append({"t": series[-1][0], "stage": out["latest_ft"], "source": "gage"})
    return out


def tool_get_upstream(ctx: Context, hours: float = 3) -> dict:
    res = []
    for up in ctx.cfg.get("upstream", []):
        series = ctx.upstream_override(ctx, up, hours) if ctx.upstream_override else usgs.gage_series(up["site"], ctx.at, hours)
        s = _series_summary(series, ctx.cfg, f'{up["site"]} {up["name"]}')
        s["distance_km"] = up["km"]
        res.append(s)
    return {"upstream": res}


def tool_forecast_crossing(ctx: Context) -> dict:
    if FORECAST_MODEL is None:
        return {"available": False, "note": "forecast model not installed"}
    if ctx.gage_override:
        down = ctx.gage_override(ctx, 24)
    else:
        down = usgs.gage_series(ctx.cfg["site"], ctx.at, 24)
    if len(down) < 5:  # gage silent: fall back to the camera readings in memory
        down = [(x["t"], x["stage"]) for x in ctx.state["readings"] if x["source"] == "camera" and ctx.at - x["t"] <= 86400]
    ups = ctx.cfg.get("upstream", [])
    up = []
    if ups:
        up = ctx.upstream_override(ctx, ups[0], 24) if ctx.upstream_override else usgs.gage_series(ups[0]["site"], ctx.at, 24)
    fc = forecast.from_series(FORECAST_MODEL, down, up, ctx.cfg["crossing_deck_ft"], ctx.at)
    if fc is None:
        return {"available": False, "note": "not enough recent data at this gage or upstream for a forecast"}
    fc.update({"available": True, "horizon_min": 60, "skill": FORECAST_SKILL})
    if fc["warning"] and ctx.state["alert"] == "clear" and not ctx.state.get("pending"):
        fc["recommended_action"] = ("call request_human_approval with level='caution' now, as an EARLY WARNING based on "
                                    "this forecast (quote the reason), and set next_check_minutes 15. Do not request "
                                    "close_crossing or danger on a forecast.")
    ctx.state["forecast"] = {"t": ctx.at, "p": fc["p_close_within_60min"], "warning": fc["warning"], "why": fc["why"],
                             "mode": fc["mode"]}
    return fc


def tool_set_watch_level(ctx: Context, level: str, next_check_minutes: int, reason: str) -> dict:
    if next_check_minutes not in CADENCES:
        next_check_minutes = 60
    # guardrail: no relaxing while an alert is public or any recent reading implies caution or worse
    top, src = _highest_trusted_level(ctx)
    fc = ctx.state.get("forecast") or {}
    fc_warn = bool(fc.get("warning")) and ctx.at - fc.get("t", 0) <= 3600
    if (level == "normal" or next_check_minutes == 60) and (ctx.state["alert"] != "clear" or top != "clear" or fc_warn):
        why = (f"public alert is {ctx.state['alert']}" if ctx.state["alert"] != "clear" else
               f"{src} implies {top}" if top != "clear" else f"the forecast warns ({fc.get('why')})")
        return {"ok": False, "blocked_by_guardrail": True,
                "note": f"cannot relax to normal / 60 min while the {why}. Use elevated with 15 or 30 min."}
    ctx.state["watch"] = level
    ctx.state["next_check_minutes"] = int(next_check_minutes)
    return {"ok": True, "watch": level, "next_check_minutes": next_check_minutes}


LEVEL_ORDER = ["clear", "caution", "close_crossing", "danger"]


def _highest_trusted_level(ctx: Context) -> tuple[str, str]:
    """Highest alert level implied by any reading from the last 30 minutes (camera readings are only stored
    when usable and plausible)."""
    best, src = "clear", ""
    for x in ctx.state["readings"]:
        if ctx.at - x["t"] <= 1800:
            lv = level_for(x["stage"], ctx.cfg)
            if LEVEL_ORDER.index(lv) > LEVEL_ORDER.index(best):
                best, src = lv, f"{x['source']} {x['stage']:.2f} ft at {usgs.iso(x['t'])[11:16]}"
    return best, src


def _supported_level(ctx: Context) -> str:
    """Highest level backed by at least two distinct readings from the last 30 minutes."""
    recent = {(x["source"], round(x["t"])): x["stage"] for x in ctx.state["readings"] if ctx.at - x["t"] <= 1800}
    best = "clear"
    for lv in LEVEL_ORDER[1:]:
        if sum(LEVEL_ORDER.index(level_for(v, ctx.cfg)) >= LEVEL_ORDER.index(lv) for v in recent.values()) >= 2:
            best = lv
    return best


def tool_request_human_approval(ctx: Context, level: str, reason: str, evidence: str) -> dict:
    # guardrail in code, not only in the prompt: never ask to lower the public alert while any recent
    # trusted reading still implies a higher level
    if level in LEVEL_ORDER and LEVEL_ORDER.index(level) < LEVEL_ORDER.index(ctx.state["alert"]):
        top, src = _highest_trusted_level(ctx)
        if LEVEL_ORDER.index(top) > LEVEL_ORDER.index(level):
            return {"ok": False, "blocked_by_guardrail": True,
                    "note": f"cannot request a lower alert ({level}) while a recent reading implies {top} ({src})"}
    # guardrail: closing the crossing or a danger alert needs two independent confirmations in the last 30 min
    # (two camera frames, or camera and gage); one surprising frame is never enough to wake an operator
    if level in ("close_crossing", "danger") and LEVEL_ORDER.index(level) > LEVEL_ORDER.index(ctx.state["alert"]):
        support = [x for x in ctx.state["readings"]
                   if ctx.at - x["t"] <= 1800 and LEVEL_ORDER.index(level_for(x["stage"], ctx.cfg)) >= LEVEL_ORDER.index(level)]
        if len({(x["source"], round(x["t"])) for x in support}) < 2:
            supported = _supported_level(ctx)
            hint = (f" Two readings already support '{supported}': request that now and escalate when confirmed."
                    if LEVEL_ORDER.index(supported) > LEVEL_ORDER.index(ctx.state["alert"]) else "")
            return {"ok": False, "blocked_by_guardrail": True,
                    "note": f"{level} needs two independent readings at or above it within 30 min (have "
                            f"{len(support)}).{hint} Re-check in 15 min."}
    pend = ctx.state.get("pending")
    if pend and pend["level"] == level:
        return {"ok": False, "note": f"an approval request for {level} is already pending since {pend['requested']}"}
    if ctx.state["alert"] == level:
        return {"ok": False, "note": f"{level} is already the approved public alert"}
    ctx.state["pending"] = {"level": level, "reason": reason, "evidence": evidence, "requested": usgs.iso(ctx.at),
                            "requested_t": ctx.at}
    return {"ok": True, "note": "operator notified; the public alert changes only after approval"}


def tool_flag_sensor_issue(ctx: Context, sensor: str, detail: str) -> dict:
    # a "stuck" gage is a checkable claim: refuse it when the gage values are visibly changing
    if sensor == "gage" and "stuck" in detail.lower():
        g = [x for x in ctx.state["readings"] if x["source"] == "gage" and ctx.at - x["t"] <= 3600]
        span = max(x["stage"] for x in g) - min(x["stage"] for x in g) if len(g) >= 2 else 0.0
        series = []
        if len(g) < 2:
            series = (ctx.gage_override(ctx, 1) if ctx.gage_override else usgs.gage_series(ctx.cfg["site"], ctx.at, 1))
            span = (max(v for _, v in series) - min(v for _, v in series)) if len(series) >= 2 else 0.0
        if span > 0.05:
            return {"ok": False, "blocked_by_guardrail": True,
                    "note": f"the gage changed by {span:.2f} ft in the last hour, so it is not stuck. If the camera "
                            "disagrees, the camera is the more likely fault (look at the frame)."}
    ctx.state["issues"].append({"t": usgs.iso(ctx.at), "sensor": sensor, "detail": detail})
    ctx.state["issues"] = ctx.state["issues"][-20:]
    return {"ok": True}


IMPL = {"forecast_crossing": tool_forecast_crossing, "measure_camera": tool_measure_camera, "view_camera_frame": tool_view_camera_frame, "get_gage": tool_get_gage, "get_upstream": tool_get_upstream,
        "set_watch_level": tool_set_watch_level, "request_human_approval": tool_request_human_approval,
        "flag_sensor_issue": tool_flag_sensor_issue}


# ---------------------------------------------------------------- loop

def context_message(ctx: Context) -> str:
    cfg, st = ctx.cfg, ctx.state
    recent = [x for x in st["readings"] if ctx.at - x["t"] <= 3 * 3600][-8:]
    lines = [
        f"Time now (UTC): {usgs.iso(ctx.at)}. Camera: {cfg['label']} (USGS {cfg['site']}).",
        f"Crossing deck is at stage {cfg['crossing_deck_ft']} ft. Alert levels by stage: " +
        "; ".join(f"{a['level']} >= {a['stage_ft']} ft ({a['meaning']})" for a in cfg["alerts"]) + ".",
        f"Camera can resolve stage from about {ctx.prof.calib.get('floor_ft')} ft; highest stage seen during calibration "
        f"{ctx.prof.calib.get('max_stage_seen_ft')} ft.",
        ("Last forecast: " + (f"p={st['forecast']['p']} warning={st['forecast']['warning']} ({st['forecast']['why']})"
                              if st.get("forecast") else "none") + "."),
        f"Current public alert: {st['alert']}. Internal watch: {st['watch']}. Pending approval: "
        f"{json.dumps(st['pending']) if st['pending'] else 'none'}.",
        "Recent readings (UTC, stage ft, source): " +
        (", ".join(f"{usgs.iso(x['t'])[11:16]} {x['stage']:.2f} {x['source']}" for x in recent) or "none"),
        "Previous summary: " + (st["summary"] or "none"),
    ]
    return "\n".join(lines)


def run_cycle(ctx: Context, model, clock: Callable[[], float] = time.time) -> dict:
    """One perception-decision-action cycle. Returns the trace record."""
    t0 = clock()
    messages = [{"role": "user", "content": [{"text": context_message(ctx)}]}]
    summary, calls, usage = None, 0, {"inputTokens": 0, "outputTokens": 0}
    while calls < MAX_MODEL_CALLS and summary is None:
        resp = model.converse(SYSTEM, messages, TOOLS)
        calls += 1
        for k in usage:
            usage[k] += resp.get("usage", {}).get(k, 0)
        if "error" in resp:
            ctx.trace.append({"step": "model_error", "detail": resp["error"]})
            break
        msg = resp["message"]
        messages.append(msg)
        results, images = [], []
        for block in msg["content"]:
            if "text" in block and block["text"].strip():
                ctx.trace.append({"step": "thought", "text": block["text"].strip()})
            if "toolUse" in block:
                tu = block["toolUse"]
                name, args = tu["name"], tu.get("input") or {}
                if name == "finish":
                    summary = str(args.get("summary", ""))
                    out = {"ok": True}
                elif name in IMPL:
                    try:
                        out = IMPL[name](ctx, **args)
                    except Exception as e:  # tool failures are evidence too
                        out = {"error": f"{type(e).__name__}: {e}"}
                else:
                    out = {"error": f"unknown tool {name}"}
                if isinstance(out, dict) and "_image" in out:
                    # Nova does not take images inside toolResult: attach it right after, in the same user turn
                    ok, jpg = cv2.imencode(".jpg", cv2.resize(out.pop("_image"), (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    images.append(jpg.tobytes())
                ctx.trace.append({"step": "tool", "tool": name, "input": args, "output": out})
                results.append({"toolResult": {"toolUseId": tu["toolUseId"], "content": [{"json": _jsonable(out)}]}})
        if results:
            for b in images:
                results += [{"text": "Annotated camera frame from view_camera_frame:"},
                            {"image": {"format": "jpeg", "source": {"bytes": b}}}]
            messages.append({"role": "user", "content": results})
        elif summary is None:
            messages.append({"role": "user", "content": [{"text": "Use a tool or call finish."}]})
    if summary is None:
        summary = "Cycle ended without a summary (step limit or model error); last known state kept."
        ctx.trace.append({"step": "guard", "detail": "forced finish"})
    ctx.state["summary"] = summary
    ctx.state["readings"] = [x for x in ctx.state["readings"] if ctx.at - x["t"] <= 24 * 3600][-96:]
    return {"at": usgs.iso(ctx.at), "t": ctx.at, "cam_id": ctx.cam_id, "summary": summary, "model_calls": calls,
            "usage": usage, "latency_s": round(clock() - t0, 2), "steps": ctx.trace,
            "state": {k: ctx.state.get(k) for k in ("watch", "alert", "pending", "next_check_minutes", "forecast")}}


def _jsonable(o: Any):
    return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


# ---------------------------------------------------------------- overlay

def annotate(img: np.ndarray, mask: np.ndarray | None, prof: vision.Profile, m: dict) -> np.ndarray:
    out = img.copy()
    cnts, _ = cv2.findContours(prof.channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)
    if mask is not None and mask.any():
        tint = out.copy()
        tint[mask > 0] = (235, 140, 40)
        out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)
        if m.get("waterline") is not None:
            rows = np.where(prof.channel.any(axis=1))[0]
            y = int(rows.max() - m["waterline"] * (rows.max() - rows.min()))
            cv2.line(out, (0, y), (out.shape[1], y), (0, 220, 255), 2, cv2.LINE_AA)
    label = (f"stage {m['stage_ft']:.2f} ft (camera)" if m.get("stage_ft") is not None
             else "camera unusable: " + "; ".join(m.get("reasons", []))[:70])
    cv2.rectangle(out, (0, out.shape[0] - 34), (out.shape[1], out.shape[0]), (20, 20, 20), -1)
    cv2.putText(out, label, (10, out.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out
