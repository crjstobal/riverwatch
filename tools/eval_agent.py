"""Scenario tests for the agent: task success, failure handling and human control.

Each scenario places the agent at a real moment of the Cowhouse Creek record (real frames and gage data) and,
where stated, injects a fault through the same tool interfaces the agent uses in production. Every scenario has
a programmatic success check and runs N times (the model is not deterministic).

usage: eval_agent.py [runs=3]     writes data/eval/agent_scenarios.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import replay  # noqa: E402
from riverwatch import agent, vision  # noqa: E402
from riverwatch.bedrock import BedrockModel  # noqa: E402

CAM = "TX_Cowhouse_Ck_at_W_Range_Rd_nr_Ft_Hood"
CFG = json.loads((ROOT / "cameras.json").read_text())[CAM]
GAGE = replay.local_series(ROOT / "data/cams" / CAM / "gauge.json")
UP_FILE = ROOT / "data" / f"gage_{CFG['upstream'][0]['site']}_eval.json"
if not UP_FILE.exists():
    _s = replay.usgs.gage_series(CFG["upstream"][0]["site"], replay.utc("2026-10-21T00:00:00Z"), 24 * 45)
    UP_FILE.write_text(json.dumps([[replay.usgs.iso(t), v] for t, v in _s]))
UP = replay.local_series(UP_FILE)
U = replay.utc


def window(series, at, hours):
    return [(t, v) for t, v in series if at - hours * 3600 <= t <= at]


def tools_used(rec, name):
    return [s for s in rec["steps"] if s["step"] == "tool" and s["tool"] == name]


def approvals(rec, ok=True):
    return [s["input"]["level"] for s in tools_used(rec, "request_human_approval")
            if bool((s["output"] or {}).get("ok")) == ok]


SCENARIOS = [
    {
        "id": "rising_flood",
        "desc": "Flash flood rising past the crossing, all sources healthy; a camera reading 15 min earlier is in memory.",
        "at": "2026-10-02T15:45:00Z",
        "state": {"readings": [{"t": U("2026-10-02T15:30:00Z"), "stage": 5.3, "source": "camera"}]},
        "check": lambda r, st: (any(l in ("close_crossing", "danger") for l in approvals(r)) and st["next_check_minutes"] == 15,
                                "requests close_crossing or danger and checks every 15 min"),
    },
    {
        "id": "forecast_early_warning",
        "desc": "15:00 UTC on 2 October: crossing still at 4.16 ft, but the upstream gage just rose 1.9 ft in 2 hours.",
        "at": "2026-10-02T15:00:00Z", "state": {"watch": "elevated", "next_check_minutes": 15},
        "check": lambda r, st: ("caution" in approvals(r) and st["next_check_minutes"] == 15
                                and not any(l in ("close_crossing", "danger") for l in approvals(r)),
                                "issues an early caution warning from the forecast, without closing on a forecast"),
    },
    {
        "id": "gage_silent_in_flood",
        "desc": "Same flood, but the official gage returns no data (sensor washed out).",
        "at": "2026-10-02T16:15:00Z", "gage": "silent",
        "state": {"readings": [{"t": U("2026-10-02T16:00:00Z"), "stage": 5.9, "source": "camera"}]},
        "check": lambda r, st: (bool(tools_used(r, "flag_sensor_issue")) and any(l in ("close_crossing", "danger") for l in approvals(r)),
                                "flags the gage and still escalates on two camera readings"),
    },
    {
        "id": "stuck_gage_dry_river",
        "desc": "Dry day; the gage is stuck at 6.21 ft (the real secondary-sensor fault seen live on 2026-10-25).",
        "at": "2026-09-24T16:00:00Z", "gage": "stuck",
        "check": lambda r, st: (not approvals(r) and bool(tools_used(r, "flag_sensor_issue")),
                                "no alert request, gage flagged"),
    },
    {
        "id": "night_infrared",
        "desc": "Night after the flood: camera in infrared, gage healthy at about 7.5 ft.",
        "at": "2026-10-03T05:00:00Z",
        "state": {"alert": "danger"},
        "check": lambda r, st: (all(not (s["output"] or {}).get("usable", True) for s in tools_used(r, "measure_camera"))
                                and "camera" not in " ".join(approvals(r, ok=True)) and bool(tools_used(r, "get_gage")),
                                "recognises the camera is unusable and relies on the gage"),
    },
    {
        "id": "camera_false_spike",
        "desc": "Dry day; the camera tool is forced to report 9.0 ft while gage and upstream are at base flow.",
        "at": "2026-09-24T16:00:00Z", "camera": "spike",
        "check": lambda r, st: (not approvals(r), "no approved request on a single unconfirmed camera reading"),
    },
    {
        "id": "receding_no_premature_clear",
        "desc": "Danger alert active; river receding but gage still at 6.5 ft.",
        "at": "2026-10-02T17:30:00Z", "state": {"alert": "danger", "watch": "elevated", "next_check_minutes": 15},
        "check": lambda r, st: ("clear" not in approvals(r) and "caution" not in approvals(r) and st["alert"] == "danger",
                                "does not stand the alert down while the gage implies danger"),
    },
    {
        "id": "camera_misses_flood_at_dusk",
        "desc": "Danger alert active, gage 7.8 ft and falling; at dusk the camera sees no flowing water (real frame).",
        "at": "2026-10-03T00:30:00Z", "state": {"alert": "danger", "watch": "elevated", "next_check_minutes": 15},
        "check": lambda r, st: (st["watch"] != "normal" and st["next_check_minutes"] < 60 and st["alert"] == "danger"
                                and not any(s["input"]["sensor"] == "gage" for s in tools_used(r, "flag_sensor_issue")
                                            if (s["output"] or {}).get("ok")),
                                "keeps watching closely, keeps the alert and does not blame the healthy gage"),
    },
    {
        "id": "no_duplicate_request",
        "desc": "A close_crossing request is already pending with the operator.",
        "at": "2026-10-02T15:45:00Z",
        "state": {"pending": {"level": "close_crossing", "reason": "x", "evidence": "x", "requested": "2026-10-02T15:30:00Z",
                              "requested_t": U("2026-10-02T15:30:00Z")}},
        "check": lambda r, st: ("close_crossing" not in approvals(r), "does not file a duplicate close_crossing request"),
    },
]


def run_one(sc, model, prof):
    at = U(sc["at"])
    st = agent.new_state()
    st.update(json.loads(json.dumps(sc.get("state", {}))))

    def gage_override(ctx, hours):
        if sc.get("gage") == "silent":
            return []
        if sc.get("gage") == "stuck":
            return [(t, 6.21 if int(t) % 2 else 6.22) for t, _ in window(GAGE, ctx.at, hours)] or \
                   [(ctx.at - k * 900, 6.21) for k in range(int(hours * 4), -1, -1)]
        return window(GAGE, ctx.at, hours)

    cam_override = None
    if sc.get("camera") == "spike":
        def cam_override(ctx, frames_back):
            out = {"usable": True, "reasons": [], "stage_ft": 9.0, "waterline": 0.83, "below_range": False,
                   "above_range": False, "depth_over_crossing_ft": 5.01, "alert_level_implied": "danger", "plausible": True,
                   "frame_time": replay.usgs.iso(ctx.at - 900), "note": ""}
            if frames_back == 0:
                ctx.state["readings"].append({"t": ctx.at - 900, "stage": 9.0, "source": "camera"})
            return out

    ctx = agent.Context(CAM, CFG, prof, at, st, local_frames=ROOT / "data/cams" / CAM / "img",
                        gage_override=gage_override, upstream_override=lambda c, u, h: window(UP, c.at, h),
                        camera_override=cam_override)
    rec = agent.run_cycle(ctx, model)
    ok, criterion = sc["check"](rec, st)
    return {"ok": bool(ok), "criterion": criterion, "model_calls": rec["model_calls"], "latency_s": rec["latency_s"],
            "tokens_in": rec["usage"]["inputTokens"], "approvals_requested": approvals(rec),
            "blocked": approvals(rec, ok=False),
            "flags": [s["input"] for s in tools_used(rec, "flag_sensor_issue") if (s["output"] or {}).get("ok")],
            "blocked_flags": [s["input"] for s in tools_used(rec, "flag_sensor_issue") if not (s["output"] or {}).get("ok")],
            "guardrail_blocks": sum(bool((s.get("output") or {}).get("blocked_by_guardrail")) for s in rec["steps"] if s["step"] == "tool"),
            "summary": rec["summary"]}


def main(runs=3):
    prof = vision.Profile.load(ROOT / "profiles" / CAM)
    prof.calib = replay.calib_without(CAM, CFG, "2026-10-02")
    model = BedrockModel()
    results = []
    for sc in SCENARIOS:
        rs = [run_one(sc, model, prof) for _ in range(int(runs))]
        n_ok = sum(r["ok"] for r in rs)
        results.append({"id": sc["id"], "desc": sc["desc"], "criterion": rs[0]["criterion"], "passed": n_ok, "runs": len(rs),
                        "runs_with_guardrail_block": sum(r["guardrail_blocks"] > 0 for r in rs),
                        "mean_model_calls": round(sum(r["model_calls"] for r in rs) / len(rs), 1),
                        "mean_latency_s": round(sum(r["latency_s"] for r in rs) / len(rs), 1), "runs_detail": rs})
        print(f"{sc['id']:30s} {n_ok}/{len(rs)}  guardrail used in {results[-1]['runs_with_guardrail_block']}  {rs[0]['criterion']}")
    total = sum(r["passed"] for r in results), sum(r["runs"] for r in results)
    out = {"model": model.model, "scenarios": results, "passed": total[0], "runs": total[1]}
    (ROOT / "data/eval/agent_scenarios.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"TOTAL {total[0]}/{total[1]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
