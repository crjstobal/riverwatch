"""Publish the status page, the flood replay and the evaluation to the stack's S3 bucket (served by CloudFront).

usage: publish_site.py BUCKET [replay_name ...]
"""
import json
import mimetypes
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
CAM = "TX_Cowhouse_Ck_at_W_Range_Rd_nr_Ft_Hood"


def compact_replay(name):
    d = ROOT / "data/replay" / CAM / name
    cycles = [json.loads(l) for l in (d / "cycles.jsonl").read_text().splitlines() if l.strip()]
    gage = json.loads((d / "gage.json").read_text())
    up = json.loads((d / "upstream.json").read_text())
    out = []
    for c in cycles:
        cam = next((s["output"] for s in c["steps"] if s["step"] == "tool" and s["tool"] == "measure_camera"
                    and not (s.get("input") or {}).get("frames_back")), {})
        steps = [{"step": s["step"], "tool": s.get("tool"), "input": s.get("input"),
                  "output": {k: v for k, v in (s.get("output") or {}).items()
                             if k in ("usable", "reasons", "stage_ft", "below_range", "above_range", "available", "note",
                                      "latest_ft", "trend_ft_per_h", "upstream", "blocked_by_guardrail", "ok")},
                  "detail": s.get("detail")} for s in c["steps"] if s["step"] != "thought"]
        out.append({"at": c["at"], "t": c["t"], "summary": c["summary"], "state": c["state"], "frame": c.get("frame"),
                    "truth_stage_ft": c["truth_stage_ft"], "model_calls": c["model_calls"],
                    "camera_stage_ft": cam.get("stage_ft") if cam.get("usable") else None,
                    "camera_note": None if cam.get("usable") else "; ".join(cam.get("reasons", [])) or None,
                    "steps": steps})
    return {"t0": cycles[0]["t"] - 1800, "t1": cycles[-1]["t"] + 900, "gage": gage,
            "upstream": next(iter(up.values()), []), "cycles": out}


def main(bucket, *replays):
    s3 = boto3.Session().client("s3")

    def put(key, body, ctype=None, cache="no-cache"):
        ctype = ctype or mimetypes.guess_type(key)[0] or "application/octet-stream"
        s3.put_object(Bucket=bucket, Key="site/" + key, Body=body, ContentType=ctype, CacheControl=cache)

    for f in (ROOT / "web").iterdir():
        if f.is_file():
            put(f.name, f.read_bytes())
    ev = ROOT / "data/eval" / CAM
    put("data/eval/summary.json", (ev / "summary.json").read_bytes(), "application/json")
    put("data/eval/timeseries.png", (ev / "timeseries.png").read_bytes(), "image/png")
    for name in replays or ["flood-2026-10-02"]:
        rp = compact_replay(name)
        put(f"data/replay/{name}/replay.json", json.dumps(rp).encode(), "application/json", "max-age=300")
        for fr in (ROOT / "data/replay" / CAM / name / "frames").glob("*.jpg"):
            put(f"data/replay/{name}/frames/{fr.name}", fr.read_bytes(), "image/jpeg", "max-age=86400")
        print(name, len(rp["cycles"]), "cycles")


if __name__ == "__main__":
    main(*sys.argv[1:])
