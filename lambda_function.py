"""AWS Lambda entry point (arm64 / Graviton).

Two triggers:
  * EventBridge schedule (every 15 min): for each watched camera whose next check is due, run one agent cycle
    on live USGS data, then publish status, overlay frame and trace to S3 (served by CloudFront).
  * Function URL (behind CloudFront at /api/*):
      GET  /api/health
      POST /api/decision  {"cam_id", "decision": "approve"|"reject", "code"}   duty-operator approval

State lives in S3 (state/<cam>.json). Public files live under site/data/ in the same bucket.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2

from riverwatch import agent, usgs, vision
from riverwatch.bedrock import BedrockModel
from riverwatch.store import S3Store

ROOT = Path(__file__).resolve().parent
BUCKET = os.environ.get("RIVERWATCH_BUCKET", "")
TOPIC = os.environ.get("RIVERWATCH_TOPIC_ARN", "")
CODE_SHA256 = os.environ.get("RIVERWATCH_OPERATOR_CODE_SHA256", "")
CAMERAS = json.loads((ROOT / "cameras.json").read_text())
WATCHED = [c for c in os.environ.get("RIVERWATCH_CAMERAS", ",".join(CAMERAS)).split(",") if c]


def _store():
    return S3Store(BUCKET)


def _publish_status(store, cam, state, rec=None, overlay=None):
    cfg = CAMERAS[cam]
    status = store.get_json("site/data/live/status.json", {"cameras": {}})
    entry = {"cam_id": cam, "label": cfg["label"], "site": cfg["site"], "updated": usgs.iso(time.time()),
             "alert": state["alert"], "watch": state["watch"], "pending": state["pending"],
             "next_check_minutes": state["next_check_minutes"], "summary": state["summary"],
             "issues": state["issues"][-3:], "readings": state["readings"][-48:]}
    if rec:
        entry["last_cycle"] = {"at": rec["at"], "model_calls": rec["model_calls"], "latency_s": rec["latency_s"],
                               "trace_key": rec.get("trace_key")}
    if overlay is not None:
        ok, jpg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        store.put_bytes(f"site/data/live/{cam}.jpg", jpg.tobytes(), "image/jpeg")
        entry["frame"] = f"data/live/{cam}.jpg?t={int(time.time())}"
    status["cameras"][cam] = entry
    status["updated"] = usgs.iso(time.time())
    store.put_json("site/data/live/status.json", status)


def _notify(subject, message):
    if not TOPIC:
        return
    import boto3
    boto3.client("sns").publish(TopicArn=TOPIC, Subject=subject[:99], Message=message)


def tick(event=None):
    store = _store()
    model = BedrockModel()
    now = time.time()
    done = []
    for cam in WATCHED:
        cfg = CAMERAS[cam]
        state = store.get_json(f"state/{cam}.json") or agent.new_state()
        last = state.get("last_cycle_t", 0)
        if now - last < 60 * state.get("next_check_minutes", 60) - 90 and not (event or {}).get("force"):
            continue
        prof = vision.Profile.load(ROOT / "profiles" / cam)
        ctx = agent.Context(cam, cfg, prof, now, state)
        pending_before = json.dumps(state.get("pending"))
        rec = agent.run_cycle(ctx, model)
        state["last_cycle_t"] = now
        key = f"site/data/traces/{cam}/{datetime.fromtimestamp(now, timezone.utc):%Y/%m/%d/%H%M}.json"
        rec["trace_key"] = key[len("site/"):]
        store.put_json(key, rec)
        store.put_json(f"state/{cam}.json", state)
        _publish_status(store, cam, state, rec, ctx.overlay)
        if state.get("pending") and json.dumps(state["pending"]) != pending_before:
            p = state["pending"]
            _notify(f"RiverWatch: approval needed ({p['level']}) {cfg['label']}",
                    f"{p['reason']}\n\nEvidence: {p['evidence']}\n\nApprove or reject on the RiverWatch page.")
        done.append({"cam": cam, "summary": rec["summary"], "calls": rec["model_calls"]})
    return {"ran": done}


def decision(body):
    code = str(body.get("code", ""))
    if not CODE_SHA256 or not hmac.compare_digest(hashlib.sha256(code.encode()).hexdigest(), CODE_SHA256):
        return 403, {"error": "invalid operator code"}
    cam = body.get("cam_id")
    if cam not in CAMERAS:
        return 400, {"error": "unknown camera"}
    store = _store()
    state = store.get_json(f"state/{cam}.json") or agent.new_state()
    p = state.get("pending")
    if not p:
        return 409, {"error": "nothing pending"}
    now = usgs.iso(time.time())
    if body.get("decision") == "approve":
        state["alert"] = p["level"]
        state["approvals"].append({**p, "approved": now, "by": "duty operator"})
        _notify(f"RiverWatch ALERT: {p['level']} at {CAMERAS[cam]['label']}", f"{p['reason']}\n\nEvidence: {p['evidence']}")
    else:
        state["approvals"].append({**p, "rejected": now, "by": "duty operator"})
    state["approvals"] = state["approvals"][-20:]
    state["pending"] = None
    store.put_json(f"state/{cam}.json", state)
    _publish_status(store, cam, state)
    return 200, {"ok": True, "alert": state["alert"]}


def _http(status, obj):
    return {"statusCode": status, "headers": {"content-type": "application/json", "cache-control": "no-store"},
            "body": json.dumps(obj)}


def lambda_handler(event, context):
    try:
        if "requestContext" in event and "http" in event["requestContext"]:
            method = event["requestContext"]["http"]["method"]
            path = event.get("rawPath", "")
            if path.endswith("/health"):
                return _http(200, {"ok": True, "cameras": WATCHED, "opencv": cv2.__version__})
            if path.endswith("/decision") and method == "POST":
                body = event.get("body") or "{}"
                if event.get("isBase64Encoded"):
                    import base64
                    body = base64.b64decode(body).decode()
                return _http(*decision(json.loads(body)))
            return _http(404, {"error": "not found"})
        return tick(event)
    except Exception as e:
        traceback.print_exc()
        if "requestContext" in event:
            return _http(500, {"error": type(e).__name__})
        raise
