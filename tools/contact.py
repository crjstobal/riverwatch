"""Contact sheet: for each candidate camera, a low-water and a peak image side by side."""
import json, sys, urllib.request, urllib.parse
from pathlib import Path
import cv2, numpy as np

NIMS = "https://api.waterdata.usgs.gov/nims/v0"
ROOT = Path(__file__).resolve().parents[1]

def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()

def nearest(cam, day):
    q = urllib.parse.urlencode({"camId": cam, "after": f"{day}T16:00:00Z", "before": f"{day}T20:00:00Z", "limit": 1})
    files = json.loads(get(f"{NIMS}/listFiles?{q}"))
    return files[0] if files else None

def img(cam, f):
    base = json.loads(get(f"{NIMS}/cameras?camId={cam}"))[0]["smallDir"]
    a = np.frombuffer(get(base + f), np.uint8)
    return cv2.resize(cv2.imdecode(a, cv2.IMREAD_COLOR), (480, 270))

ev = json.loads((ROOT / "data/events.json").read_text())
seen, rows = set(), []
for e in ev:
    cam = e["cams"][0]
    if cam in seen or e["rise_ft"] < 4 or e["rise_ft"] > 30:
        continue
    seen.add(cam)
    try:
        a, b = nearest(cam, e["from"][0]), nearest(cam, e["peak"][0])
        if not (a and b):
            continue
        row = np.hstack([img(cam, a), img(cam, b)])
    except Exception as ex:
        print(cam, ex); continue
    cv2.putText(row, f"{len(rows)} {cam[:48]} +{e['rise_ft']}ft", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    rows.append(row); print(len(rows) - 1, cam, e["from"], e["peak"])
    if len(rows) == 16:
        break
for k in range(0, len(rows), 4):
    cv2.imwrite(str(ROOT / f"data/contact_{k // 4}.jpg"), np.vstack(rows[k:k + 4]))
