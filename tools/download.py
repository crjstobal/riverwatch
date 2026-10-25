"""Download a camera's image series and its paired gage-height record (USGS NIMS + NWIS IV).

usage: download.py CAM_ID START END   (dates YYYY-MM-DD, UTC)
Writes data/cams/<cam>/img/*.jpg (720 px), gauge.json [[iso_utc, feet], ...] and meta.json.
"""
import json, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NIMS = "https://api.waterdata.usgs.gov/nims/v0"
IV = "https://waterservices.usgs.gov/nwis/iv/"
ROOT = Path(__file__).resolve().parents[1]


def get(url, raw=False, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                b = r.read()
            return b if raw else json.loads(b)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def main(cam, start, end):
    out = ROOT / "data" / "cams" / cam
    (out / "img").mkdir(parents=True, exist_ok=True)
    meta = get(f"{NIMS}/cameras?camId={cam}")[0]
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    q = urllib.parse.urlencode({"camId": cam, "after": f"{start}T00:00:00Z", "before": f"{end}T23:59:59Z",
                                "limit": 50000, "recent": "false"})
    files = get(f"{NIMS}/listFiles?{q}")

    def fetch(f):
        p = out / "img" / f
        if not p.exists():
            p.write_bytes(get(meta["smallDir"] + f, raw=True))

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, files))
    ts = get(f"{IV}?format=json&sites={meta['nwisId']}&parameterCd=00065&startDT={start}&endDT={end}")
    vals = []
    # a site can publish several sensors for the same parameter: keep the primary one
    blocks = [b for s in ts["value"]["timeSeries"] for b in s["values"] if b["value"]]
    primary = [b for b in blocks if "primary" in (b["method"][0].get("methodDescription") or "").lower()]
    block = (primary or blocks)[0]
    vals = [[v["dateTime"], float(v["value"])] for v in block["value"] if float(v["value"]) > -100]
    (out / "gauge.json").write_text(json.dumps(vals))
    print(cam, len(files), "images,", len(vals), "gauge readings")


if __name__ == "__main__":
    main(*sys.argv[1:])
