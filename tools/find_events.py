"""Scan every USGS HIVIS camera's paired streamgage and rank recent rise events.

Output: data/events.json, sorted by the largest gage-height rise in the window.
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NIMS = "https://api.waterdata.usgs.gov/nims/v0"
IV = "https://waterservices.usgs.gov/nwis/dv/"


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def main(start="2026-06-27", end="2026-10-25"):
    cams = [c for c in get(f"{NIMS}/cameras") if c.get("nwisId") and not c.get("hideCam")]
    by_site = {}
    for c in cams:
        by_site.setdefault(c["nwisId"], []).append(c)
    sites = list(by_site)
    events = []
    for i in range(0, len(sites), 100):
        batch = ",".join(sites[i:i + 100])
        url = f"{IV}?format=json&sites={batch}&parameterCd=00065&startDT={start}&endDT={end}&siteStatus=all"
        try:
            ts = get(url)["value"]["timeSeries"]
        except Exception as e:
            print("batch failed", i, e, file=sys.stderr)
            continue
        for s in ts:
            site = s["sourceInfo"]["siteCode"][0]["value"]
            vals = []
            for block in s["values"]:
                for v in block["value"]:
                    try:
                        vals.append((v["dateTime"][:10], float(v["value"])))
                    except ValueError:
                        pass
            vals = [v for v in vals if v[1] > -100]
            if len(vals) < 20:
                continue
            # largest rise from a running minimum to a later maximum
            best, lo = (0, None, None), vals[0]
            for d, h in vals:
                if h < lo[1]:
                    lo = (d, h)
                if h - lo[1] > best[0]:
                    best = (h - lo[1], lo, (d, h))
            if best[1]:
                events.append({"site": site, "rise_ft": round(best[0], 2), "from": best[1], "peak": best[2],
                               "cams": [c["camId"] for c in by_site.get(site, [])],
                               "name": s["sourceInfo"]["siteName"]})
        print(f"{i + 100}/{len(sites)}", file=sys.stderr)
    events.sort(key=lambda e: -e["rise_ft"])
    out = ROOT / "data" / "events.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(events, indent=1))
    for e in events[:40]:
        print(e["rise_ft"], e["from"], e["peak"], e["name"], e["cams"])


if __name__ == "__main__":
    main(*sys.argv[1:])
