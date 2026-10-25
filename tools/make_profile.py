"""Production profile: calibration fitted on every measured day (data/eval/<cam>/frames.json)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from riverwatch import vision  # noqa: E402

cam = sys.argv[1]
frames = json.loads((ROOT / "data/eval" / cam / "frames.json").read_text())
ok = [r for r in frames if r["usable"] and r["truth"] is not None]
floor = float(np.quantile([r["truth"] for r in ok], 0.05))
pairs = [(r["waterline"], r["truth"]) for r in ok if r["waterline"] is not None and r["truth"] > floor + 0.7]
prof = vision.Profile.load(ROOT / "profiles" / cam)
prof.calib = vision.fit_calibration(pairs, floor)
prof.save(ROOT / "profiles" / cam)
print(prof.calib)
