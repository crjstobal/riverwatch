"""Plot a camera's gage record and tile frames at several stages (diagnostic only)."""
import json, sys
from datetime import datetime
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load(cam):
    d = ROOT / "data/cams" / cam
    g = json.loads((d / "gauge.json").read_text())
    gt = np.array([datetime.fromisoformat(t).timestamp() for t, _ in g]); gv = np.array([v for _, v in g])
    o = np.argsort(gt); gt, gv = gt[o], gv[o]
    imgs = sorted((d / "img").glob("*.jpg"))
    it = np.array([datetime.strptime(p.stem.split("___")[1], "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=__import__("datetime").timezone.utc).timestamp() for p in imgs])
    return gt, gv, imgs, it, np.interp(it, gt, gv)

if __name__ == "__main__":
    cam = sys.argv[1]
    gt, gv, imgs, it, stage = load(cam)
    print("gauge", len(gt), "range", gv.min(), gv.max(), "dt median min", np.median(np.diff(gt)) / 60)
    # plot
    W, H = 1200, 300; pl = np.full((H, W, 3), 255, np.uint8)
    x = ((gt - gt[0]) / (gt[-1] - gt[0]) * (W - 1)).astype(int); y = (H - 10 - (gv - gv.min()) / (np.ptp(gv) + 1e-6) * (H - 20)).astype(int)
    cv2.polylines(pl, [np.stack([x, y], 1).reshape(-1, 1, 2)], False, (200, 0, 0), 1)
    cv2.imwrite(str(ROOT / "data/peek_gauge.png"), pl)
    # frames at stage quantiles, daytime only (mean brightness)
    picks = []
    for q in [0.02, 0.3, 0.6, 0.8, 0.95, 1.0]:
        target = np.quantile(stage, q)
        for k in np.argsort(np.abs(stage - target)):
            im = cv2.imread(str(imgs[k]))
            if im is not None and im.mean() > 70:
                picks.append((stage[k], imgs[k].stem[-20:], im)); break
    tiles = []
    for s, name, im in picks:
        im = cv2.resize(im, (640, 360)); cv2.putText(im, f"{s:.2f} ft {name}", (8, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2); tiles.append(im)
    cv2.imwrite(str(ROOT / "data/peek_frames.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 6, 2)]))
    print(im.shape if picks else None, cv2.imread(str(imgs[0])).shape)
