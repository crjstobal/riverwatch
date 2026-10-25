"""Prototype: wetted fraction from local NCC against dry references vs official stage."""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from peek import load

ROOT = Path(__file__).resolve().parents[1]
cam = sys.argv[1]
gt, gv, imgs, it, stage = load(cam)
hours = ((it / 3600) - 5) % 24  # local CDT approx

def gray(p):
    im = cv2.imread(str(p))
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)

def lncc(a, b, k=21):
    bf = lambda x: cv2.boxFilter(x, -1, (k, k))
    ma, mb = bf(a), bf(b)
    va, vb = bf(a * a) - ma * ma, bf(b * b) - mb * mb
    cov = bf(a * b) - ma * mb
    return cov / np.sqrt(np.maximum(va * vb, 1e-3)), va, vb

base = np.quantile(stage, 0.05)
dry = [k for k in range(len(imgs)) if stage[k] < base + 0.05 and 8 <= hours[k] <= 18 and it[k] < it[0] + 20 * 86400]
refs = [gray(imgs[k]) for k in dry[:: max(1, len(dry) // 8)][:8]]
print("refs", len(refs), "dry candidates", len(dry))
H, W = refs[0].shape
roi = np.zeros((H, W), np.uint8); roi[int(H * 0.30):int(H * 0.72), :] = 1  # band between far bank and near ramp

rows = []
for k in range(0, len(imgs), 2):
    if not (8 <= hours[k] <= 18):
        continue
    g = gray(imgs[k])
    if g is None or g.mean() < 60:
        continue
    best = None
    for r in refs:
        n, va, vb = lncc(g, r)
        tex_ok = vb > 20  # only where the reference has texture
        n = np.where(tex_ok, n, 1.0)
        best = n if best is None else np.maximum(best, n)
    wet = (best < 0.3) & (roi > 0)
    rows.append((stage[k], wet.sum() / roi.sum(), k))
rows = np.array(rows)
print("n", len(rows), "corr", np.corrcoef(rows[:, 0], rows[:, 1])[0, 1])
# scatter
pl = np.full((400, 600, 3), 255, np.uint8)
s0, s1 = rows[:, 0].min(), rows[:, 0].max()
for s, f, _ in rows:
    cv2.circle(pl, (int(20 + f * 560), int(380 - (s - s0) / (s1 - s0) * 360)), 2, (180, 0, 0), -1)
cv2.imwrite(str(ROOT / "data/proto_scatter.png"), pl)
# save a wet overlay example at peak
k = int(rows[np.argmax(rows[:, 0]), 2]); g = gray(imgs[k])
best = np.max([np.where(lncc(g, r)[2] > 20, lncc(g, r)[0], 1.0) for r in refs], axis=0)
im = cv2.imread(str(imgs[k])); im[(best < 0.3) & (roi > 0)] = (0.5 * im[(best < 0.3) & (roi > 0)] + (0, 0, 127)).astype(np.uint8)
k2 = int(rows[np.argsort(rows[:, 0])[len(rows) // 2], 2]); g2 = gray(imgs[k2])
best2 = np.max([np.where(lncc(g2, r)[2] > 20, lncc(g2, r)[0], 1.0) for r in refs], axis=0)
im2 = cv2.imread(str(imgs[k2])); m2 = (best2 < 0.3) & (roi > 0); im2[m2] = (0.5 * im2[m2] + (0, 0, 127)).astype(np.uint8)
cv2.imwrite(str(ROOT / "data/proto_overlay.jpg"), np.hstack([im, im2]))
