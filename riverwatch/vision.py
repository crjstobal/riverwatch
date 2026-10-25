"""OpenCV 5 perception for RiverWatch: image quality, registration, wetted area and stage estimate.

Everything here is classical computer vision, no learned weights:
  1. quality()    - is this frame usable? (darkness, blur, fog / lens obstruction, camera shift)
  2. register()   - align the frame to the camera's reference view (ECC, translation + small rotation)
  3. flow_mask()  - flowing water: the surface decorrelates between two frames 15 minutes apart while
                    banks, piers and debris do not (local normalized cross-correlation, box filters)
  4. waterline()  - per-column top edge of the water body inside the channel polygon; the median row is
                    a "virtual staff gauge" reading in image coordinates
  5. Calibration  - monotone map from waterline height to stage, fitted on a few frames with known stage

A camera profile (see profiles/) stores the channel polygon, the dry references and the calibration table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

NCC_WINDOW = 21           # px, local correlation window on 720 px frames


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)


def local_ncc(a: np.ndarray, b: np.ndarray, k: int = NCC_WINDOW):
    """Per-pixel normalized cross-correlation over a k x k window, computed with box filters.

    Returns (ncc, var_a, var_b). Equivalent to TM_CCOEFF_NORMED evaluated at every pixel, but O(1) per pixel.
    """
    bf = lambda x: cv2.boxFilter(x, cv2.CV_32F, (k, k))
    ma, mb = bf(a), bf(b)
    va = bf(a * a) - ma * ma
    vb = bf(b * b) - mb * mb
    cov = bf(a * b) - ma * mb
    return cov / np.sqrt(np.maximum(va * vb, 1e-3)), va, vb


# ---------------------------------------------------------------- quality

@dataclass
class Quality:
    brightness: float      # mean gray level, 0-255
    saturation: float      # mean HSV saturation; ~0 when the camera switches to infrared at night
    sharpness: float       # variance of the Laplacian
    contrast: float        # RMS contrast (std of gray)
    usable: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def quality(img: np.ndarray, sharp_floor: float = 40.0) -> Quality:
    g = to_gray(img)
    brightness = float(g.mean())
    saturation = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()) if img.ndim == 3 else 0.0
    sharpness = float(cv2.Laplacian(g, cv2.CV_32F).var())
    contrast = float(g.std())
    reasons = []
    if saturation < 5:
        reasons.append("night infrared mode (no colour, water texture not visible)")
    elif brightness < 55:
        reasons.append("too dark (night or underexposed)")
    if contrast < 25:
        reasons.append("low contrast (fog, condensation or glare)")
    if sharpness < sharp_floor:
        reasons.append("blurred (lens water, smudge or defocus)")
    return Quality(brightness, saturation, sharpness, contrast, not reasons, reasons)


# ---------------------------------------------------------------- registration

def register(img: np.ndarray, ref_gray: np.ndarray, static: np.ndarray | None = None,
             iters: int = 60) -> tuple[np.ndarray, float]:
    """Align img to the reference view with ECC (Euclidean model) on a half-size pyramid level.

    `static` masks the pixels that should not change with the river (bridge, piers, banks); the water
    itself is excluded so a flood does not drag the alignment. Returns (aligned image, ECC correlation),
    or the input and 0 on failure.
    """
    g = to_gray(img)
    small = lambda x: cv2.resize(x, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    mask = None if static is None else cv2.resize(static, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(small(ref_gray), small(g), warp, cv2.MOTION_EUCLIDEAN,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-4), mask, 5)
    except cv2.error:
        return img, 0.0
    warp[:, 2] *= 2.0
    h, w = g.shape
    out = cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, float(cc)


# ---------------------------------------------------------------- profile

MIN_ECC = 0.70            # below this the frame does not match the calibrated view
FLOW_NCC = 0.50           # surface patches that decorrelate in 15 min are flowing water
MIN_FLOW_TEXTURE = 4.0
MIN_COLUMNS = 0.35        # share of channel columns that must see water to report a waterline


@dataclass
class Profile:
    cam_id: str
    channel: np.ndarray            # uint8 mask, 1 inside the channel polygon
    ref: np.ndarray                # reference view (gray float32) used for registration
    calib: dict = field(default_factory=dict)  # see fit_calibration
    sharp_floor: float = 40.0

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "channel.png"), self.channel * 255)
        cv2.imwrite(str(folder / "ref.png"), self.ref.astype(np.uint8))
        (folder / "profile.json").write_text(json.dumps(
            {"cam_id": self.cam_id, "calib": self.calib, "sharp_floor": self.sharp_floor}, indent=1))

    @classmethod
    def load(cls, folder: Path) -> "Profile":
        meta = json.loads((folder / "profile.json").read_text())
        channel = (cv2.imread(str(folder / "channel.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        ref = cv2.imread(str(folder / "ref.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return cls(meta["cam_id"], channel, ref, meta["calib"], meta.get("sharp_floor", 40.0))


def build_profile(cam_id: str, ref_frame: np.ndarray, polygon: list[tuple[float, float]]) -> Profile:
    """polygon is in relative coordinates (0-1) so it survives a change of image size."""
    ref = to_gray(ref_frame)
    h, w = ref.shape
    channel = np.zeros((h, w), np.uint8)
    cv2.fillPoly(channel, [np.array([(x * w, y * h) for x, y in polygon], np.int32)], 1)
    return Profile(cam_id, channel, ref)


# ---------------------------------------------------------------- measurement

WATER_DENSITY = 0.92      # share of a water body that must decorrelate: a flood surface changes everywhere
WATER_MAX_TEXTURE = 14.0  # mean local gray std (9x9) inside a body: turbid water is smooth, sunlit gravel is rough


def flow_mask(cur: np.ndarray, prev: np.ndarray, prof: Profile) -> np.ndarray:
    """Flowing water inside the channel: patches that decorrelate between two registered frames.

    Candidate bodies are then tested one by one. Moving sun and shadows also decorrelate a dry gravel bed, but only
    in parts (shadow edges) and on a rough surface; a flood decorrelates densely and is smooth.
    """
    g = to_gray(cur)
    n, va, vb = local_ncc(g, to_gray(prev), 15)
    raw = (n < FLOW_NCC) & (np.minimum(va, vb) > MIN_FLOW_TEXTURE) & (prof.channel > 0)
    m = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mean = cv2.boxFilter(g, cv2.CV_32F, (9, 9))
    tex = np.sqrt(np.maximum(cv2.boxFilter(g * g, cv2.CV_32F, (9, 9)) - mean * mean, 0))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(num, bool)
    min_area = 0.01 * max(1, int(prof.channel.sum()))
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= min_area:
            continue
        body = lab == i
        keep[i] = raw[body].mean() >= WATER_DENSITY and tex[body].mean() <= WATER_MAX_TEXTURE
    return keep[lab].astype(np.uint8)


def waterline(mask: np.ndarray, prof: Profile) -> tuple[float | None, float, np.ndarray]:
    """Median top edge of the water body over channel columns.

    Returns (height in 0-1 where 1 is the top of the channel polygon, share of columns with water,
    per-column top rows or -1).
    """
    rows = np.where(prof.channel.any(axis=1))[0]
    top, bottom = int(rows.min()), int(rows.max())
    cols = np.where(prof.channel.any(axis=0))[0]
    tops = np.full(mask.shape[1], -1, np.int32)
    has = mask[:, cols].any(axis=0)
    first = mask[:, cols].argmax(axis=0)
    tops[cols[has]] = first[has]
    share = float(has.mean()) if len(cols) else 0.0
    if share < MIN_COLUMNS:
        return None, share, tops
    med = float(np.median(tops[cols[has]]))
    return (bottom - med) / max(1, bottom - top), share, tops


def fit_calibration(pairs: list[tuple[float, float]], floor_ft: float | None = None) -> dict:
    """Robust monotone fit  stage = a + b * exp(c * h)  (b, c > 0), so it can extrapolate above the
    highest water seen during calibration. Far banks steepen with height, hence the convex shape.

    Fit, drop points more than 1.5 ft off, refit; c is chosen by grid search on the median residual.
    """
    xy = np.array(pairs, np.float64)
    best = None
    for c in np.linspace(0.5, 8.0, 31):
        X = np.c_[np.ones(len(xy)), np.exp(c * xy[:, 0])]
        w, *_ = np.linalg.lstsq(X, xy[:, 1], rcond=None)
        keep = np.abs(X @ w - xy[:, 1]) < 1.5
        if keep.sum() >= 5:
            w, *_ = np.linalg.lstsq(X[keep], xy[keep, 1], rcond=None)
        if w[1] <= 0:
            continue
        med = float(np.median(np.abs(X @ w - xy[:, 1])))
        if best is None or med < best[0]:
            best = (med, c, w)
    if best is None:
        raise ValueError("calibration failed: waterline does not increase with stage")
    med, c, w = best
    floor = float(floor_ft if floor_ft is not None else xy[:, 1].min())
    return {"a": round(float(w[0]), 4), "b": round(float(w[1]), 6), "c": round(float(c), 3),
            "floor_ft": round(floor, 2), "median_abs_residual_ft": round(med, 3), "n": int(len(xy)),
            "max_stage_seen_ft": round(float(xy[:, 1].max()), 2),
            "max_waterline_seen": round(float(np.quantile(xy[:, 0], 0.98)), 4)}


EXTRAPOLATION_MARGIN = 0.08  # waterline height above the highest calibrated water that we still trust


def stage_from_height(h: float, calib: dict) -> float:
    h = min(h, calib.get("max_waterline_seen", 1.0) + EXTRAPOLATION_MARGIN)
    return max(calib["floor_ft"], calib["a"] + calib["b"] * float(np.exp(calib["c"] * h)))


def above_range(h: float, calib: dict) -> bool:
    """Waterline far above anything seen in calibration: wind in vegetation, glare or a real record flood."""
    return h > calib.get("max_waterline_seen", 1.0) + EXTRAPOLATION_MARGIN


@dataclass
class Reading:
    usable: bool
    quality: dict
    reasons: list[str] = field(default_factory=list)
    waterline: float | None = None       # 0-1 within the channel polygon
    columns_with_water: float | None = None
    stage_ft: float | None = None
    below_range: bool = False            # no flowing water seen: stage is at or below the calibrated floor
    above_range: bool = False            # waterline above the calibrated range: needs confirmation
    floor_ft: float | None = None
    ecc: float | None = None

    def as_dict(self):
        return dict(self.__dict__)


def read_stage(img: np.ndarray, prev: np.ndarray | None, prof: Profile) -> tuple[Reading, np.ndarray | None]:
    """Full perception step for one frame and the frame before it. Returns the reading and the water mask."""
    q = quality(img, prof.sharp_floor)
    reasons = list(q.reasons)
    if prev is None:
        reasons.append("no previous frame to measure flow against")
    elif not quality(prev, prof.sharp_floor).usable:
        reasons.append("previous frame unusable")
    if reasons:
        return Reading(False, q.as_dict(), reasons), None
    static = (cv2.dilate(prof.channel, np.ones((31, 31), np.uint8)) == 0).astype(np.uint8)
    a, cc_a = register(img, prof.ref, static)
    b, cc_b = register(prev, prof.ref, static)
    cc = min(cc_a, cc_b)
    if cc < MIN_ECC:
        return Reading(False, q.as_dict(), ["view does not match the calibrated camera (moved or obstructed)"],
                       ecc=round(cc, 3)), None
    mask = flow_mask(a, b, prof)
    h, share, _ = waterline(mask, prof)
    floor = prof.calib.get("floor_ft") if prof.calib else None
    if h is None:
        return Reading(True, q.as_dict(), [], waterline=None, columns_with_water=round(share, 3), stage_ft=floor,
                       below_range=True, floor_ft=floor, ecc=round(cc, 3)), mask
    stage = stage_from_height(h, prof.calib) if prof.calib else None
    hi = above_range(h, prof.calib) if prof.calib else False
    return Reading(True, q.as_dict(), [], waterline=round(h, 4), columns_with_water=round(share, 3),
                   stage_ft=None if stage is None else round(stage, 2), above_range=hi, floor_ft=floor,
                   ecc=round(cc, 3)), mask
