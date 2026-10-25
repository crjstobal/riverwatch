"""Still images and the flood timelapse for the demo video (1920x1080), from the real data.

usage: video_assets.py      writes docs/video/assets/
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from riverwatch import vision  # noqa: E402

CAM = "TX_Cowhouse_Ck_at_W_Range_Rd_nr_Ft_Hood"
IMG = ROOT / "data/cams" / CAM / "img"
OUT = ROOT / "docs/video/assets"
W, H = 1920, 1080
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
BG = (14, 22, 30)
INK = (235, 241, 245)
MUTED = (150, 166, 178)
ACCENT = (95, 178, 230)


def font(size, weight=600):
    return ImageFont.truetype(FONT_BOLD if weight >= 600 else FONT, size)


def canvas():
    return Image.new("RGB", (W, H), BG)


def paste_cv(im, bgr, box):
    x, y, w, h = box
    rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    im.paste(Image.fromarray(rgb), (x, y))


def frame_at(iso):
    t = datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    files = sorted(IMG.glob("*.jpg"))
    times = [datetime.strptime(p.stem.split("___")[1], "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=timezone.utc).timestamp() for p in files]
    k = int(np.argmin([abs(x - t) for x in times]))
    return files[k], files[k - 1], times[k]


def gage():
    g = json.loads((ROOT / "data/cams" / CAM / "gauge.json").read_text())
    return sorted((datetime.fromisoformat(t).timestamp(), v) for t, v in g)


PHOTO = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/_Fotos/Fotos Javi/javier-cristobal.webp"
AUTHOR = "Javier Cristóbal"


def avatar(im, x, y, size=220):
    """Round author photo with a name label; the team on screen, as the rules ask."""
    ph = Image.open(PHOTO).convert("RGB").resize((size, size), Image.LANCZOS)
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).ellipse((0, 0, size - 1, size - 1), fill=255)
    im.paste(ph, (x, y), m)
    d = ImageDraw.Draw(im)
    d.ellipse((x - 3, y - 3, x + size + 2, y + size + 2), outline=ACCENT, width=4)
    d.text((x + size + 36, y + size // 2 - 46), AUTHOR, font=font(44, 700), fill=INK)
    d.text((x + size + 36, y + size // 2 + 10), "design, computer vision, AWS", font=font(32, 400), fill=MUTED)


def title_card(name, title, subtitle, foot=""):
    im = canvas(); d = ImageDraw.Draw(im)
    d.text((140, 250), title, font=font(110, 700), fill=INK)
    d.text((140, 400), subtitle, font=font(44, 400), fill=MUTED)
    avatar(im, 140, 560)
    if foot:
        d.text((140, 860), foot, font=font(30, 400), fill=ACCENT)
    im.save(OUT / f"{name}.png")


def timelapse():
    """Frames of 2026-10-02 from 08:00 to 17:00 CDT with local time and official stage."""
    g = gage(); gt = np.array([x[0] for x in g]); gv = np.array([x[1] for x in g])
    d = OUT / "timelapse"; d.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2026, 10, 2, 13, 0, tzinfo=timezone.utc).timestamp()
    n = 0
    for k in range(0, 37):
        f, _, t = frame_at(datetime.fromtimestamp(t0 + k * 900, timezone.utc).isoformat())
        bgr = cv2.imread(str(f))
        im = canvas(); paste_cv(im, bgr, (300, 150, 1320, 990))
        dr = ImageDraw.Draw(im)
        local = datetime.fromtimestamp(t, timezone.utc) - timedelta(hours=5)
        stage = float(np.interp(t, gt, gv)); depth = stage - 3.99
        dr.rectangle((300, 30, 1620, 130), fill=(0, 0, 0))
        dr.text((330, 52), f"Oct 2, {local:%I:%M %p} CDT", font=font(42, 600), fill=INK)
        col = (90, 200, 120) if depth < 0.5 else (240, 170, 60) if depth < 1 else (240, 110, 60) if depth < 2 else (235, 70, 80)
        txt = f"official gage {stage:.1f} ft   " + ("crossing dry" if depth <= 0.1 else f"{depth:.1f} ft over the crossing")
        dr.text((860, 56), txt, font=font(38, 600), fill=col)
        im.save(d / f"{n:03d}.png"); n += 1
    print("timelapse frames", n)


def pipeline():
    """Perception steps on a real flood frame pair (2026-10-02 16:45 UTC)."""
    prof = vision.Profile.load(ROOT / "profiles" / CAM)
    cur_p, prev_p, _ = frame_at("2026-10-02T16:45:00Z")
    cur, prev = cv2.imread(str(cur_p)), cv2.imread(str(prev_p))
    static = (cv2.dilate(prof.channel, np.ones((31, 31), np.uint8)) == 0).astype(np.uint8)
    a, cc = vision.register(cur, prof.ref, static)
    b, _ = vision.register(prev, prof.ref, static)
    n, va, vb = vision.local_ncc(vision.to_gray(a), vision.to_gray(b), 15)
    heat = cv2.applyColorMap(np.clip((1 - n) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    raw = ((n < vision.FLOW_NCC) & (np.minimum(va, vb) > vision.MIN_FLOW_TEXTURE) & (prof.channel > 0)).astype(np.uint8)
    mask = vision.flow_mask(a, b, prof)
    h, share, tops = vision.waterline(mask, prof)
    stage = vision.stage_from_height(h, prof.calib)

    def step(name, bgr, title, caption):
        im = canvas(); paste_cv(im, bgr, (90, 170, 1040, 780))
        d = ImageDraw.Draw(im)
        d.text((90, 70), title, font=font(54, 700), fill=INK)
        y = 190
        for line in caption:
            d.text((1180, y), line, font=font(32, 400), fill=MUTED); y += 50
        im.save(OUT / f"{name}.png")

    ov = a.copy(); cnt, _ = cv2.findContours(prof.channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnt, -1, (255, 255, 255), 2)
    step("p1_quality", ov, "1. Quality gate and registration",
         ["Infrared night, blur, fog: rejected.", "", "findTransformECC aligns the frame", "to the calibrated view using only", "the bridge, piers and banks.", "", f"ECC correlation: {cc:.2f}", "", "White: the channel polygon."])
    step("p2_ncc", heat, "2. What changed in 15 minutes",
         ["Dense local normalized", "cross-correlation between two", "registered frames (box filters).", "", "Bright: the surface decorrelated.", "Flowing water changes everywhere;", "concrete and debris do not."])
    tint = a.copy(); tint[raw > 0] = (0.4 * tint[raw > 0] + np.array([150, 90, 20])).astype(np.uint8)
    tint[mask > 0] = (0.45 * a[mask > 0] + np.array([235, 140, 40]) * 0.55).astype(np.uint8)
    step("p3_bodies", tint, "3. Keep only real water bodies",
         ["connectedComponentsWithStats,", "then two physical tests per body:", "", "dense: at least 92% decorrelated", "smooth: mean local std at most 14", "", "Sunlit gravel under a moving", "shadow fails both tests."])
    wl = a.copy(); wl[mask > 0] = (0.5 * wl[mask > 0] + np.array([235, 140, 40]) * 0.5).astype(np.uint8)
    rows = np.where(prof.channel.any(axis=1))[0]
    y = int(rows.max() - h * (rows.max() - rows.min()))
    for x in range(wl.shape[1]):
        if tops[x] >= 0:
            cv2.circle(wl, (x, int(tops[x])), 1, (0, 220, 255), -1)
    cv2.line(wl, (0, y), (wl.shape[1], y), (0, 220, 255), 3)
    step("p4_waterline", wl, "4. Waterline: a virtual staff gauge",
         ["Top edge of the water body in each", "column (dots), median across the", "channel (line).", "", f"waterline height {h:.2f}", f"calibrated stage {stage:.2f} ft", "", "Official gage at this time: 6.43 ft"])

    # calibration plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = json.loads((ROOT / "data/eval" / CAM / "frames_scored.json").read_text())
    pts = [(r["waterline"], r["truth"]) for r in fr if r["usable"] and r["waterline"] is not None and r["truth"]]
    fig, ax = plt.subplots(figsize=(11.2, 8.4), dpi=100)
    fig.patch.set_facecolor("#0e161e"); ax.set_facecolor("#0e161e")
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color="#e8a33d", alpha=.7, label="usable daytime frames")
    xs = np.linspace(0.3, 0.95, 100)
    ax.plot(xs, [vision.stage_from_height(x, prof.calib) for x in xs], color="#5fb2e6", lw=3, label="stage = a + b·exp(c·h)")
    ax.set_xlabel("waterline height h", color="#c9d3db", fontsize=16); ax.set_ylabel("official stage (ft)", color="#c9d3db", fontsize=16)
    ax.tick_params(colors="#c9d3db", labelsize=13)
    for s in ax.spines.values():
        s.set_color("#33424e")
    ax.legend(fontsize=14, facecolor="#0e161e", labelcolor="#c9d3db", edgecolor="#33424e")
    fig.tight_layout(); fig.savefig(OUT / "_calib.png", facecolor=fig.get_facecolor()); plt.close(fig)
    step("p5_calibration", cv2.imread(str(OUT / "_calib.png")), "5. Calibrated to the official gage",
         ["Robust monotone fit; readings far", "above the calibrated range are", "flagged, not extrapolated.", "", "Held out day by day:", "MAE 0.67 ft (20 cm)", "87% within ±1 ft", "crossing alert precision 97%"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    title_card("t_open", "RiverWatch", "OpenCV 5 + Amazon Nova on AWS watch a flooded low-water crossing",
               "OpenCV AI Competition 2026")
    title_card("t_close", "RiverWatch", "Live: d206jwadatehui.cloudfront.net\nCode: github.com/crjstobal/riverwatch",
               "Data: USGS (public domain) · research prototype, not an official warning service")
    timelapse()
    pipeline()


if __name__ == "__main__":
    main()
