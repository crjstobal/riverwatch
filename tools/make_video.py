"""Assemble the demo video from the clips and stills, with burned-in subtitles.

  make_video.py draft     durations from word count, no audio  -> docs/video/riverwatch_draft.mp4
  make_video.py final     durations from docs/video/audio/<id>.mp3 (ElevenLabs), with audio -> riverwatch.mp4

The ffmpeg build on this Mac has no libass/drawtext, so subtitles are rendered as PNG overlays with Pillow.
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "docs/video"
A = V / "assets"
C = ROOT / "docs/clips"
W, H, FPS = 1920, 1080, 30
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
WPS = 2.5  # words per second for the draft (about 150 wpm, close to the ElevenLabs pace)
FADE = 0.6  # cross-dissolve between scenes, seconds
TAIL = 1.0  # silence after each narration block (the next dissolve eats part of it)

SEGMENTS = [
    {"id": "s0_title", "visual": {"type": "image", "src": "t_open.png"}, "text": "", "min": 4},
    {"id": "s1_hook", "visual": {"type": "seq", "src": "timelapse", "fps": 4},
     "text": "On October second, this low-water crossing near Fort Hood, Texas, went from dry to more than seven feet "
             "under water in about six hours. Over half of flood drownings happen when someone drives into water like "
             "this. RiverWatch watches crossings like this one, through cameras that already exist."},
    {"id": "s2_problem", "visual": {"type": "clip", "src": "1_hero_live.webm", "start": 0},
     "text": "The U.S. Geological Survey runs more than thirteen hundred river cameras, and nobody measures their images. "
             "Gages fail exactly when they matter. Right now, the official gage at this crossing has a silent primary "
             "sensor and a backup stuck at six point two feet, while the camera shows a dry creek."},
    {"id": "s3_opencv", "visual": {"type": "cued", "src": [["p1_quality.png", None], ["p2_ncc.png", "A dense normalized"],
                                                            ["p3_bodies.png", "Each water body"],
                                                            ["p4_waterline.png", "The top edge"],
                                                            ["p5_calibration.png", "calibrated against"]]},
     "text": "Everything visual is classical OpenCV 5, with no trained network. A quality gate rejects infrared night "
             "frames, blur and fog. ECC registration aligns each frame using only the bridge and the banks. A dense "
             "normalized cross-correlation between frames fifteen minutes apart shows what moved: flowing water changes "
             "everywhere, concrete does not. Each water body must be dense and smooth, which rejects sunlit gravel under "
             "a moving shadow. The top edge of the water across the channel becomes a virtual staff gauge, calibrated "
             "against the official gage."},
    {"id": "s4_results", "visual": {"type": "clip", "src": "3_evaluation.webm", "start": 0},
     "text": "We evaluated on almost four thousand frames with cross-validation by day, so the flood day was never used "
             "for calibration. When water flows, the camera is within twenty centimetres on average, and eighty-seven "
             "percent of readings are within one foot. For the question that matters, is the crossing under a foot of "
             "water, precision is ninety-seven percent."},
    {"id": "s4b_forecast", "visual": {"type": "cued", "src": [["f1_oct2.png", None], ["f2_leads.png", "Tested year by year"]]},
     "text": "RiverWatch also looks ahead. A small forecast, trained on nine and a half years of both gages, predicts "
             "whether the crossing will close within the next hour. On October second, the upstream gage jumped at ten "
             "in the morning, and the warning fired twenty-five minutes before the water covered the crossing. Tested "
             "year by year on data it never saw, it warned before thirty-two of forty-six closures, most of them at "
             "least half an hour ahead."},
    {"id": "s5_agent", "visual": {"type": "cued", "src": [["@replay_02.png", None], ["@replay_08.png", "At ten twenty-five"],
                                                           ["@replay_09.png", "Fifteen minutes later"],
                                                           ["@replay_10.png", "Nothing reaches the public"],
                                                           ["@replay_12.png", "When the camera misses"]]},
     "text": "An Amazon Nova agent on Bedrock decides what to do next. This is a replay of October second on a simulated "
             "clock. At nine in the morning the forecast warns, and the agent sends the duty operator an early caution, "
             "an hour and a half before the water arrives. At ten twenty-five the water covers the crossing. The agent "
             "asks to close it, but a guardrail in code blocks the request: escalation needs two independent readings. "
             "Fifteen minutes later, with camera and gage agreeing, it asks the operator to close the crossing. Nothing "
             "reaches the public without a human. When the camera misses water, the agent looks at the frame itself and "
             "flags the camera."},
    {"id": "s6_aws", "visual": {"type": "clips", "src": [["4_architecture.webm", 1.2], ["1_hero_live.webm", 5.0]]},
     "text": "It runs live on AWS. A Lambda function on Graviton runs OpenCV and the agent every fifteen minutes, S3 and "
             "CloudFront serve this page, and SNS notifies the operator. Today it reports the crossing as dry and the "
             "official gage as broken, instead of raising a false alarm."},
    {"id": "s7_close", "visual": {"type": "image", "src": "t_close.png"},
     "text": "It works on one camera so far, in daylight, with a quick calibration per site. The limits are in the "
             "report. But the cameras are already there. RiverWatch turns them into a second pair of eyes for every "
             "crossing."},
]


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def chunks(text, max_words=10):
    """Split narration into subtitle lines at sentence or clause boundaries."""
    words, out, cur = text.split(), [], []
    for w in words:
        cur.append(w)
        end = w[-1] in ".:;,?"
        if (end and len(cur) >= 4) or len(cur) >= max_words or (w[-1] in ".?:" and len(cur) >= 2):
            out.append(" ".join(cur)); cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def sub_png(text, path):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 46)
    box = d.textbbox((0, 0), text, font=f)
    tw, th = box[2] - box[0], box[3] - box[1]
    x, y = (W - tw) // 2, H - 120 - th
    d.rounded_rectangle((x - 28, y - 20, x + tw + 28, y + th + 26), radius=14, fill=(0, 0, 0, 175))
    d.text((x, y - box[1]), text, font=f, fill=(255, 255, 255, 255))
    im.save(path)


def cue_times(seg, mode, speech, lead=0.3):
    """Start time of each cued image: when its phrase is spoken (alignment) or by word share (draft)."""
    text, out = seg["text"], []
    align_f = V / "audio" / f"{seg['id']}.json"
    align = json.loads(align_f.read_text()) if mode == "final" and align_f.exists() else None
    for _, cue in seg["visual"]["src"]:
        if not cue:
            out.append(0.0); continue
        a = text.find(cue)
        if align:
            out.append(lead + align["character_start_times_seconds"][a] - 0.15)
        else:
            out.append(lead + speech * len(text[:a].split()) / len(text.split()))
    return out


def visual_input(vis, dur, tmp):
    """Return ffmpeg input args and a filter chain producing [v] of exactly `dur` seconds."""
    norm = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0e161e,fps={FPS},format=yuv420p"
    t = vis["type"]
    if t == "image":
        return ["-loop", "1", "-t", f"{dur:.3f}", "-i", str(A / vis["src"])], f"[0:v]{norm}[v]"
    if t == "seq":
        n = len(list((A / vis["src"]).glob("*.png")))
        return (["-framerate", str(vis["fps"]), "-i", str(A / vis["src"] / "%03d.png")],
                f"[0:v]{norm},tpad=stop_mode=clone:stop_duration={max(0, dur - n / vis['fps']):.3f},trim=duration={dur:.3f}[v]")
    if t == "clip":
        return (["-ss", str(vis.get("start", 0)), "-i", str(C / vis["src"])],
                f"[0:v]{norm},tpad=stop_mode=clone:stop_duration={dur:.3f},trim=duration={dur:.3f}[v]")
    if t == "images":
        each = dur / len(vis["src"])
        args, parts = [], []
        for i, s in enumerate(vis["src"]):
            args += ["-loop", "1", "-t", f"{each:.3f}", "-i", str(A / s)]
            parts.append(f"[{i}:v]{norm}[i{i}]")
        return args, ";".join(parts) + ";" + "".join(f"[i{i}]" for i in range(len(vis["src"]))) + f"concat=n={len(vis['src'])}:v=1:a=0[v]"
    if t == "cued":
        times = vis["_times"] + [dur]
        args, parts = [], []
        for i, (s, _) in enumerate(vis["src"]):
            path = C / s[1:] if s.startswith("@") else A / s
            args += ["-loop", "1", "-t", f"{times[i + 1] - times[i]:.3f}", "-i", str(path)]
            parts.append(f"[{i}:v]{norm}[i{i}]")
        return args, ";".join(parts) + ";" + "".join(f"[i{i}]" for i in range(len(vis["src"]))) + f"concat=n={len(vis['src'])}:v=1:a=0[v]"
    if t == "clips":
        each = dur / len(vis["src"])
        args, parts = [], []
        for i, (s, st) in enumerate(vis["src"]):
            args += ["-ss", str(st), "-i", str(C / s)]
            parts.append(f"[{i}:v]{norm},tpad=stop_mode=clone:stop_duration={each:.3f},trim=duration={each:.3f},setpts=PTS-STARTPTS[i{i}]")
        return args, ";".join(parts) + ";" + "".join(f"[i{i}]" for i in range(len(vis["src"]))) + f"concat=n={len(vis['src'])}:v=1:a=0[v]"
    raise ValueError(t)


def build(mode):
    tmp = V / "build"; tmp.mkdir(parents=True, exist_ok=True)
    seg_files, srt, clock = [], [], 0.0
    for k, seg in enumerate(SEGMENTS):
        words = len(seg["text"].split())
        audio = V / "audio" / f"{seg['id']}.mp3"
        if mode == "final" and seg["text"]:
            speech = probe(audio)
            dur = max(seg.get("min", 0), 0.3 + speech + TAIL)
        else:
            speech = words / WPS
            dur = max(seg.get("min", 0), 0.3 + speech + TAIL)
        if seg["visual"]["type"] == "cued":
            seg["visual"]["_times"] = cue_times(seg, mode, speech)
        args, chain = visual_input(seg["visual"], dur, tmp)
        n_in = args.count("-i")
        # subtitles: proportional to words inside the spoken span
        subs = chunks(seg["text"]) if seg["text"] else []
        t, lead = 0.3, 0.3
        span = speech
        ov_args, ov_chain, last = [], "", "[v]"
        total = sum(len(s.split()) for s in subs) or 1
        align = None
        if mode == "final" and seg["text"] and (V / "audio" / f"{seg['id']}.json").exists():
            align = json.loads((V / "audio" / f"{seg['id']}.json").read_text())
        pos = 0
        for j, s in enumerate(subs):
            d = span * len(s.split()) / total
            if align:  # exact timing from the ElevenLabs character alignment
                a = seg["text"].find(s, pos); b = a + len(s) - 1; pos = b + 1
                st, en = align["character_start_times_seconds"], align["character_end_times_seconds"]
                t = lead + st[min(a, len(st) - 1)]
                nxt = st[min(b + 2, len(st) - 1)] if j < len(subs) - 1 else en[min(b, len(en) - 1)] + 0.4
                d = lead + nxt - t
            png = tmp / f"{seg['id']}_{j:02d}.png"
            sub_png(s, png)
            ov_args += ["-i", str(png)]
            idx = n_in + j
            out = f"[o{j}]"
            ov_chain += f";{last}[{idx}:v]overlay=0:0:enable='between(t,{t:.2f},{t + d:.2f})'{out}"
            srt.append((k, t, t + d, s))
            last, t = out, t + d
        if mode == "final" and seg["text"]:
            ov_args += ["-i", str(audio)]
            a_idx = n_in + len(subs)
            audio_map = ["-map", f"{a_idx}:a", "-af", f"adelay={int(lead * 1000)}:all=1,apad", "-c:a", "aac", "-b:a", "160k"]
        else:
            audio_map = ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo"] if mode == "final" else []
            if mode == "final":
                audio_map += ["-map", f"{n_in + len(subs)}:a", "-c:a", "aac"]
        outf = tmp / f"{k:02d}_{seg['id']}.mp4"
        cmd = ["ffmpeg", "-y", *args, *ov_args]
        if mode == "final" and not seg["text"]:
            cmd += audio_map[:6]
            audio_map = audio_map[6:]
        cmd += ["-filter_complex", chain + ov_chain, "-map", last, *audio_map,
                "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(outf)]
        run(cmd)
        seg_files.append(outf)
        clock += dur
        print(f"{seg['id']:12s} {dur:6.1f}s  ({words} words)")
    # chain the scenes with cross-dissolves (video xfade, audio acrossfade)
    final = V / ("riverwatch_draft.mp4" if mode == "draft" else "riverwatch.mp4")
    durs = [probe(f) for f in seg_files]
    ins = sum((["-i", str(f)] for f in seg_files), [])
    fv, fa, lastv, lasta, off = [], [], "[0:v]", "[0:a]", 0.0
    for i in range(1, len(seg_files)):
        off += durs[i - 1] - FADE
        fv.append(f"{lastv}[{i}:v]xfade=transition=fade:duration={FADE}:offset={off:.3f}[v{i}]")
        fa.append(f"{lasta}[{i}:a]acrossfade=d={FADE}[a{i}]")
        lastv, lasta = f"[v{i}]", f"[a{i}]"
    graph = ";".join(fv + (fa if mode == "final" else []))
    if mode == "final":  # YouTube-level loudness
        graph += f";{lasta}loudnorm=I=-15:TP=-1.5:LRA=11[aout]"
        lasta = "[aout]"
    maps = ["-map", lastv] + (["-map", lasta, "-c:a", "aac", "-b:a", "192k", "-ar", "48000"] if mode == "final" else [])
    run(["ffmpeg", "-y", *ins, "-filter_complex", graph, *maps, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)])
    # subtitle times: scene k starts at sum of the previous durations minus one dissolve per transition
    starts = [sum(durs[:k]) - k * FADE for k in range(len(durs))]
    srt = [(starts[k] + a, starts[k] + b, t) for k, a, b, t in srt]
    clock = sum(durs) - FADE * (len(durs) - 1)
    fmt = lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(s * 1000 % 1000):03d}"
    (final.with_suffix(".srt")).write_text("".join(f"{i + 1}\n{fmt(a)} --> {fmt(b)}\n{t}\n\n" for i, (a, b, t) in enumerate(srt)))
    print(final, f"{clock:.1f}s")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "draft")
