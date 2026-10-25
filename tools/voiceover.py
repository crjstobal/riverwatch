"""Narration for the demo video with ElevenLabs, one file per scene, with character timestamps for the subtitles.

The API key is read at run time from AWS SSM (never stored on disk).
usage: voiceover.py [voice_id]      writes docs/video/audio/<scene>.mp3 and <scene>.json (alignment)
"""
import base64
import json
import sys
import urllib.request
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_video import SEGMENTS  # noqa: E402

VOICE = sys.argv[1] if len(sys.argv) > 1 else "SSfU0eLfP3qeuR4j2bwD"  # Christopher Smooth: warm, professional
MODEL = "eleven_multilingual_v2"
OUT = ROOT / "docs/video/audio"


def api_key():
    sess = boto3.Session(profile_name="servikaki-secrets", region_name="eu-west-1")
    raw = sess.client("ssm").get_parameter(Name="/servikaki/secrets/elevenlabs.env", WithDecryption=True)["Parameter"]["Value"]
    env = dict(l.split("=", 1) for l in raw.splitlines() if "=" in l and not l.strip().startswith("#"))
    return next(v.strip().strip('"').strip("'") for k, v in env.items() if "KEY" in k)


def main():
    key = api_key()
    OUT.mkdir(parents=True, exist_ok=True)
    texts = [s for s in SEGMENTS if s["text"]]
    for i, seg in enumerate(texts):
        body = {"text": seg["text"], "model_id": MODEL,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.15, "use_speaker_boost": True}}
        # continuity between scenes: tell the model what comes before and after
        if i > 0:
            body["previous_text"] = texts[i - 1]["text"]
        if i < len(texts) - 1:
            body["next_text"] = texts[i + 1]["text"]
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}/with-timestamps?output_format=mp3_44100_192",
            data=json.dumps(body).encode(), headers={"xi-api-key": key, "content-type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=120))
        (OUT / f"{seg['id']}.mp3").write_bytes(base64.b64decode(r["audio_base64"]))
        (OUT / f"{seg['id']}.json").write_text(json.dumps(r["alignment"]))
        print("ok", seg["id"])


if __name__ == "__main__":
    main()
