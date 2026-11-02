"""Build a synthetic 2-take test project (no real speech) to exercise the render pipeline
without a Whisper model: tone bursts stand in for words, and a matching transcript is
written straight into the transcription cache.

    .venv/bin/python tests/make_fixture.py /path/to/fixture_dir
    ./edit --input /path/to/fixture_dir/input --script /path/to/fixture_dir/script.txt --yes
"""

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from editor.transcribe import CACHE_VERSION, _key  # noqa: E402
from editor.common import BUILD, load_config, ROOT  # noqa: E402

SR = 48000
TAKES = {
    "take1.mov": (1080, 1920, ["Okay.", 1.2, "So today I want to", 1.4, "So today I want to talk about saving money.", 0.4,
                               "um", 0.3, "The the first thing is a budget for every month.", 1.5,
                               "Wait, let me start over.", 1.0]),
    "take2.mov": (1280, 720, ["The first thing you need is a budget for every single month.", 0.8,
                              "Then you automate it.", 0.6, "My spending dropped a lot.", 0.5, "Follow for more."]),
}
SCRIPT = """So today I want to talk about saving money. [TEXT: Save more money]
The first thing you need is a budget for every single month. [CHART: Monthly spend | Jan 820, Feb 640, Mar 410]
[PIC: receipt.png]
Then you automate it.
My spending dropped a lot.
Follow for more. [TEXT: Follow for part 2]
"""


def synth(script, lead=0.8, tail=1.0):
    t, words, bursts = lead, [], []
    for item in script:
        if isinstance(item, float):
            t += item
            continue
        for raw in item.split():
            d = 0.18 + 0.035 * len(raw.strip(".,"))
            words.append({"w": raw, "s": round(t, 3), "e": round(t + d, 3), "p": 0.95, "seg_end": raw.endswith(".")})
            bursts.append((t, d, 160 + 12 * (len(words) % 9)))
            t += d + 0.07
    total = t + tail
    n = int(total * SR)
    rng = np.random.default_rng(1)
    audio = rng.normal(0, 0.0008, n)  # ~ -60 dB room tone
    for s, d, f in bursts:
        i0, m = int(s * SR), int(d * SR)
        tt = np.arange(m) / SR
        env = np.minimum(1, np.minimum(tt / 0.02, (d - tt) / 0.03))
        audio[i0:i0 + m] += 0.25 * env * (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(4 * np.pi * f * tt))
    return words, (np.clip(audio, -1, 1) * 32767).astype(np.int16), total


def main(out_dir: Path):
    cfg = load_config(ROOT / "config.json")
    inp = out_dir / "input"
    inp.mkdir(parents=True, exist_ok=True)
    (BUILD / "cache").mkdir(parents=True, exist_ok=True)
    for name, (w, h, script) in TAKES.items():
        words, pcm, total = synth(script)
        wav = out_dir / (name + ".wav")
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(1), wf.setsampwidth(2), wf.setframerate(SR), wf.writeframes(pcm.tobytes())
        video = inp / name
        draw = ("drawtext=text='%s  %%{pts\\:hms}':fontsize=%d:fontcolor=white:box=1:boxcolor=black@0.6:"
                "x=(w-tw)/2:y=h*0.45" % (name, int(h / 22)))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=30:duration={total:.3f}",
                        "-i", str(wav), "-vf", draw, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(video)], check=True)
        wav.unlink()
        key = _key(video, cfg["whisper_model"], cfg.get("language", "en"))
        (BUILD / "cache" / f"{video.stem}.{key}.words.json").write_text(json.dumps(words))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x2E7D32:s=700x500",
                    "-vf", "drawtext=text='RECEIPT':fontsize=90:fontcolor=white:x=(w-tw)/2:y=(h-th)/2",
                    "-frames:v", "1", str(inp / "receipt.png")], check=True)
    (out_dir / "script.txt").write_text(SCRIPT)
    print(f"fixture ready in {out_dir} (cache keys for model {cfg['whisper_model']}, version {CACHE_VERSION})")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
