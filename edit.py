#!/usr/bin/env python3
"""Talking-head editor: input/ (raw takes) + script.txt  ->  output/<name>.mp4 (1080x1920).

Run it through ./edit, which installs anything missing first.

  ./edit                 analyze, show the cut list, ask before rendering
  ./edit --yes           render without asking (uses your edits to build/cutlist.txt)
  ./edit --analyze-only  just transcribe + show the cut list
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from editor import captions, cutlist, popups, render
from editor.analyze import analyze
from editor.common import BUILD, ROOT, check_tools, find_takes, fmt_time, load_config, media_duration
from editor.transcribe import audio_for, transcribe


def step(msg):
    print(f"\n\033[1m==> {msg}\033[0m" if sys.stdout.isatty() else f"\n==> {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="input", help="folder of raw takes (default: input/)")
    ap.add_argument("--script", default="script.txt", help="script with [TEXT:] [PIC:] [CHART:] notes")
    ap.add_argument("--output", default="output", help="where the finished MP4 goes")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--name", help="output file name (default: edit_<date>.mp4)")
    ap.add_argument("--model", help="faster-whisper model, e.g. small.en, medium.en, large-v3")
    ap.add_argument("-y", "--yes", action="store_true", help="skip the approval prompt")
    ap.add_argument("--analyze-only", action="store_true", help="stop after showing the cut list")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--no-popups", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    check_tools()
    cfg = load_config(ROOT / args.config)
    if args.model:
        cfg["whisper_model"] = args.model
    in_dir, out_dir = (ROOT / args.input), (ROOT / args.output)
    script = ROOT / args.script
    BUILD.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    takes = find_takes(in_dir) if in_dir.exists() else []
    if not takes:
        sys.exit(f"No video takes found in {in_dir}/. Drop your clips there (they're used in name order).")

    # 1. Transcribe -----------------------------------------------------------
    step(f"Transcribing {len(takes)} take(s) with faster-whisper ({cfg['whisper_model']})")
    words_by_take, wavs = [], []
    for p in takes:
        print(f"  {p.name}  ({fmt_time(media_duration(p))})", flush=True)
        wavs.append(audio_for(p))
        words_by_take.append(transcribe(p, cfg["whisper_model"], cfg.get("language", "en")))

    # 2. Find cuts -------------------------------------------------------------
    step("Finding dead air, fillers, stutters and retakes")
    result = analyze(takes, cfg, words_by_take, wavs)
    prof = result["profile"]
    (BUILD / "voice_profile.json").write_text(json.dumps(prof, indent=2))
    print(f"  voice: {prof['words_per_minute']} wpm, natural pauses up to {prof['natural_pause_p90']:.2f}s, "
          f"noise floor {prof['noise_floor_db']} dB, speech {prof['speech_level_db']} dB")
    print(f"  tuned: cut pauses longer than {prof['max_gap']:.2f}s, silence below {prof['silence_threshold_db']} dB")

    items = cutlist.build_items(result)
    aid = cutlist.analysis_id(items, result["takes"])
    cl_path = BUILD / "cutlist.txt"
    ok = cutlist.approve(items, result["takes"], cl_path, aid, assume_yes=args.yes and not args.analyze_only)
    if args.analyze_only or not ok:
        return 0 if args.analyze_only else 2

    segments = cutlist.final_segments(items)
    if not segments:
        sys.exit("Nothing left to keep. Check build/cutlist.txt.")

    # 3. Cut + reframe -----------------------------------------------------------
    fps = cfg["video"]["fps"]
    plan, total = render.plan_timeline(segments, fps)
    step(f"Cutting {len(plan)} segments and reframing to {cfg['video']['width']}x{cfg['video']['height']}")
    cut = render.cut_segments(plan, result["takes"], cfg, BUILD)

    out_words = []
    for w in result["words"]:
        if w.cut in ("filler", "noise") or w.n in {f.lower() for f in cfg["cut"]["remove_fillers"]}:
            continue
        t = render.to_output_time(plan, w.take, (w.s + w.e) / 2)
        if t is None:
            continue
        off = t - (w.s + w.e) / 2
        out_words.append({"text": w.w, "start": round(w.s + off, 3), "end": round(w.e + off, 3)})
    out_words.sort(key=lambda x: x["start"])
    (BUILD / "transcript.json").write_text(json.dumps(out_words, indent=1))
    (BUILD / "transcript.txt").write_text(" ".join(w["text"] for w in out_words) + "\n")

    # 4. Captions -----------------------------------------------------------------
    ass = None
    if cfg["captions"]["enabled"] and not args.no_captions:
        step("Writing captions")
        ass = captions.write_ass(out_words, cfg, BUILD / "captions.ass", total)

    # 5. Popups (HyperFrames) -------------------------------------------------------
    overlay, offset = None, 0.0
    if script.exists() and not args.no_popups:
        cues = popups.parse_script(script)
        if cues:
            step(f"Timing {len(cues)} popup(s) from {script.name} to the transcript")
            timed = popups.time_cues(cues, out_words, cfg, total)
            for c in timed:
                print(f"  {fmt_time(c['start'])}  {c['kind']:<5} {c['arg'][:50]:<50}  (line {c['line_no']}, match {c['match']:.0%})")
            (BUILD / "popups.json").write_text(json.dumps(timed, indent=1))
            if timed:
                proj = BUILD / "popups"
                offset, dur = popups.build_project(timed, cfg, [in_dir, script.parent, ROOT / "assets"], proj)
                step(f"Rendering popups with HyperFrames ({dur:.1f}s overlay)")
                overlay = popups.render_overlay(proj, BUILD / "overlay.mov", fps)
    elif not script.exists():
        print(f"\n  (no {script.name}, so no popups)")

    # 6. Final -----------------------------------------------------------------------
    name = args.name or f"edit_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    if not name.endswith(".mp4"):
        name += ".mp4"
    out = out_dir / name
    step("Compositing final video")
    render.composite(cut, out, cfg, ass=ass, fonts_dir=captions.fonts_dir() if ass else None,
                     overlay=overlay, overlay_offset=offset)
    shutil.rmtree(BUILD / "segments", ignore_errors=True)

    print(f"\nDone in {time.time() - t0:.0f}s: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}  ({fmt_time(media_duration(out))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
