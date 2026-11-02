"""FFmpeg: cut kept segments, reframe to vertical, then composite popups + captions."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import ffmpeg_has_filter, is_hdr, run

FADE = 0.012  # 12 ms audio fade at every cut so joins don't click


def video_filter(src: Path, W: int, H: int, fps: int) -> str:
    chain = []
    if is_hdr(src) and ffmpeg_has_filter("zscale") and ffmpeg_has_filter("tonemap"):
        # iPhone HDR (HLG / Dolby Vision) -> SDR, otherwise it looks washed out on TikTok
        chain.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                     "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv")
    chain += [
        f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={W}:{H}",
        f"fps={fps}",
        "setsar=1",
        "format=yuv420p",
    ]
    return ",".join(chain)


def plan_timeline(segments, fps: int):
    """Snap each segment to whole frames and give it an output start time."""
    out_t = 0.0
    plan = []
    for s in segments:
        frames = max(1, round((s["end"] - s["start"]) * fps))
        dur = frames / fps
        plan.append({**s, "end": s["start"] + dur, "frames": frames, "dur": dur, "out_start": out_t})
        out_t += dur
    return plan, out_t


def to_output_time(plan, take: int, t: float):
    for p in plan:
        if p["take"] == take and p["start"] <= t < p["end"]:
            return p["out_start"] + (t - p["start"])
    return None


def cut_segments(plan, takes, cfg, workdir: Path) -> Path:
    v = cfg["video"]
    W, H, fps = v["width"], v["height"], v["fps"]
    seg_dir = workdir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    vf_cache = {}

    def one(k_p):
        k, p = k_p
        src = takes[p["take"]].path
        if src not in vf_cache:
            vf_cache[src] = video_filter(src, W, H, fps)
        out = seg_dir / f"seg_{k:04d}.mov"
        af = (f"aresample=48000,aformat=channel_layouts=stereo,"
              f"afade=t=in:d={FADE},afade=t=out:st={max(0.0, p['dur'] - FADE):.4f}:d={FADE},"
              f"apad,atrim=0:{p['dur']:.6f}")
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{p['start']:.4f}", "-i", src,
             "-t", f"{p['dur'] + 0.5:.4f}", "-map", "0:v:0", "-map", "0:a:0",
             "-vf", vf_cache[src], "-frames:v", str(p["frames"]), "-af", af,
             "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", "-g", str(fps),
             "-c:a", "pcm_s16le", out])
        return out

    workers = max(1, min(4, (os.cpu_count() or 2) // 2))
    for src in {takes[p["take"]].path for p in plan}:  # warm the filter cache single-threaded
        vf_cache[src] = video_filter(src, W, H, fps)
    with ThreadPoolExecutor(workers) as ex:
        files = list(ex.map(one, enumerate(plan)))

    lst = workdir / "segments.txt"
    lst.write_text("".join(f"file '{f.resolve()}'\n" for f in files))
    cut = workdir / "cut.mov"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", cut])
    return cut


def _ass_path_for_filter(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    return s


def composite(cut: Path, out: Path, cfg, ass: Path = None, fonts_dir: Path = None,
              overlay: Path = None, overlay_offset: float = 0.0):
    v = cfg["video"]
    inputs = ["-i", cut]
    graph = []
    last = "[0:v]"
    if overlay:
        inputs += ["-i", overlay]
        graph.append(f"[1:v]format=yuva420p,setpts=PTS-STARTPTS+{overlay_offset:.4f}/TB[ov]")
        graph.append(f"{last}[ov]overlay=0:0:eof_action=pass:format=auto[v1]")
        last = "[v1]"
    if ass:
        sub = f"subtitles=filename='{_ass_path_for_filter(ass)}'"
        if fonts_dir:
            sub += f":fontsdir='{_ass_path_for_filter(fonts_dir)}'"
        graph.append(f"{last}{sub}[v2]")
        last = "[v2]"
    graph.append(f"{last}format=yuv420p[vout]")
    af = f"loudnorm=I={v['loudness_lufs']}:TP=-1.5:LRA=11,aresample=48000" if v.get("loudness_lufs") else "anull"
    graph.append(f"[0:a]{af}[aout]")
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(graph),
         "-map", "[vout]", "-map", "[aout]",
         "-c:v", "libx264", "-crf", str(v["crf"]), "-preset", "medium", "-profile:v", "high",
         "-pix_fmt", "yuv420p", "-r", str(v["fps"]),
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out])
    return out
