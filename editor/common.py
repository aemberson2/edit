"""Shared helpers: paths, config, ffmpeg wrappers, and the Smart-Cut import."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
SMART_CUT_DIR = ROOT / "vendor" / "Smart-Cut"

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp", ".hevc"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def load_smart_cut():
    """Import qkirara/Smart-Cut's smart_cut.py from vendor/."""
    if not (SMART_CUT_DIR / "smart_cut.py").exists():
        sys.exit("Smart-Cut is missing. Run ./edit once (it clones it into vendor/Smart-Cut).")
    if str(SMART_CUT_DIR) not in sys.path:
        sys.path.insert(0, str(SMART_CUT_DIR))
    import smart_cut  # noqa: E402
    return smart_cut


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    defaults_path = ROOT / "config.json"
    cfg = json.loads(defaults_path.read_text()) if defaults_path.exists() else {}
    if path and path.exists() and path.resolve() != defaults_path.resolve():
        cfg = deep_merge(cfg, json.loads(path.read_text()))
    return cfg


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def find_takes(input_dir: Path):
    takes = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith(".")]
    return sorted(takes, key=natural_key)


def run(cmd, quiet=True, check=True):
    res = subprocess.run([str(c) for c in cmd], capture_output=quiet, text=True)
    if check and res.returncode != 0:
        msg = res.stderr[-3000:] if quiet and res.stderr else ""
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(map(str, cmd[:6]))} ...\n{msg}")
    return res


def ffprobe(path: Path) -> dict:
    res = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    return json.loads(res.stdout)


def media_duration(path: Path) -> float:
    return float(ffprobe(path)["format"]["duration"])


def is_hdr(path: Path) -> bool:
    for s in ffprobe(path).get("streams", []):
        if s.get("codec_type") == "video":
            return s.get("color_transfer") in ("smpte2084", "arib-std-b67")
    return False


_filters_cache = None


def ffmpeg_has_filter(name: str) -> bool:
    global _filters_cache
    if _filters_cache is None:
        _filters_cache = run(["ffmpeg", "-hide_banner", "-filters"]).stdout
    return re.search(rf"\s{re.escape(name)}\s", _filters_cache) is not None


def fmt_time(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:05.2f}"


def check_tools():
    """Step 1: make sure Python, Node and FFmpeg are present (./edit installs them if not)."""
    problems = []
    py = sys.version_info
    if py < (3, 9):
        problems.append(f"Python 3.9+ needed (found {py.major}.{py.minor})")
    for tool in ("ffmpeg", "ffprobe", "node", "npx"):
        if not shutil.which(tool):
            problems.append(f"{tool} not found on PATH")
    if shutil.which("node"):
        v = run(["node", "-v"]).stdout.strip().lstrip("v")
        if int(v.split(".")[0]) < 22:
            problems.append(f"Node 22+ needed for HyperFrames (found {v})")
    if not (ROOT / "node_modules" / "hyperframes").exists():
        problems.append("HyperFrames not installed (run ./edit, or npm install)")
    if problems:
        sys.exit("Setup problems:\n  - " + "\n  - ".join(problems) + "\nRun ./edit to install what's missing.")
