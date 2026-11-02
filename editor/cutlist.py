"""The cut list: what gets kept and cut, shown for approval and editable as text."""

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .common import fmt_time

HEADER = """# CUT LIST  (id: {aid})
# Every line is a stretch of one take. KEEP = in the video, CUT = removed.
# To override: change KEEP to CUT (or CUT to KEEP), save, then run ./edit --yes
# Times are positions inside each original take.
"""
LINE_RE = re.compile(r"^\s*(KEEP|CUT)\s+#(\d+)", re.I)


def _mid(w):
    return (w.s + w.e) / 2


def _describe_cut(words, start, end, take):
    inside = [w for w in words if start <= _mid(w) < end]
    cut = [w for w in inside if w.cut]
    if not cut:
        dur = end - start
        sil = sum(max(0.0, min(end, s.end) - max(start, s.start)) for s in take.silences)
        label = "dead air"
        if start <= 0.01:
            label = "dead air (start of take)"
        elif end >= take.duration - 0.01:
            label = "dead air (end of take)"
        if dur > 1.0 and sil / dur < 0.5:
            label += "  [has sound but no words, check]"
        return label
    groups, cur = [], None
    for w in cut:
        if cur and cur[0] == w.cut and cur[1] == w.note:
            cur[2].append(w.w)
        else:
            cur = [w.cut, w.note, [w.w]]
            groups.append(cur)
    parts = []
    for kind, note, ws in groups:
        said = " ".join(ws)
        if kind in ("filler", "stutter", "noise"):
            parts.append(f'{kind}: "{said}"')
        else:
            parts.append(f'{kind}: "{said[:60]}{"..." if len(said) > 60 else ""}" ({note})')
    return "; ".join(parts)


def build_items(result) -> list:
    takes, words = result["takes"], result["words"]
    items = []
    for t in takes:
        segs = [s for s in result["segments"] if s["take"] == t.idx]
        cursor = 0.0
        tw = [w for w in words if w.take == t.idx]
        for s in segs:
            if s["start"] - cursor > 0.05:
                items.append({"take": t.idx, "start": cursor, "end": s["start"], "action": "CUT",
                              "why": _describe_cut(tw, cursor, s["start"], t)})
            text = " ".join(w.w for w in tw if s["start"] <= _mid(w) < s["end"] and w.cut not in ("filler",))
            items.append({"take": t.idx, "start": s["start"], "end": s["end"], "action": "KEEP", "why": text})
            cursor = s["end"]
        if t.duration - cursor > 0.05:
            items.append({"take": t.idx, "start": cursor, "end": t.duration, "action": "CUT",
                          "why": _describe_cut(tw, cursor, t.duration, t)})
    for k, it in enumerate(items, 1):
        it["id"] = k
        it["auto"] = it["action"]
    return items


def analysis_id(items, takes) -> str:
    raw = "|".join(f"{takes[i['take']].path.name}:{i['start']:.3f}:{i['end']:.3f}:{i['auto']}" for i in items)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def format_items(items, takes, color=False) -> str:
    G, R, D, X = ("\033[32m", "\033[31m", "\033[2m", "\033[0m") if color else ("", "", "", "")
    lines = []
    for it in items:
        name = takes[it["take"]].path.name
        span = f'{fmt_time(it["start"])} -> {fmt_time(it["end"])}  ({it["end"] - it["start"]:4.1f}s)'
        flag = "" if it["action"] == it["auto"] else "  (your change)"
        if it["action"] == "KEEP":
            why = it["why"]
            why = f'"{why[:90]}{"..." if len(why) > 90 else ""}"'
            lines.append(f'{G}KEEP  #{it["id"]:03d}{X}  {name:<16} {span}  {why}{flag}')
        else:
            lines.append(f'{R}CUT   #{it["id"]:03d}{X}  {D}{name:<16} {span}  {it["why"]}{X}{flag}')
    return "\n".join(lines)


def summary(items, takes) -> str:
    total = sum(t.duration for t in takes)
    kept = sum(i["end"] - i["start"] for i in items if i["action"] == "KEEP")
    n_keep = sum(1 for i in items if i["action"] == "KEEP")
    pct = 100 * (1 - kept / total) if total else 0
    return (f"{len(takes)} take(s), {fmt_time(total)} of footage -> {fmt_time(kept)} final "
            f"({pct:.0f}% removed, {n_keep} kept segments)")


def write_file(path: Path, items, takes, aid):
    path.write_text(HEADER.format(aid=aid) + format_items(items, takes) + "\n\n# " + summary(items, takes) + "\n")


def read_overrides(path: Path, aid: str) -> dict:
    """{item_id: 'KEEP'|'CUT'} from an edited cutlist.txt, if it belongs to this analysis."""
    if not path.exists():
        return {}
    text = path.read_text()
    m = re.search(r"\(id: (\w+)\)", text)
    if not m or m.group(1) != aid:
        return {}
    out = {}
    for line in text.splitlines():
        lm = LINE_RE.match(line)
        if lm:
            out[int(lm.group(2))] = lm.group(1).upper()
    return out


def apply_overrides(items, overrides):
    for it in items:
        if it["id"] in overrides:
            it["action"] = overrides[it["id"]]
    return items


def final_segments(items) -> list:
    """Merge adjacent KEEP items into continuous segments per take."""
    segs = []
    for it in items:
        if it["action"] != "KEEP":
            continue
        if segs and segs[-1]["take"] == it["take"] and it["start"] - segs[-1]["end"] < 0.01:
            segs[-1]["end"] = it["end"]
        else:
            segs.append({"take": it["take"], "start": it["start"], "end": it["end"]})
    return segs


def approve(items, takes, path: Path, aid: str, assume_yes: bool) -> bool:
    """Show the cut list and wait for a yes. Returns True to render."""
    apply_overrides(items, read_overrides(path, aid))
    write_file(path, items, takes, aid)
    tty = sys.stdin.isatty() and sys.stdout.isatty()
    while True:
        print("\n" + format_items(items, takes, color=sys.stdout.isatty()))
        print("\n  " + summary(items, takes))
        print(f"  Full list saved to {path}")
        if assume_yes:
            return True
        if not tty:
            print("\nNot running in an interactive terminal. Review the list above (or edit "
                  f"{path.name}), then run again with --yes to render.")
            return False
        ans = input("\nApprove these cuts?  [y] yes, render  /  [e] edit the list  /  [n] stop: ").strip().lower()
        if ans in ("y", "yes", ""):
            return True
        if ans in ("n", "no", "q"):
            return False
        if ans in ("e", "edit"):
            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or next(
                (e for e in ("nano", "vim", "vi") if shutil.which(e)), None)
            if editor:
                subprocess.call([*editor.split(), str(path)])
            else:
                input(f"Edit {path} in any text editor, save it, then press Enter...")
            apply_overrides(items, read_overrides(path, aid))
            write_file(path, items, takes, aid)
