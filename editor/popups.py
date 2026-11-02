"""Script markers -> timed popups, rendered by HyperFrames as a transparent overlay.

script.txt markers:
    [TEXT: Save $500 a month]                      text callout
    [PIC: receipt.jpg]                             picture popup (file in input/ or next to script.txt)
    [CHART: Monthly spend | Jan 820, Feb 640, Mar 410]   simple bar chart (title optional)

A marker belongs to the spoken text on the same line, and pops up when you start
saying that line. A marker alone on its own line belongs to the next line of text.
"""

import html
import json
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

from .common import IMAGE_EXTS, ROOT, run
from .analyze import norm

MARK = re.compile(r"\[(TEXT|PIC|PICTURE|IMG|IMAGE|CHART)\s*:\s*(.*?)\]", re.I)


# ---------------------------------------------------------------------------
# Script parsing + alignment
# ---------------------------------------------------------------------------

def parse_script(path: Path) -> list:
    cues, pending, last_text = [], [], None
    for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        marks = [(m.group(1).upper(), m.group(2).strip()) for m in MARK.finditer(raw)]
        marks = [("PIC" if k in ("PICTURE", "IMG", "IMAGE") else k, a) for k, a in marks]
        text = MARK.sub(" ", raw).strip()
        if text.startswith("#"):
            continue
        if not text:
            pending += [(k, a, no) for k, a in marks]
            continue
        for k, a, n in pending + [(k, a, no) for k, a in marks]:
            cues.append({"kind": k, "arg": a, "line": text, "line_no": n})
        pending = []
        last_text = text
    for k, a, n in pending:  # markers after the last line of text
        if last_text:
            cues.append({"kind": k, "arg": a, "line": last_text, "line_no": n})
    return cues


def script_text(path: Path) -> str:
    return " ".join(MARK.sub(" ", l) for l in path.read_text(encoding="utf-8").splitlines() if not l.strip().startswith("#"))


def align_line(line_toks, out_toks, hint):
    """Best matching window of the spoken transcript for a script line."""
    L, N = len(line_toks), len(out_toks)
    if not L or not N:
        return None
    best = (0.0, None)
    for l in sorted({L, max(1, round(L * 0.8)), round(L * 1.25)}):
        for i in range(0, max(1, N - l + 1)):
            sm = SequenceMatcher(None, line_toks, out_toks[i:i + l], autojunk=False)
            r = sm.ratio() - (0.12 if i < hint else 0.0)
            if r > best[0]:
                blocks = [b for b in sm.get_matching_blocks() if b.size]
                if blocks:
                    best = (r, (i + blocks[0].b, i + blocks[-1].b + blocks[-1].size - 1, sm.ratio()))
    if best[1] and best[1][2] >= 0.5:
        return best[1]
    return None


def time_cues(cues, out_words, cfg, total=None):
    """out_words: [{"text","start","end"}] in output time. Returns timed popups."""
    p = cfg["popups"]
    toks = [norm(w["text"]) for w in out_words]
    timed, hint, cache = [], 0, {}
    for c in cues:
        line_toks = [t for t in (norm(x) for x in c["line"].split()) if t]
        key = c["line"]
        if key not in cache:
            cache[key] = align_line(line_toks, toks, hint)
        hit = cache[key]
        if not hit:
            print(f"  ! couldn't find script line {c['line_no']} in the edit, skipping its {c['kind']} popup:\n"
                  f"      \"{c['line'][:70]}\"")
            continue
        a, b, score = hit
        hint = a
        start = max(0.0, out_words[a]["start"] - 0.1)
        end = out_words[b]["end"] + 0.6
        end = min(max(end, start + p["min_duration"]), start + p["max_duration"])
        if total:
            end = min(end, total - 0.05)
        timed.append({**c, "start": start, "end": end, "match": round(score, 2)})
    timed.sort(key=lambda x: x["start"])
    for k in range(len(timed) - 1):  # one popup at a time
        if timed[k]["end"] > timed[k + 1]["start"] - 0.05:
            timed[k]["end"] = max(timed[k]["start"] + 0.8, timed[k + 1]["start"] - 0.05)
    return timed


# ---------------------------------------------------------------------------
# HyperFrames composition
# ---------------------------------------------------------------------------

def parse_chart(arg: str):
    title, data = (arg.split("|", 1) + [""])[:2] if "|" in arg else ("", arg)
    sep = ";" if ";" in data else ","
    rows = []
    for item in data.split(sep):
        item = item.strip()
        m = re.match(r"^(.*?)[\s:=]+([$€£]?)(-?\d+(?:\.\d+)?)\s*(%|k|K|M|x)?$", item)
        if m and m.group(1).strip():
            rows.append({"label": m.group(1).strip(), "value": float(m.group(3)),
                         "display": f"{m.group(2)}{m.group(3)}{m.group(4) or ''}"})
    return title.strip(), rows


def _text_size(t: str) -> int:
    n = len(t)
    return 96 if n <= 12 else 84 if n <= 22 else 70 if n <= 40 else 58


def build_project(timed, cfg, asset_dirs, proj: Path):
    W, H = cfg["video"]["width"], cfg["video"]["height"]
    p = cfg["popups"]
    if proj.exists():
        shutil.rmtree(proj)
    (proj / "assets").mkdir(parents=True)
    shutil.copy(ROOT / "node_modules" / "gsap" / "dist" / "gsap.min.js", proj / "assets" / "gsap.min.js")

    offset = timed[0]["start"]
    duration = round(timed[-1]["end"] - offset + 0.2, 3)
    zone = {"top": "align-items:flex-start;padding-top:230px;",
            "middle": "align-items:center;",
            "bottom": "align-items:flex-end;padding-bottom:760px;"}.get(p.get("zone", "top"))

    clips, tweens = [], []
    for k, c in enumerate(timed):
        cid = f"pop{k}"
        s, d = round(c["start"] - offset, 3), round(c["end"] - c["start"], 3)
        end = s + d
        if c["kind"] == "PIC":
            src = next((dd / c["arg"] for dd in asset_dirs if (dd / c["arg"]).exists()), None)
            if not src or src.suffix.lower() not in IMAGE_EXTS:
                print(f"  ! picture '{c['arg']}' not found in input/ (line {c['line_no']}), skipping")
                continue
            dst = f"assets/{k}_{re.sub(r'[^A-Za-z0-9._-]', '_', src.name)}"
            shutil.copy(src, proj / dst)
            inner = f'<div id="{cid}-card" class="card pic"><img src="{dst}" alt=""></div>'
            tweens.append(f'tl.fromTo("#{cid}-card",{{opacity:0,scale:0.6,rotation:-4}},'
                          f'{{opacity:1,scale:1,rotation:0,duration:0.42,ease:"back.out(1.7)"}},{s});')
        elif c["kind"] == "CHART":
            title, rows = parse_chart(c["arg"])
            if not rows:
                print(f"  ! couldn't read chart data on line {c['line_no']}: '{c['arg']}'. "
                      f"Use [CHART: Title | Label 10, Label 20]. Showing it as text.")
                c = {**c, "kind": "TEXT"}
            else:
                vmax = max(r["value"] for r in rows) or 1
                bars = "".join(
                    f'<div class="row"><div class="lab">{html.escape(r["label"])}</div>'
                    f'<div class="track"><div id="{cid}-b{j}" class="bar" style="--f:{max(r["value"], 0) / vmax:.4f}"></div></div>'
                    f'<div id="{cid}-v{j}" class="val">{html.escape(r["display"])}</div></div>'
                    for j, r in enumerate(rows))
                ttl = f'<div class="ctitle">{html.escape(title)}</div>' if title else ""
                inner = f'<div id="{cid}-card" class="card chart">{ttl}{bars}</div>'
                tweens.append(f'tl.fromTo("#{cid}-card",{{opacity:0,y:50}},{{opacity:1,y:0,duration:0.35,ease:"power3.out"}},{s});')
                for j, r in enumerate(rows):
                    frac = max(r["value"], 0) / vmax
                    t0 = round(s + 0.25 + 0.12 * j, 3)
                    tweens.append(f'tl.fromTo("#{cid}-b{j}",{{scaleX:0}},{{scaleX:{frac:.4f},duration:0.6,ease:"power3.out"}},{t0});')
                    tweens.append(f'tl.fromTo("#{cid}-v{j}",{{opacity:0}},{{opacity:1,duration:0.25}},{t0 + 0.4:.3f});')
        if c["kind"] == "TEXT":
            txt = html.escape(c["arg"])
            inner = (f'<div id="{cid}-card" class="card text"><div class="t" style="font-size:{_text_size(c["arg"])}px">{txt}</div>'
                     f'<div id="{cid}-bar" class="accent"></div></div>')
            tweens.append(f'tl.fromTo("#{cid}-card",{{opacity:0,y:40,scale:0.9}},'
                          f'{{opacity:1,y:0,scale:1,duration:0.38,ease:"back.out(1.6)"}},{s});')
            tweens.append(f'tl.fromTo("#{cid}-bar",{{scaleX:0}},{{scaleX:1,duration:0.4,ease:"power2.out"}},{s + 0.2:.3f});')
        # exit + hard kill
        tweens.append(f'tl.to("#{cid}-card",{{opacity:0,y:-24,scale:0.96,duration:0.3,ease:"power2.in"}},{max(s + 0.5, end - 0.32):.3f});')
        tweens.append(f'tl.set("#{cid}-card",{{opacity:0}},{end - 0.01:.3f});')
        clips.append(f'<div id="{cid}" class="clip slot" data-start="{s}" data-duration="{d}" '
                     f'data-track-index="{k % 3}" style="z-index:{k + 1}">{inner}</div>')

    doc = f"""<!doctype html>
<html lang="en" data-resolution="portrait">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width={W}, height={H}" />
<script src="assets/gsap.min.js"></script>
<style>
  :root {{ --accent: {p['accent_color']}; --card: {p['card_color']}; --ink: {p['text_color']}; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {W}px; height: {H}px; overflow: hidden; background: transparent; }}
  #root {{ position: relative; width: 100%; height: 100%; font-family: Montserrat, sans-serif; }}
  .slot {{ position: absolute; left: 0; top: 0; width: {W}px; height: {H}px; display: flex; justify-content: center; {zone} }}
  .card {{ opacity: 0; background: var(--card); color: var(--ink); border-radius: 34px;
           box-shadow: 0 18px 50px rgba(0,0,0,.35); }}
  .text {{ max-width: 900px; padding: 34px 48px 30px; text-align: center; }}
  .text .t {{ font-weight: 800; line-height: 1.08; letter-spacing: -0.01em; }}
  .text .accent {{ height: 12px; margin: 18px auto 0; width: 60%; border-radius: 6px; background: var(--accent); transform-origin: left center; }}
  .pic {{ padding: 14px; border-radius: 30px; }}
  .pic img {{ display: block; max-width: 800px; max-height: 620px; border-radius: 20px; }}
  .chart {{ width: 900px; padding: 40px 44px 34px; }}
  .ctitle {{ font-weight: 800; font-size: 54px; margin-bottom: 26px; line-height: 1.1; }}
  .row {{ display: flex; align-items: center; gap: 20px; margin: 16px 0; }}
  .lab {{ width: 190px; font-weight: 700; font-size: 36px; text-align: right; }}
  .track {{ flex: 1; height: 56px; border-radius: 14px; background: rgba(0,0,0,.08); overflow: hidden; }}
  .bar {{ height: 100%; width: 100%; border-radius: 14px; background: var(--accent); transform-origin: left center; }}
  .val {{ width: 150px; font-weight: 800; font-size: 40px; }}
</style>
</head>
<body>
<div id="root" data-composition-id="main" data-start="0" data-duration="{duration}" data-width="{W}" data-height="{H}">
{chr(10).join(clips)}
</div>
<script>
  const tl = gsap.timeline({{ paused: true }});
  {chr(10).join('  ' + t for t in tweens).strip()}
  window.__timelines["main"] = tl;
</script>
</body>
</html>
"""
    (proj / "index.html").write_text(doc, encoding="utf-8")
    (proj / "hyperframes.json").write_text(json.dumps({"paths": {"assets": "assets"}}, indent=2))
    return offset, duration


def render_overlay(proj: Path, out: Path, fps: int) -> Path:
    hf = ROOT / "node_modules" / ".bin" / "hyperframes"
    run([hf, "render", proj, "-o", out, "--format", "mov", "--fps", str(fps), "--quiet"], quiet=True)
    return out
