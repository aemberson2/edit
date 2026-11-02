"""Burned-in captions: short word groups, the spoken word highlighted, as an ASS file."""

from pathlib import Path

from .common import ROOT

FONT_DIR = ROOT / "node_modules" / "@expo-google-fonts" / "montserrat"


def fonts_dir() -> Path:
    """libass gets every Montserrat weight from one flat folder."""
    out = ROOT / "build" / "fonts"
    out.mkdir(parents=True, exist_ok=True)
    if FONT_DIR.exists():
        for ttf in FONT_DIR.glob("*/*.ttf"):
            dst = out / ttf.name
            if not dst.exists():
                dst.write_bytes(ttf.read_bytes())
    return out


def ass_color(hex_color: str, alpha: int = 0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def group_words(words, max_words: int):
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] >= 0.3):
            groups.append(cur)
            cur = []
        cur.append(w)
        if w["text"].rstrip("\"')").endswith((".", "?", "!", ",")):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def write_ass(words, cfg, path: Path, total: float):
    """words: [{"text", "start", "end"}] in output time."""
    c = cfg["captions"]
    W, H = cfg["video"]["width"], cfg["video"]["height"]
    groups = group_words(words, int(c["max_words"]))
    base, hi = ass_color(c["color"]), ass_color(c["highlight_color"])
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{c['font']},{c['font_size']},{base},{hi},&H00000000,&H80000000,0,0,0,0,100,100,4,0,1,{c['outline']},3,5,70,70,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    x, y = W // 2, int(c["y"])
    for gi, g in enumerate(groups):
        g_start = g[0]["start"]
        nxt = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else total
        g_end = g[-1]["end"] + 0.15
        if nxt - g_end < 0.35:  # bridge short gaps so captions don't flicker
            g_end = nxt
        g_end = min(max(g_end, g_start + 0.35), nxt if nxt > g_start else g_end, total)
        texts = [esc(w["text"].upper() if c["uppercase"] else w["text"]) for w in g]
        for k, w in enumerate(g):
            s = g_start if k == 0 else w["start"]
            e = g[k + 1]["start"] if k + 1 < len(g) else g_end
            if e <= s:
                continue
            parts = [f"{{\\c{hi}}}{t}{{\\c{base}}}" if j == k else t for j, t in enumerate(texts)]
            pop = "\\fscx88\\fscy88\\t(0,90,\\fscx100\\fscy100)" if k == 0 else ""
            lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Cap,,0,0,0,,{{\\pos({x},{y}){pop}}}" + " ".join(parts))
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return path
