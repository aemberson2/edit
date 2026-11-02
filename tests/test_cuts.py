"""Unit tests for the cut detectors. Run: .venv/bin/python -m unittest discover tests"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from editor import analyze as A  # noqa: E402
from editor.common import load_smart_cut  # noqa: E402

CFG = json.loads((Path(__file__).resolve().parent.parent / "config.json").read_text())


def make_take(script, idx=0, start=0.5, word=0.28, gap=0.06):
    """script: list of strings or floats (a float = extra pause in seconds)."""
    t, words = start, []
    for item in script:
        if isinstance(item, float):
            t += item
            continue
        for raw in item.split():
            words.append(A.Word(0, idx, raw, A.norm(raw), round(t, 3), round(t + word, 3), 0.95, False))
            t += word + gap
    take = A.Take(idx=idx, path=Path(f"take{idx}.mov"), duration=t + 1.0, words=words)
    return take


def run_passes(takes, max_gap=0.6):
    sc = load_smart_cut()
    allw = [w for t in takes for w in t.words]
    for i, w in enumerate(allw):
        w.i = i
    A.flag_fillers(allw, CFG["cut"])
    A.flag_stutters(allw, CFG["cut"])
    A.flag_restarts(allw)
    A.flag_cues(A.build_phrases(allw, max_gap))
    A.flag_retakes(A.build_phrases(allw, max_gap), CFG["cut"]["retake_similarity"], sc.text_similarity)
    return allw


def kept_text(words):
    return " ".join(w.w for w in words if not w.cut)


class TestDetectors(unittest.TestCase):
    def test_fillers_and_stutters(self):
        t = make_take(["So um I I think this is uh really really good."])
        run_passes([t])
        self.assertEqual(kept_text(t.words), "So I think this is really really good.")

    def test_truncated_word(self):
        t = make_take(["We went to the sto- store yesterday."])
        run_passes([t])
        self.assertEqual(kept_text(t.words), "We went to the store yesterday.")

    def test_restart_inside_sentence(self):
        t = make_take(["The best way to", 0.5, "the best way to save money is automating it."])
        run_passes([t])
        self.assertEqual(kept_text(t.words), "the best way to save money is automating it.")

    def test_parallel_phrase_not_cut(self):
        t = make_take(["The more you save, the more you have."])
        run_passes([t])
        self.assertEqual(kept_text(t.words), "The more you save, the more you have.")

    def test_false_start_then_full_line(self):
        t = make_take(["So today I want to", 1.4, "So today I want to talk about saving money."])
        run_passes([t])
        self.assertEqual(kept_text(t.words), "So today I want to talk about saving money.")

    def test_retake_across_takes_keeps_later(self):
        t0 = make_take(["Okay.", 1.0, "First you need a budget for every month.", 1.0, "That part is easy."], idx=0)
        t1 = make_take(["First, you need a budget for every single month.", 0.8, "Then automate it."], idx=1)
        run_passes([t0, t1])
        self.assertEqual(kept_text(t0.words), "That part is easy.")
        self.assertEqual(kept_text(t1.words), "First, you need a budget for every single month. Then automate it.")

    def test_retake_cue_removed(self):
        t = make_take(["Here is the trick that", 0.8, "Wait, let me start over.", 1.0, "Here is the trick that saves me money."])
        run_passes([t])
        self.assertEqual(kept_text(t.words), "Here is the trick that saves me money.")

    def test_distinct_lines_kept(self):
        lines = ["I started saving last year.", "It changed everything for me.", "Now I save every month.",
                 "And you can do it too."]
        t = make_take([x for l in lines for x in (l, 0.5)])
        run_passes([t])
        self.assertEqual(kept_text(t.words), " ".join(lines))


class TestSegments(unittest.TestCase):
    def test_padding_and_dead_air(self):
        t = make_take(["Hello there friend.", 2.0, "Second line here."])
        run_passes([t])
        segs = A.build_segments(t, CFG["cut"], max_gap=0.6)
        self.assertEqual(len(segs), 2)
        w = t.words
        self.assertAlmostEqual(segs[0][0], w[0].s - 0.15, places=3)
        self.assertAlmostEqual(segs[0][1], w[2].e + 0.25, places=3)
        self.assertAlmostEqual(segs[1][0], w[3].s - 0.15, places=3)
        self.assertAlmostEqual(segs[1][1], w[5].e + 0.25, places=3)

    def test_pad_never_includes_cut_word(self):
        t = make_take(["So", 0.4, "um", 0.4, "here we go."])
        run_passes([t])
        segs = A.build_segments(t, CFG["cut"], max_gap=0.6)
        um = next(w for w in t.words if w.n == "um")
        for a, b in segs:
            self.assertFalse(a < (um.s + um.e) / 2 < b, "the um ended up inside a kept segment")
        self.assertLessEqual(segs[0][1], um.s + 1e-6)
        self.assertGreaterEqual(segs[1][0], um.e - 1e-6)

    def test_short_pause_not_cut(self):
        t = make_take(["One two three.", 0.3, "Four five six."])
        run_passes([t])
        self.assertEqual(len(A.build_segments(t, CFG["cut"], max_gap=0.6)), 1)


class TestScript(unittest.TestCase):
    def test_parse_and_align(self):
        from editor import popups
        p = Path(__file__).resolve().parent / "_tmp_script.txt"
        p.write_text("Today we talk money. [TEXT: Money talk]\n[PIC: chart.png]\nThis is my budget.\n"
                     "It went down a lot. [CHART: Spend | Jan 800, Feb 600, Mar 400]\n")
        try:
            cues = popups.parse_script(p)
        finally:
            p.unlink()
        self.assertEqual([c["kind"] for c in cues], ["TEXT", "PIC", "CHART"])
        self.assertEqual(cues[1]["line"], "This is my budget.")
        title, rows = popups.parse_chart(cues[2]["arg"])
        self.assertEqual(title, "Spend")
        self.assertEqual([r["value"] for r in rows], [800, 600, 400])
        words = "today we talk money this is my budget it went down a lot".split()
        out = [{"text": w, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, w in enumerate(words)]
        timed = popups.time_cues(cues, out, CFG)
        self.assertAlmostEqual(timed[1]["start"], out[4]["start"] - 0.1, places=3)
        self.assertAlmostEqual(timed[2]["start"], out[8]["start"] - 0.1, places=3)


if __name__ == "__main__":
    unittest.main()
