"""English, word-level cut detection on top of Smart-Cut.

Smart-Cut (vendor/Smart-Cut) supplies audio extraction, FFmpeg silence detection,
the text-similarity measure and EDL export. Its detectors are tuned for Chinese
(character-level matching, Chinese filler list), so this module re-implements the
same passes (fillers, stutters, intra-segment restarts, repeats/retakes, false
starts) on English words and faster-whisper word timestamps.

Every kept segment starts `pad_before` (0.15 s) before its first word and ends
`pad_after` (0.25 s) after its last word, unless that would pull in a word we're
cutting, in which case it stops at that word's edge.
"""

import re
import wave
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

import numpy as np

from .common import load_smart_cut, media_duration

STOPWORDS = set("""a an the and or but so to of in on at for with is are was were be been it it's its i i'm
you you're we they he she this that these those my your our their me us them do does did not no yes
just about as if then than there here what when how why who which can will would could should have has had
""".split())

CUE_PATTERNS = [
    r"\b(let me|lemme|let's|i'?ll|i'm gonna|gonna) (try|start|do|say|redo|take|go) (that|this|it)?\s*(again|over|one more time)\b",
    r"\bstart (that )?over\b",
    r"\bone more time\b",
    r"\btake (two|three|four|2|3|4)\b",
    r"\bscratch that\b",
    r"\b(where|what) was i( saying)?\b",
    r"\bhold on\b",
    r"\bwait,? wait\b",
    r"\bmessed (that|it) up\b",
    r"\bhang on\b",
]
THROWAWAY = {"okay", "ok", "alright", "all", "right", "so", "and", "yeah", "cool", "um", "uh", "wait", "sorry", "oops"}


@dataclass
class Word:
    i: int
    take: int
    w: str
    n: str
    s: float
    e: float
    p: float
    seg_end: bool
    cut: str = ""
    note: str = ""

    @property
    def truncated(self):
        return self.w.rstrip(".,!?\"'").endswith(("-", "—", "–"))

    @property
    def ends_sentence(self):
        return self.w.rstrip("\"')").endswith((".", "?", "!"))


@dataclass
class Take:
    idx: int
    path: Path
    duration: float
    words: List[Word] = field(default_factory=list)
    silences: list = field(default_factory=list)


def norm(raw: str) -> str:
    t = raw.lower().replace("’", "'")
    t = re.sub(r"^[^\w$%']+|[^\w%']+$", "", t)
    return t


# ---------------------------------------------------------------------------
# Voice calibration ("tune to my voice")
# ---------------------------------------------------------------------------

def frame_db(wav_path: Path, frame_ms: int = 30) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    n = int(sr * frame_ms / 1000)
    if len(data) < n:
        return np.array([-90.0])
    frames = data[: len(data) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms)


def calibrate(wavs: List[Path], words_by_take: List[List[dict]], cfg_cut: dict) -> dict:
    """Measure the speaker's noise floor, speech level and natural pause length."""
    db = np.concatenate([frame_db(w) for w in wavs])
    noise = float(np.percentile(db, 10))
    speech = float(np.percentile(db, 90))
    auto_thr = float(np.clip(noise + 0.35 * (speech - noise), -55, -25))

    gaps, n_words, talk_time = [], 0, 0.0
    for words in words_by_take:
        n_words += len(words)
        for a, b in zip(words, words[1:]):
            g = b["s"] - a["e"]
            if 0.02 < g < 3.0 and not a["w"].rstrip("\"')").endswith((".", "?", "!")):
                gaps.append(g)
        if words:
            talk_time += words[-1]["e"] - words[0]["s"]
    p90 = float(np.percentile(gaps, 90)) if len(gaps) >= 5 else 0.35
    floor = cfg_cut["pad_before"] + cfg_cut["pad_after"] + 0.15  # a cut must remove at least 0.15 s
    auto_gap = float(np.clip(p90 + 0.15, max(0.55, floor), 1.0))

    thr = cfg_cut["silence_threshold_db"]
    gap = cfg_cut["max_gap"]
    return {
        "noise_floor_db": round(noise, 1),
        "speech_level_db": round(speech, 1),
        "silence_threshold_db": round(auto_thr if thr == "auto" else float(thr), 1),
        "natural_pause_p90": round(p90, 3),
        "max_gap": round(auto_gap if gap == "auto" else max(float(gap), floor), 3),
        "words_per_minute": round(60 * n_words / talk_time) if talk_time > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Detection passes
# ---------------------------------------------------------------------------

def inside_silence(t: float, silences) -> Optional[object]:
    for s in silences:
        if s.start <= t < s.end:
            return s
    return None


def refine_with_silence(take: Take):
    """Whisper word edges drift into pauses; pull them back to where sound starts/stops."""
    for w in take.words:
        s = inside_silence(w.s, take.silences)
        if s and s.end < w.e - 0.05:
            w.s = s.end
        s = inside_silence(w.e - 1e-3, take.silences)
        if s and s.start > w.s + 0.05:
            w.e = s.start


def drop_hallucinations(take: Take):
    for w in take.words:
        dur = max(w.e - w.s, 1e-3)
        covered = sum(max(0.0, min(w.e, s.end) - max(w.s, s.start)) for s in take.silences)
        if covered / dur > 0.8 and w.p < 0.5:
            w.cut, w.note = "noise", "transcribed over silence"


def flag_fillers(words: List[Word], cfg_cut: dict):
    hard = {f.lower() for f in cfg_cut["remove_fillers"]}
    for w in words:
        if not w.cut and w.n in hard:
            w.cut, w.note = "filler", w.n
    if cfg_cut.get("remove_soft_fillers"):
        multi = [f.split() for f in cfg_cut["soft_fillers"] if " " in f]
        single = {f for f in cfg_cut["soft_fillers"] if " " not in f}
        live = [w for w in words if not w.cut]
        for k, w in enumerate(live):
            prev = live[k - 1] if k else None
            comma = w.w.endswith(",") or (prev is not None and prev.w.endswith(","))
            if w.n in single and comma:
                w.cut, w.note = "filler", w.n
            for m in multi:
                seq = live[k:k + len(m)]
                if [x.n for x in seq] == m and seq[-1].w.endswith(","):
                    for x in seq:
                        x.cut, x.note = "filler", " ".join(m)


def flag_stutters(words: List[Word], cfg_cut: dict):
    keep_doubles = set(cfg_cut["keep_doubled_words"])
    for w in words:
        if not w.cut and w.truncated:
            w.cut, w.note = "stutter", w.w
    live = [w for w in words if not w.cut]
    for a, b in zip(live, live[1:]):
        if a.take == b.take and a.n and a.n == b.n and a.n not in keep_doubles and b.s - a.e < 1.5:
            a.cut, a.note = "stutter", f"{a.n} {b.n}"


def _disfluent_before(words_all: List[Word], live: List[Word], j: int) -> bool:
    """Is there a pause, cut word, truncation or dash right before live[j]?"""
    a, b = live[j - 1], live[j]
    if b.s - a.e >= 0.35 or a.truncated or a.w.endswith(("-", "—", "...")):
        return True
    return b.i - a.i > 1  # a filler/stutter word sat between them


def flag_restarts(words: List[Word]):
    """'the best way to, the best way to save money' -> keep the second attempt."""
    live = [w for w in words if not w.cut]
    i = 0
    while i < len(live):
        hit = None
        for n in (6, 5, 4, 3, 2):
            if i + 2 * n > len(live):
                continue
            gram = [w.n for w in live[i:i + n]]
            if n == 2 and all(g in STOPWORDS for g in gram):
                continue
            for j in range(i + n, min(i + n + 4, len(live) - n + 1)):
                if live[j].take != live[i].take or live[j].s - live[i].s > 10:
                    break
                if [w.n for w in live[j:j + n]] != gram:
                    continue
                extra = j - (i + n)
                if extra == 0 or (n >= 4 or _disfluent_before(words, live, j)):
                    hit = j
                    break
            if hit:
                break
        if hit:
            for w in live[i:hit]:
                w.cut, w.note = "restart", " ".join(x.w for x in live[i:hit])
            i = hit
        else:
            i += 1


def build_phrases(words: List[Word], max_gap: float) -> List[List[Word]]:
    """Split kept words into sentences/lines. A pause only counts if it's real silence,
    not the space left by a filler or stutter we removed."""
    by_i = {w.i: w for w in words}
    phrases, cur = [], []
    for w in words:
        if w.cut:
            continue
        pause = 0.0
        if cur and cur[-1].take == w.take:
            pause = max(by_i[k].s - by_i[k - 1].e for k in range(cur[-1].i + 1, w.i + 1))
        if cur and (w.take != cur[-1].take or pause > max(0.7, max_gap) or cur[-1].ends_sentence):
            phrases.append(cur)
            cur = []
        cur.append(w)
    if cur:
        phrases.append(cur)
    return phrases


def _contains(small, big):
    n = len(small)
    return any(big[k:k + n] == small for k in range(len(big) - n + 1))


def flag_retakes(phrases: List[List[Word]], sim_threshold: float, text_similarity):
    """Repeated lines across the take: keep the later, complete delivery."""
    toks = [[w.n for w in p if w.n] for p in phrases]
    dead = [False] * len(phrases)

    def kill(k, kind, note):
        dead[k] = True
        for w in phrases[k]:
            w.cut, w.note = kind, note

    for i in range(len(phrases)):
        if dead[i] or not toks[i]:
            continue
        A = toks[i]
        for j in range(i + 1, len(phrases)):
            if dead[j] or not toks[j]:
                continue
            B = toks[j]
            near = j - i <= 3
            ref = " ".join(w.w for w in phrases[j])[:40]
            # 1. false start: an abandoned line that the next one restarts
            if near and 2 <= len(A) < len(B) and SequenceMatcher(None, A, B[:len(A)]).ratio() >= 0.75:
                kind = "retake" if phrases[i][-1].ends_sentence else "false start"
                kill(i, kind, f'restarted as "{ref}"')
                break
            # 2. abandoned line fully contained in the next one
            if near and 3 <= len(A) < len(B) and _contains(A, B):
                kill(i, "repeat", f'said again in "{ref}"')
                break
            # 3. the same line delivered again (another take)
            short = min(len(A), len(B)) <= 3
            if short:
                if near and A == B:
                    kill(i, "retake", f'redone as "{ref}"')
                    break
                continue
            sim = max(SequenceMatcher(None, A, B).ratio(), text_similarity(" ".join(A), " ".join(B)))
            if sim >= sim_threshold:
                if len(B) >= 0.7 * len(A):
                    kill(i, "retake", f'better take later: "{ref}"')
                    break
                kill(j, "retake", f'shorter repeat of "{" ".join(w.w for w in phrases[i])[:40]}"')
    return phrases


def flag_cues(phrases: List[List[Word]]):
    for p in phrases:
        live = [w for w in p if not w.cut]
        if not live:
            continue
        text = " ".join(w.n for w in live)
        if all(w.n in THROWAWAY for w in live) and len(live) <= 3:
            for w in live:
                w.cut, w.note = "throwaway", text
            continue
        for pat in CUE_PATTERNS:
            m = re.search(pat, text)
            if not m:
                continue
            if len(live) <= 10:
                cut_from = 0
            else:  # cut from the cue onward
                cut_from = len(text[: m.start()].split())
            for w in live[cut_from:]:
                w.cut, w.note = "retake cue", m.group(0)
            break


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

def build_segments(take: Take, cfg_cut: dict, max_gap: float, min_cut: float = 0.12):
    """Kept intervals for one take, padded around words, never padded into a cut word."""
    pb, pa = cfg_cut["pad_before"], cfg_cut["pad_after"]
    W = take.words
    runs, cur = [], []
    for k, w in enumerate(W):
        if w.cut:
            if cur:
                runs.append(cur)
                cur = []
            continue
        if cur and w.s - cur[-1].e > max_gap:
            runs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        runs.append(cur)

    segs = []
    for run in runs:
        first, last = run[0], run[-1]
        fi, li = W.index(first), W.index(last)
        start = first.s - pb
        if fi > 0:
            prev = W[fi - 1]
            start = max(start, prev.e if prev.cut else min(first.s - 0.05, prev.e + pa))
        end = last.e + pa
        if li + 1 < len(W):
            nxt = W[li + 1]
            end = min(end, nxt.s if nxt.cut else max(last.e + 0.05, nxt.s - pb))
        segs.append([max(0.0, start), min(take.duration, end)])

    # Merge segments whose cut would remove almost nothing (it'd be a pointless jump cut)
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_cut:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    # Words swallowed by a merge are kept after all; say so honestly
    for w in W:
        if w.cut and any(a <= (w.s + w.e) / 2 <= b for a, b in merged):
            w.note = f"kept, too tight to cut ({w.cut}: {w.note})"
            w.cut = ""
    return merged


def analyze(takes_paths: List[Path], cfg: dict, words_by_take: List[List[dict]], wavs: List[Path]) -> dict:
    sc = load_smart_cut()
    cc = cfg["cut"]
    profile = calibrate(wavs, words_by_take, cc)

    takes, all_words, gi = [], [], 0
    for ti, (path, raw_words, wav) in enumerate(zip(takes_paths, words_by_take, wavs)):
        t = Take(idx=ti, path=path, duration=media_duration(path))
        t.silences = sc.detect_silences(str(wav), profile["silence_threshold_db"], 0.2)
        for rw in raw_words:
            t.words.append(Word(gi, ti, rw["w"], norm(rw["w"]), rw["s"], rw["e"], rw["p"], rw.get("seg_end", False)))
            gi += 1
        refine_with_silence(t)
        drop_hallucinations(t)
        takes.append(t)
        all_words.extend(t.words)

    flag_fillers(all_words, cc)
    flag_stutters(all_words, cc)
    flag_restarts(all_words)
    phrases = build_phrases(all_words, profile["max_gap"])
    flag_cues(phrases)
    phrases = build_phrases(all_words, profile["max_gap"])
    flag_retakes(phrases, cc["retake_similarity"], sc.text_similarity)

    segments = []
    for t in takes:
        for a, b in build_segments(t, cc, profile["max_gap"]):
            segments.append({"take": t.idx, "start": round(a, 3), "end": round(b, 3)})

    return {"takes": takes, "words": all_words, "segments": segments, "profile": profile}
