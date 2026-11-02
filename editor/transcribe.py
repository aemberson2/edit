"""faster-whisper transcription with word timestamps, cached per take."""

import hashlib
import json
from pathlib import Path

from .common import BUILD, load_smart_cut

# Whisper tends to "clean up" speech and silently drop ums and stutters.
# A disfluent prompt nudges it to write them down, which is what lets us cut them.
DISFLUENT_PROMPT = (
    "Umm, so, uh, I- I was like, you know, hmm... Okay. Let me, let me start again. "
    "Uh, so today I'm gonna, um, talk about this."
)
CACHE_VERSION = 2


def _key(video: Path, model: str, language: str) -> str:
    st = video.stat()
    raw = f"{video.resolve()}|{st.st_size}|{int(st.st_mtime)}|{model}|{language}|{CACHE_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def audio_for(video: Path) -> Path:
    """16 kHz mono WAV for a take (extracted with Smart-Cut's helper, cached)."""
    sc = load_smart_cut()
    cache = BUILD / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    st = video.stat()
    key = hashlib.sha1(f"{video.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
    wav = cache / f"{video.stem}.{key}.wav"
    if not wav.exists():
        sc.extract_audio(str(video), str(wav))
    return wav


_model = None


def _load_model(name: str):
    global _model
    if _model is None:
        import ctranslate2
        from faster_whisper import WhisperModel

        cuda = ctranslate2.get_cuda_device_count() > 0
        device, compute = ("cuda", "float16") if cuda else ("cpu", "int8")
        print(f"  loading faster-whisper {name} ({device})")
        _model = WhisperModel(name, device=device, compute_type=compute)
    return _model


def transcribe(video: Path, model: str = "small.en", language: str = "en") -> list:
    """Returns words: [{"w": "Hello,", "s": 0.52, "e": 0.81, "p": 0.97, "seg_end": False}, ...]"""
    cache = BUILD / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{video.stem}.{_key(video, model, language)}.words.json"
    if out.exists():
        return json.loads(out.read_text())

    wav = audio_for(video)
    m = _load_model(model)
    segments, _info = m.transcribe(
        str(wav),
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
        condition_on_previous_text=False,
        initial_prompt=DISFLUENT_PROMPT,
    )
    words = []
    for seg in segments:
        seg_words = list(seg.words or [])
        for i, w in enumerate(seg_words):
            text = w.word.strip()
            if not text:
                continue
            words.append({
                "w": text,
                "s": round(float(w.start), 3),
                "e": round(float(w.end), 3),
                "p": round(float(w.probability), 3),
                "seg_end": i == len(seg_words) - 1,
            })
    out.write_text(json.dumps(words))
    return words
