# Talking-head editor

Turns a folder of raw phone takes into a finished vertical MP4 (1080x1920) with jump cuts, burned-in captions and popups.

```bash
./edit
```

1. Drop your takes in `input/`. They're used in name order (IMG_0001, IMG_0002, ...).
2. Put your script in `script.txt`, with popup notes where you want them:
   - `[TEXT: Save $500 a month]` text callout
   - `[PIC: receipt.jpg]` picture popup (put the image in `input/`)
   - `[CHART: Monthly spend | Jan 820, Feb 640, Mar 410]` bar chart (title optional)

   A note pops up when you start saying the line it's on. A note alone on its own line belongs to the next line.
3. Run `./edit`. It shows the cut list and waits for you to approve it. Type `e` to edit the list, `y` to render.
4. Your video lands in `output/`.

The first run installs anything missing: Python, FFmpeg and Node 22 (via Homebrew on Mac, apt on Linux), faster-whisper, HyperFrames, and it clones [qkirara/Smart-Cut](https://github.com/qkirara/Smart-Cut) into `vendor/`. The Whisper model downloads from HuggingFace the first time it's used.

## How cutting works

faster-whisper transcribes each take with word timestamps. Smart-Cut's silence detection snaps word edges to where sound actually starts and stops, and then these get cut:

- dead air (pauses longer than your natural pause length)
- fillers (um, uh, er, hmm...)
- stutters (`I I think`, `sto- store`)
- restarts inside a sentence (`the best way to, the best way to save`)
- false starts and retakes. When you say a line more than once, the last full delivery wins, across takes too
- retake cues like "let me start over" or "wait, one more time", plus throwaway "okay"s between attempts

Every kept segment starts 0.15s before its first word and ends 0.25s after its last, unless that would pull in a word being cut.

It tunes itself to your voice on every run: it measures your noise floor, speech level and how long you naturally pause mid-sentence, then sets the silence threshold and the pause cutoff from those. The numbers print during the run and get saved to `build/voice_profile.json`. To pin them, set `max_gap` / `silence_threshold_db` in `config.json` instead of `"auto"`.

## Options

```
./edit --yes            render without the prompt (applies your edits in build/cutlist.txt)
./edit --analyze-only   just show the cut list
./edit --model medium.en   better transcription, slower (default small.en)
./edit --no-popups --no-captions --name my_video
```

Look and feel (caption font, size, colors, popup zone `top`/`middle`/`bottom`, loudness) live in `config.json`.

## Files

- `edit` sets everything up, then runs `edit.py`
- `editor/analyze.py` English word-level cut detection on top of Smart-Cut
- `editor/cutlist.py` cut list display, approval and manual overrides
- `editor/render.py` FFmpeg cut, vertical reframe (HDR iPhone footage gets tone-mapped), final composite
- `editor/captions.py` word-highlight captions (ASS, Montserrat)
- `editor/popups.py` script notes -> transcript timing -> HyperFrames transparent overlay
- `build/` intermediates: transcripts (cached), cut list, captions, popup project, overlay
- `tests/` unit tests (`.venv/bin/python -m unittest discover tests`) and a synthetic fixture for testing the render path
