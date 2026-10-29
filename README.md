# Podcaster

Turn a document or a web page into a finished podcast episode.

Podcaster reads your source material, asks an LLM to write a broadcast-ready
dialogue script, speaks it with a TTS model (one voice per host), and
assembles the result into a single MP3 — or an MP4 with an animated
audio-waveform screen, or an MP4 of AI-illustrated slides, when you want video.

It runs entirely on your machine. No database, no accounts, no cloud
orchestration: a FastAPI backend plus a small web UI.

```
source (pdf / md / url / text)
   └─ parse ─▶ chunk ─▶ script (LLM, per-segment)
                          └─ TTS per turn ─▶ clips ─▶ ffmpeg merge ─▶ mp3
                                                        ├─ waveform render ─▶ mp4
                                                        └─ slide plan ─▶ images ─▶ mp4
```

---

## Quick start

```powershell
cd E:\Voice\Podcaster
.\run.ps1
```

Then open <http://127.0.0.1:8760>.

`run.ps1` loads `env.ps1` (your MiniMax key), installs any missing Python
packages, checks for `ffmpeg`, and starts the server. Use `-Port 9000` to
change the port or `-Reload` while developing.

### Settings tab

Everything is pre-filled for MiniMax and works immediately with the key in
`env.ps1`:

| Setting | Default |
| --- | --- |
| LLM base URL | `https://api.minimaxi.com/v1` |
| LLM model | `MiniMax-M2.5` |
| TTS endpoint | `https://api.minimaxi.com/v1` (`/v1/t2a_v2`) |
| TTS model | `speech-2.5-hd-preview` |
| Image endpoint | same as the TTS endpoint (`/v1/image_generation`) |
| Image model | `image-01` |

The LLM slot accepts any OpenAI-compatible `/chat/completions` endpoint
(MiniMax, OpenAI, DeepSeek, vLLM, Ollama's compat mode…). The TTS slot accepts
MiniMax's native T2A v2 API or any OpenAI-compatible `/audio/speech` endpoint.
The image slot is only used by the illustrated-slides export; it reuses the API
key above unless you give it a different one.

Use **Test connections** to verify both endpoints, and **Test voice** to check
a specific `voice_id` before assigning it to a host.

---

## Using it

### 1 · Source material

- **Web link** — any article page. The readable text is extracted with
  BeautifulSoup (scripts, navs and footers stripped).
- **Upload file** — PDF, Markdown, TXT, HTML or DOCX. PDF text is extracted
  with `pypdf` (falling back to `pdfplumber`). Scanned/image-only PDFs are
  detected and reported rather than producing a silent empty episode.
- **Paste text** — for notes, transcripts or anything already on screen.
- You can also point at a file already on disk with the path field.

Long sources are split into paragraph-aware chunks (~6000 characters) so no
single LLM call receives the whole document.

### 2 · Podcast setup

| Control | Notes |
| --- | --- |
| Language | English or Chinese — the script is written, the TTS is boosted, and the video font all follow this choice. |
| Style | Narrative, Interview, Chitchat, Deep dive, News, Educational, Storytelling, Comedy. |
| Target length | 1–45 minutes. The writer aims for the budget and the finished job reports the real duration. |
| Hosts | 1–6 hosts. Set a name, a gender (voice picking follows it) or pin a specific voice. |

Each style also implies a role pairing — Interview makes host #1 the
interviewer and host #2 the guest; Educational makes #1 the teacher and #2 the
curious listener.

### 3 · Export

- **Audio** → one MP3 with a short pause between turns.
- **Video** → an MP4 with an animated waveform, coloured per speaker, with the
  episode title and the current speaker on screen.
- **Illustrated slides** → replaces the waveform with AI-drawn illustrations of
  the points in the script (video export only).
- **Show subtitles on the video** → burn the spoken text into the picture
  under the waveform (video export only).

Set an explicit output path, or leave it blank to get a timestamped file in
the output folder.

---

## How the script is written

1. **Plan** — the source is condensed and the LLM produces a segment outline.
2. **Write** — each chunk is turned into one segment of dialogue with a word
   budget. Output is a strict `NAME: line` format that is parsed and
   validated; malformed lines are dropped rather than spoken.
3. **Intro / outro** — a separate, tightly budgeted call adds the hook and
   the sign-off.
4. **Trim** — if the result still runs long, turns are dropped from the
   middle of the episode while the hook and the closing lines are preserved.

TTS is then called once per turn. Turns longer than the provider's limit are
split on sentence boundaries and stitched back together.

## How the video is built

Each speaker turn becomes its own short clip: a dark background, an animated
`showwaves` waveform in that speaker's colour, the episode title and the
speaker's name. The clips are concatenated into the final MP4 with ffmpeg.

### Subtitles

With **Show subtitles on the video** enabled, the TTS engine is asked for word
level timestamps (`subtitle_enable` / `subtitle_type: word`); it returns them in
a subtitle file that is downloaded and parsed. The words are then grouped into
caption lines:

- a line is at most 64 half-width units wide (a CJK glyph counts as two), which
  is roughly 32 Chinese characters or 64 Latin characters — always a single line,
  never wrapped;
- a sentence that is longer than that is split across several captions, each
  with its own timing;
- a pause of more than 400 ms starts a new caption, and a caption stays on
  screen until the next one appears so the text does not flicker.

The captions are drawn under the waveform with ffmpeg's `drawtext`, one filter
per line, each enabled only while it is being spoken. The host name is not
repeated in the caption because it is already on screen above the waveform.

### Illustrated slides

With **Illustrated slides** enabled, the waveform is replaced by pictures that
illustrate what the episode is talking about.

1. **Plan** — once the audio exists, the LLM is given the numbered turns with
   their *measured* speaking time and returns a JSON slide plan: how many
   slides, what each one shows, and which turns it covers. The count comes from
   the real duration at roughly 25 seconds per slide (capped at 20), so the deck
   is sized from the episode rather than guessed.
2. **Draw** — each slide's subject is sent to MiniMax's image API
   (`/v1/image_generation`, `image-01`, 16:9 → exactly 1280×720). Every prompt
   ends with the *same* style block — one cohesive series, a fixed palette, and
   an explicit "no text, no letters, no numbers, no logos" clause, which is what
   stops the model painting garbled captions into the art.
3. **Render** — the slides fill the frame instead of the waveform, with a slow
   push-in that keeps moving across the whole episode, a soft gradient scrim so
   the captions stay readable over any picture, and the progress bar intact.

Before the first slide, a title card shows the episode title and the host
names over the opening seconds of the intro; if the intro runs longer, the
remainder carries on over slide 1 with its captions re-timed. Subtitles work
exactly as in the waveform layout.

If the image API is unavailable or the planner returns nothing, the job falls
back to the waveform video with a warning in the log rather than failing.

---

## Voices

`podcaster/voices_catalog.json` holds the catalogue. Every entry in it was
verified against the live MiniMax TTS endpoint on the key in `env.ps1` —
the account exposes 18 usable system voices (7 male, 11 female), and they
speak both English and Chinese.

Voice picking is automatic and respects the host genders you set. You can
override any host with an explicit voice from the dropdown, or paste any
`voice_id` into the **Test voice** field to check one that is not in the
catalogue (cloned voices, for example).

## Requirements

- Python 3.10+
- `ffmpeg` on `PATH` — needed for merging and for video export
- See `requirements.txt` for the Python packages (`run.ps1` installs them)

## Length control

The length target is measured against the live TTS output, not guessed:
159 spoken words/minute for English and 257 Chinese characters/minute at
speed 1.0. The writer budgets each segment against those rates, and if the
result still runs long the script is trimmed — the hook and the sign-off are
always kept, and turns are removed in alternating pairs so the exchange stays
balanced. The finished job reports the real duration.

## Tests

```powershell
python tests\run.py
```

Covers speaker-name resolution, voice assignment per host gender, the
length budget, the live job event stream (SSE), the subtitle layout, and the
slide planner (plan parsing, turn assignment, slide count and the rendered
slide and waveform layouts).

The Jobs tab is driven by a Server-Sent Events stream at
`/api/jobs/{id}/events`. Every event carries a monotonic `seq`; the first
frame is a `snapshot` of the whole job, and the client drops any event the
snapshot already covers. Only a **terminal** status (`done` / `error` /
`cancelled`) ends the stream — the non-terminal `running` status must leave
it open, otherwise the tab freezes on "Job started" until the page is
reloaded.

## Layout

```
app.py                     FastAPI routes and static hosting
podcaster/
  config.py                settings persistence (data/config.json)
  sources.py               fetch + parse pdf/md/html/docx/url
  chunking.py              paragraph-aware chunking, length estimation
  llm.py                   OpenAI-compatible chat client
  script.py                planning, segment writing, parsing, trimming
  tts.py                   MiniMax T2A v2 + OpenAI-compatible TTS
  media.py                 ffmpeg merge and waveform / slide rendering
  subtitles.py             subtitle parsing and one-line caption layout
  slides.py                slide planning, turn assignment, style block
  images.py                MiniMax image generation client
  voices.py                catalogue + host→voice assignment
  pipeline.py              the end-to-end flow
  jobs.py                  job registry, progress, event feed
  voices_catalog.json      the API-verified voice catalogue
web/                       the UI (plain HTML/CSS/JS, no build step)
tests/                     regression checks
data/                      config, job workdirs, uploads   (git-ignored)
output/                    default export folder             (git-ignored)
samples/                   sample source documents
```

Settings live in `data/config.json`; jobs and their intermediate clips live in
`data/jobs/<id>/`. Nothing is written to a database. `env.ps1` holds the API
key and `data/` and `output/` hold generated files — all three are excluded
from version control by `.gitignore`.
