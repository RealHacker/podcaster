"""Media assembly: merging audio clips and rendering the video."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .subtitles import SubtitleLine, clean_caption_text

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

ProgressFn = Callable[[str, float], None]


class MediaError(RuntimeError):
    pass


def _run(cmd: list[str], timeout: float = 1800) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise MediaError("ffmpeg failed:\n" + "\n".join(tail))
    return proc.stdout


def probe_duration(path: Path) -> float:
    try:
        out = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=nw=1:nk=1", str(path)], timeout=120)
        return float(out.strip().splitlines()[0])
    except Exception:
        return 0.0


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise MediaError("ffmpeg was not found on PATH. Install it and try again.")


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #

FONT_CANDIDATES = {
    "zh": ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc",
           "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"],
    "en": ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
           "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/calibri.ttf"],
}


def find_font(language: str = "en") -> str | None:
    order = FONT_CANDIDATES.get(language, []) + FONT_CANDIDATES["en"] + FONT_CANDIDATES["zh"]
    for path in order:
        if Path(path).exists():
            return path
    for path in (Path("/usr/share/fonts"), Path("/System/Library/Fonts")):
        if path.exists():
            for found in path.rglob("*.tt[fc]"):
                return str(found)
    return None


# --------------------------------------------------------------------------- #
# Audio merging
# --------------------------------------------------------------------------- #

@dataclass
class AudioSegment:
    path: Path
    speaker: str
    start: float = 0.0
    duration: float = 0.0


@dataclass
class MergedAudio:
    path: Path
    duration: float = 0.0
    segments: list[AudioSegment] = field(default_factory=list)


def make_silence(path: Path, seconds: float, sample_rate: int = 44100) -> Path:
    cmd = [FFMPEG, "-y", "-f", "lavfi", "-i",
           f"anullsrc=r={sample_rate}:cl=mono", "-t", f"{seconds:.3f}",
           "-c:a", "libmp3lame", "-b:a", "128k", str(path)]
    _run(cmd, timeout=120)
    return path


def merge_audio(clips: list[Path], out_path: Path, gap_seconds: float = 0.28,
                workdir: Path | None = None) -> MergedAudio:
    """Concatenate mp3 clips, inserting a short pause between turns."""
    clips = [c for c in clips if c and c.exists()]
    if not clips:
        raise MediaError("No audio clips were produced.")

    tmpdir = workdir or Path(tempfile.mkdtemp(prefix="podcaster_merge_"))
    tmpdir.mkdir(parents=True, exist_ok=True)
    gap_file = tmpdir / "_gap.mp3"
    if gap_seconds > 0:
        make_silence(gap_file, gap_seconds)

    listing = tmpdir / "_concat.txt"
    lines: list[str] = []
    for i, clip in enumerate(clips):
        if i > 0 and gap_seconds > 0 and gap_file.exists():
            lines.append(f"file '{gap_file.as_posix()}'")
        lines.append(f"file '{clip.resolve().as_posix()}'")
    listing.write_text("\n".join(lines), encoding="utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
           "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "44100", "-ac", "1",
           str(out_path)]
    _run(cmd)
    return MergedAudio(path=out_path, duration=probe_duration(out_path))


def trim_audio(source: Path, start: float, duration: float, out_path: Path) -> Path:
    """Cut a piece out of an audio file. The cut is re-encoded, so it is exact."""
    if duration <= 0:
        raise MediaError("Cannot trim a clip to zero length.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-i", str(source)]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-t", f"{duration:.3f}", "-c:a", "libmp3lame", "-b:a", "192k",
            "-ar", "44100", "-ac", "1", str(out_path)]
    _run(cmd, timeout=600)
    return out_path


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #

PALETTE = {
    "male": "0x38BDF8",
    "female": "0xF472B6",
    "mixed": "0xA78BFA",
    "default": "0x34D399",
}


@dataclass
class VideoSegment:
    audio: Path
    speaker: str
    color: str
    label: str = ""
    subtitles: list[SubtitleLine] = field(default_factory=list)
    slide: Path | None = None        # full-screen illustration for this turn
    title_card: bool = False         # show the episode title and host names
    hosts: list[str] = field(default_factory=list)


def _text_file(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    handle.write(text or "")
    handle.close()
    return Path(handle.name)


def _fit(text: str, max_chars: int) -> str:
    """Trim a title so it cannot run off the video frame."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _gradient_source(width: int, height: int, fps: int, duration: float) -> str:
    """The backdrop used by the title card and by slides without an image."""
    return (
        f"gradients=size={width}x{height}:rate={fps}:"
        f"c0=0x0B1220:c1=0x3B0A5C:c2=0x0B1220:nb_colors=3:"
        f"x0={int(width * 0.22)}:y0={int(height * 0.10)}:"
        f"x1={int(width * 0.80)}:y1={int(height * 0.92)}:d={duration:.3f}"
    )


def _ken_burns(elapsed_before: float, episode_seconds: float, fps: int,
               max_zoom: float = 1.10) -> str:
    """A slow push-in that keeps running across the whole episode.

    ``on`` is the output frame number of the zoompan filter, so adding the
    time already elapsed makes the motion continuous from one segment to the
    next instead of restarting on every clip.
    """
    if episode_seconds <= 0:
        return "1"
    rate = (max_zoom - 1.0) / episode_seconds
    return f"min(1+{rate:.8f}*({elapsed_before:.3f}+on/{fps}),{max_zoom})"


def _scrim_image(height: int) -> Path | None:
    """A black-to-transparent ramp that seats the captions over a picture.

    Drawn as a gradient rather than a flat band so the bottom of a slide
    fades into the caption area instead of ending in a hard line.
    """
    path = Path(tempfile.gettempdir()) / f"podcaster_scrim_{os.getpid()}_{height}.png"
    if path.exists():
        return path
    cmd = [FFMPEG, "-y", "-f", "lavfi", "-i", f"color=c=black:s=8x{height}",
           "-frames:v", "1",
           "-vf", "format=rgba,geq=r='0':g='0':b='0':a='255*0.66*pow(Y/H,0.42)'",
           str(path)]
    try:
        _run(cmd, timeout=120)
    except MediaError:
        return None
    return path


def render_segment_video(
    segment: VideoSegment,
    out_path: Path,
    *,
    title: str,
    font_path: str | None = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    background: str = "0x0B1220",
    elapsed_before: float = 0.0,
    episode_seconds: float = 0.0,
    mode: str = "waveform",
) -> float:
    """Render one clip of the episode.

    ``mode`` picks the picture: ``waveform`` draws the animated waveform,
    ``slides`` shows the turn's illustration full screen (or the title card
    for the opening segment).
    """
    duration = probe_duration(segment.audio)
    if duration <= 0:
        duration = 2.0

    temp_files: list[Path] = []
    font = font_path or find_font("en")

    track_w = int(width * 0.74)
    track_x = (width - track_w) // 2
    track_y = height - 34

    # --- inputs ------------------------------------------------------------ #
    input_specs: list[list[str]] = []
    if mode == "slides":
        input_specs.append(["-f", "lavfi", "-i", _gradient_source(width, height, fps, duration)])
    else:
        input_specs.append(["-f", "lavfi", "-i",
                            f"color=c={background}:s={width}x{height}:r={fps}:d={duration:.3f}"])
    input_specs.append(["-i", str(segment.audio)])
    if mode != "slides" and episode_seconds > 0:
        # Solid sources for the moving progress bar of the waveform layout.
        input_specs.append(["-f", "lavfi", "-i",
                            f"color=c=0x38BDF8:s={track_w}x5:r={fps}:d={duration:.3f}"])
        input_specs.append(["-f", "lavfi", "-i",
                            f"color=c={background}:s={track_w}x5:r={fps}:d={duration:.3f}"])
    idx_image = -1
    if mode == "slides" and segment.slide:
        idx_image = len(input_specs)
        input_specs.append(["-loop", "1", "-framerate", str(fps),
                            "-t", f"{duration:.3f}", "-i", str(segment.slide)])
    band_y = int(height * 0.74)
    band_h = height - band_y
    scrim = _scrim_image(band_h) if mode == "slides" else None
    idx_scrim = -1
    if scrim is not None:
        idx_scrim = len(input_specs)
        input_specs.append(["-loop", "1", "-framerate", str(fps),
                            "-t", f"{duration:.3f}", "-i", str(scrim)])

    parts: list[str] = []
    source = ""

    if mode == "slides":
        # --- full-screen slide ------------------------------------------- #
        parts.append("[0:v]format=rgba[base]")
        if idx_image >= 0:
            zoom = _ken_burns(elapsed_before, episode_seconds, fps)
            parts.append(
                f"[{idx_image}:v]scale={int(width * 1.5)}:{int(height * 1.5)}:flags=lanczos,"
                f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d=1:s={width}x{height}:fps={fps},format=rgba[img]"
            )
            parts.append("[base][img]overlay=0:0[ov]")
            current = "[ov]"
        else:
            current = "[base]"
        # A soft scrim keeps the captions readable over any picture.
        if idx_scrim >= 0:
            parts.append(f"[{idx_scrim}:v]scale={width}:{band_h}[ramp]")
            parts.append(f"{current}[ramp]overlay=x=0:y={band_y}[scr]")
        else:
            parts.append(
                f"{current}drawbox=x=0:y={band_y}:w={width}:h={band_h}:"
                f"color=0x000000@0.45:t=fill[scr]"
            )
        current = "[scr]"
        parts.append(
            f"{current}drawbox=x=0:y=0:w={width}:h=5:color={segment.color}@0.92:t=fill[acc]"
        )
        current = "[acc]"
        # Progress bar: drawbox evaluates its width per frame, so the bar
        # grows with the episode clock instead of needing a second overlay.
        parts.append(
            f"{current}drawbox=x={track_x}:y={track_y}:w={track_w}:h=5:"
            f"color=0xFFFFFF@0.22:t=fill[trk]"
        )
        current = "[trk]"
        if episode_seconds > 0:
            parts.append(
                f"{current}drawbox=x={track_x}:y={track_y}:"
                f"w='{track_w}*min(1,max(0,({elapsed_before:.3f}+t)/{episode_seconds:.3f}))':"
                f"h=5:color=0x38BDF8@0.95:t=fill[pb]"
            )
            current = "[pb]"
        source = current
    else:
        # --- animated waveform -------------------------------------------- #
        wave_w = int(width * 0.84)
        wave_h = 260
        wave_y = int(height * 0.62) - wave_h // 2

        parts.append(
            f"[0:v]format=rgba,"
            f"drawbox=x=0:y=0:w={width}:h=7:color={segment.color}@0.95:t=fill,"
            f"drawbox=x={track_x}:y={track_y}:w={track_w}:h=5:color=0x1E2D49@1:t=fill[bg]"
        )
        if episode_seconds > 0:
            # Animated progress fill: overlay the accent bar, then slide a
            # background-coloured cover across it. `overlay` evaluates x per
            # frame, which is what makes the bar move.
            parts.append("[2:v]format=rgba[acc]")
            parts.append(f"[bg][acc]overlay=x={track_x}:y={track_y}[b1]")
            parts.append("[3:v]format=rgba[cov]")
            parts.append(
                f"[b1][cov]overlay=x='{track_x}+{track_w}*({elapsed_before:.3f}+t)"
                f"/{episode_seconds:.3f}':y={track_y}[b2]"
            )
            bg_label = "[b2]"
        else:
            bg_label = "[bg]"

        parts.append(
            f"[1:a]showwaves=s={wave_w}x{wave_h}:mode=cline:colors={segment.color}:rate={fps}:"
            f"draw=full,format=rgba[wv]"
        )
        parts.append(f"{bg_label}[wv]overlay=x=(W-w)/2:y={wave_y}[ov]")
        parts.append("[ov]format=yuv420p[bgv]")
        source = "[bgv]"

    # --- text -------------------------------------------------------------- #
    if font:
        f = _escape_filter_path(Path(font))
        caption_size = max(18, int(width * 0.021))
        caption_y = int(height * 0.828)
        label = source.strip("[]")

        if segment.title_card:
            # Opening card: the episode title and who is speaking.
            title_file = _text_file(_fit(title, 52))
            temp_files.append(title_file)
            parts.append(
                f"[{label}]drawtext=fontfile='{f}':textfile='{_escape_filter_path(title_file)}':"
                f"fontsize={int(width * 0.050)}:fontcolor=0xF8FAFC:borderw=2:"
                f"bordercolor=0x000000@0.5:x=(w-text_w)/2:y={int(height * 0.33)}[t1]"
            )
            label = "t1"
            if segment.hosts:
                hosts_file = _text_file(_fit(" · ".join(segment.hosts), 60))
                temp_files.append(hosts_file)
                parts.append(
                    f"[{label}]drawtext=fontfile='{f}':textfile='{_escape_filter_path(hosts_file)}':"
                    f"fontsize={int(width * 0.028)}:fontcolor=0xEC4899:borderw=2:"
                    f"bordercolor=0x000000@0.5:x=(w-text_w)/2:y={int(height * 0.50)}[t2]"
                )
                label = "t2"
            source = f"[{label}]"
        else:
            title_file = _text_file(_fit(title, 70))
            temp_files.append(title_file)
            speaker_file = _text_file(_fit(segment.label or segment.speaker, 32))
            temp_files.append(speaker_file)
            box = ":box=1:boxcolor=0x0B1220@0.55:boxborderw=12" if mode == "slides" else ""
            parts.append(
                f"[{label}]drawtext=fontfile='{f}':textfile='{_escape_filter_path(title_file)}':"
                f"fontsize={int(width * 0.040)}:fontcolor=0xF8FAFC:borderw=2:"
                f"bordercolor=0x000000@0.55{box}:x=(w-text_w)/2:y={int(height * 0.12)}[t1]"
            )
            label = "t1"
            parts.append(
                f"[{label}]drawtext=fontfile='{f}':textfile='{_escape_filter_path(speaker_file)}':"
                f"fontsize={int(width * 0.030)}:fontcolor={segment.color}:borderw=2:"
                f"bordercolor=0x000000@0.5{box}:x=(w-text_w)/2:y={int(height * 0.245)}[t2]"
            )
            label = "t2"

        # One caption line at a time, each visible only while it is spoken.
        for index, line in enumerate(segment.subtitles):
            text = clean_caption_text(line.text)
            if not text:
                continue
            line_file = _text_file(text)
            temp_files.append(line_file)
            name = f"c{index}"
            parts.append(
                f"[{label}]drawtext=fontfile='{f}':"
                f"textfile='{_escape_filter_path(line_file)}':fontsize={caption_size}:"
                f"fontcolor=0xF8FAFC:borderw=2:bordercolor=0x000000@0.65:"
                f"x=(w-text_w)/2:y={caption_y}:"
                f"enable='between(t,{line.begin_ms / 1000.0:.3f},{line.end_ms / 1000.0:.3f})'[{name}]"
            )
            label = name
        source = f"[{label}]"
    else:
        source = source

    if mode == "slides":
        # The slide chain is built in RGBA; hand the encoder a YUV frame.
        parts.append(f"{source}format=yuv420p[outv]")
        source = "[outv]"

    cmd = [FFMPEG, "-y"]
    for spec in input_specs:
        cmd += spec
    cmd += ["-filter_complex", ";".join(parts),
            "-map", source, "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-shortest", str(out_path)]
    try:
        _run(cmd)
    finally:
        for path in temp_files:
            path.unlink(missing_ok=True)
    return duration


def concat_videos(videos: list[Path], out_path: Path, workdir: Path | None = None) -> Path:
    videos = [v for v in videos if v and v.exists()]
    if not videos:
        raise MediaError("No video segments were produced.")
    tmpdir = workdir or Path(tempfile.mkdtemp(prefix="podcaster_vid_"))
    tmpdir.mkdir(parents=True, exist_ok=True)
    listing = tmpdir / "_vconcat.txt"
    listing.write_text("\n".join(f"file '{v.resolve().as_posix()}'" for v in videos), encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
           "-c", "copy", "-movflags", "+faststart", str(out_path)]
    _run(cmd)
    return out_path


def render_video(
    segments: list[VideoSegment],
    out_path: Path,
    *,
    title: str,
    font_path: str | None = None,
    workdir: Path | None = None,
    progress: ProgressFn | None = None,
    mode: str = "waveform",
) -> Path:
    """Render every segment and stitch the clips together."""
    tmpdir = workdir or Path(tempfile.mkdtemp(prefix="podcaster_vid_"))
    tmpdir.mkdir(parents=True, exist_ok=True)

    durations: list[float] = []
    for segment in segments:
        value = probe_duration(segment.audio)
        durations.append(value if value > 0 else 2.0)
    total = sum(durations)

    parts: list[Path] = []
    elapsed = 0.0
    label = "slide" if mode == "slides" else "waveform"
    for i, segment in enumerate(segments):
        if progress:
            progress(f"Rendering {label} {i + 1}/{len(segments)}", (i + 0.5) / len(segments))
        part = tmpdir / f"_seg_{i:04d}.mp4"
        render_segment_video(
            segment, part, title=title,
            font_path=font_path,
            elapsed_before=elapsed,
            episode_seconds=total,
            mode=mode,
        )
        elapsed += durations[i]
        parts.append(part)
    if progress:
        progress("Muxing video", 0.98)
    return concat_videos(parts, out_path, workdir=tmpdir)


def render_waveform_video(
    segments: list[VideoSegment],
    out_path: Path,
    *,
    title: str,
    font_path: str | None = None,
    workdir: Path | None = None,
    progress: ProgressFn | None = None,
) -> Path:
    return render_video(segments, out_path, title=title, font_path=font_path,
                        workdir=workdir, progress=progress, mode="waveform")
