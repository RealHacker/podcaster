"""Subtitle cues: parsing the TTS subtitle payload and laying out one-line captions.

The TTS provider returns a JSON subtitle file with either sentence-level or
word-level timestamps.  Word-level is what we ask for, because it lets us
choose exactly where a caption breaks while keeping the timing accurate.

Captions are laid out as *single lines*: words are grouped until the line is
full, and a sentence that is too long for one line is split across several
lines rather than being allowed to wrap or run off the frame.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# A caption line is sized in half-width units: a CJK glyph counts as two, a
# Latin glyph as one.  64 units is about 830 px at the caption font size, which
# leaves comfortable margins on a 1280 px frame.
MAX_LINE_UNITS = 64

# A pause longer than this starts a new caption line.
GAP_MS = 400.0

# Never show a caption for less than this long, it only flickers.
MIN_LINE_MS = 400.0


@dataclass
class SubtitleCue:
    """One timed chunk of speech."""

    text: str
    begin_ms: float = 0.0
    end_ms: float = 0.0
    start: int = -1          # character offset of the cue in the source text
    end: int = -1


@dataclass
class SubtitleLine:
    """One caption line: short enough to sit on a single line of the video."""

    text: str
    begin_ms: float = 0.0
    end_ms: float = 0.0


# --------------------------------------------------------------------------- #
# Width
# --------------------------------------------------------------------------- #

def is_wide(ch: str) -> bool:
    """True for characters that are about twice as wide as a Latin letter."""
    code = ord(ch)
    return (
        0x1100 <= code <= 0x115F        # Hangul Jamo
        or 0x2E80 <= code <= 0x303E     # CJK radicals .. punctuation
        or 0x3041 <= code <= 0x33FF     # Kana, CJK compatibility
        or 0x3400 <= code <= 0x4DBF     # CJK extension A
        or 0x4E00 <= code <= 0x9FFF     # CJK unified ideographs
        or 0xA000 <= code <= 0xA4CF     # Yi
        or 0xAC00 <= code <= 0xD7A3     # Hangul syllables
        or 0xF900 <= code <= 0xFAFF     # CJK compatibility ideographs
        or 0xFE30 <= code <= 0xFE6F     # CJK compatibility forms
        or 0xFF00 <= code <= 0xFF60     # Fullwidth forms
        or 0xFFE0 <= code <= 0xFFE6
        or 0x20000 <= code <= 0x3FFFD   # CJK extensions B+
    )


def text_width(text: str) -> int:
    """Width of a string in half-width units."""
    width = 0
    for ch in text:
        width += 2 if is_wide(ch) else 1
    return width


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def _segments(payload: Any) -> list[dict[str, Any]]:
    """Normalise the subtitle file into a list of segment dicts."""
    if isinstance(payload, list):
        return [s for s in payload if isinstance(s, dict)]
    if isinstance(payload, dict):
        for key in ("subtitles", "subtitle", "segments", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [s for s in value if isinstance(s, dict)]
    return []


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return -1
    if isinstance(value, int):
        return value
    return -1


def parse_subtitle_payload(payload: Any, source_text: str = "") -> list[SubtitleCue]:
    """Turn a MiniMax subtitle file into word-level cues.

    Sentence-level segments (no ``timestamped_words``) are returned as a
    single cue each; word-level segments are exploded into one cue per word
    so that the line layout can pick its own break points.

    ``source_text`` is the text that was sent to the TTS engine.  When it is
    given, each cue is taken from the source by character offset, which keeps
    the caption text byte-identical to what was spoken.
    """
    cues: list[SubtitleCue] = []
    for segment in _segments(payload):
        words = segment.get("timestamped_words") or []
        if isinstance(words, list) and words:
            for word in words:
                if not isinstance(word, dict):
                    continue
                text = str(word.get("word") or "")
                start = _int(word.get("word_begin"))
                end = _int(word.get("word_end"))
                if source_text and 0 <= start < end <= len(source_text):
                    text = source_text[start:end]
                if not text:
                    continue
                cues.append(SubtitleCue(
                    text=text,
                    begin_ms=_num(word.get("time_begin")),
                    end_ms=_num(word.get("time_end")),
                    start=start,
                    end=end,
                ))
            continue
        text = str(segment.get("text") or "").strip()
        if text:
            cues.append(SubtitleCue(
                text=text,
                begin_ms=_num(segment.get("time_begin")),
                end_ms=_num(segment.get("time_end")),
            ))
    return cues


# --------------------------------------------------------------------------- #
# Line layout
# --------------------------------------------------------------------------- #

@dataclass
class _Atom:
    """A group of cues that belong to one word (or one CJK character)."""

    text: str
    begin_ms: float
    end_ms: float
    start: int = -1
    end: int = -1
    tokens: list[SubtitleCue] = field(default_factory=list)
    space_after: bool = False

    @property
    def width(self) -> int:
        return text_width(self.text)


def _atom_from_cue(cue: SubtitleCue) -> _Atom:
    return _Atom(text=cue.text, begin_ms=cue.begin_ms, end_ms=cue.end_ms,
                 start=cue.start, end=cue.end, tokens=[cue])


def _atoms(cues: Iterable[SubtitleCue]) -> list[_Atom]:
    """Group word cues into atoms; whitespace cues mark a word boundary."""
    atoms: list[_Atom] = []
    buf: list[SubtitleCue] = []

    def flush() -> None:
        if not buf:
            return
        atoms.append(_Atom(
            text="".join(c.text for c in buf),
            begin_ms=buf[0].begin_ms,
            end_ms=buf[-1].end_ms,
            start=buf[0].start,
            end=buf[-1].end,
            tokens=list(buf),
        ))
        buf.clear()

    for cue in cues:
        if not cue.text:
            continue
        if not cue.text.strip():
            # A separator: it closes the current word.
            flush()
            if atoms:
                atoms[-1].space_after = True
            continue
        buf.append(cue)
    flush()
    return atoms


def _split_token(token: SubtitleCue, max_units: int, source_text: str) -> list[_Atom]:
    """Break one over-long token into pieces, spreading its time over them."""
    text = token.text
    total = text_width(text) or 1
    span = max(0.0, token.end_ms - token.begin_ms)

    exact = (source_text and token.start >= 0
             and token.start + len(text) <= len(source_text)
             and source_text[token.start:token.start + len(text)] == text)

    chunks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_width = 0
    for index, ch in enumerate(text):
        width = 2 if is_wide(ch) else 1
        if current and current_width + width > max_units:
            chunks.append(current)
            current, current_width = [], 0
        current.append((index, ch))
        current_width += width
    if current:
        chunks.append(current)

    pieces: list[_Atom] = []
    consumed = 0
    for chunk in chunks:
        chunk_width = sum(2 if is_wide(c) else 1 for _, c in chunk)
        begin = token.begin_ms + span * consumed / total
        end = token.begin_ms + span * (consumed + chunk_width) / total
        if exact:
            start = token.start + chunk[0][0]
            end_char = token.start + chunk[-1][0] + 1
        else:
            start = end_char = -1
        pieces.append(_Atom(
            text="".join(c for _, c in chunk),
            begin_ms=begin, end_ms=end,
            start=start, end=end_char,
            tokens=[token],
        ))
        consumed += chunk_width
    return pieces


def _split_atom(atom: _Atom, max_units: int, source_text: str) -> list[_Atom]:
    """Break one over-long word into pieces that each fit the budget."""
    if atom.width <= max_units:
        return [atom]
    if len(atom.tokens) > 1:
        pieces: list[_Atom] = []
        for token in atom.tokens:
            pieces.extend(_split_token(token, max_units, source_text))
        return pieces
    return _split_token(atom.tokens[0], max_units, source_text)


def _needs_space(previous: _Atom, following: _Atom) -> bool:
    """Whether two atoms are separated by a space in the spoken text.

    Some payloads do not include the whitespace tokens themselves; in that
    case the gap is inferred from the script: Latin words are separated, CJK
    characters are not.
    """
    if previous.space_after:
        return True
    if not previous.text or not following.text:
        return False
    if is_wide(previous.text[-1]) or is_wide(following.text[0]):
        return False
    if previous.text[-1] in ".,;:!?)]}、。，！？；：…":
        return False
    return True


def _line_text(group: list[_Atom], source_text: str) -> str:
    """The caption text of a group of atoms."""
    first, last = group[0], group[-1]
    if (source_text and first.start >= 0 and last.end > first.start
            and last.end <= len(source_text)):
        return source_text[first.start:last.end].strip()
    text = ""
    for index, atom in enumerate(group):
        text += atom.text
        if index < len(group) - 1 and _needs_space(atom, group[index + 1]):
            text += " "
    return text.strip()


def build_lines(cues: Iterable[SubtitleCue], source_text: str = "",
                max_units: int = MAX_LINE_UNITS) -> list[SubtitleLine]:
    """Lay cues out as caption lines that each fit on a single line."""
    atoms: list[_Atom] = []
    for atom in _atoms(cues):
        atoms.extend(_split_atom(atom, max_units, source_text))
    if not atoms:
        return []

    groups: list[list[_Atom]] = []
    current: list[_Atom] = []
    current_width = 0
    previous_end = 0.0
    for atom in atoms:
        if current:
            gap = atom.begin_ms - previous_end
            # The space that will separate the two atoms counts towards the
            # width of the line, otherwise the caption can overflow the frame.
            separator = 1 if _needs_space(current[-1], atom) else 0
            if gap > GAP_MS:
                groups.append(current)
                current, current_width = [], 0
            elif current_width + separator + atom.width > max_units:
                groups.append(current)
                current, current_width = [], 0
            else:
                current_width += separator
        current.append(atom)
        current_width += atom.width
        previous_end = atom.end_ms
    if current:
        groups.append(current)

    lines: list[SubtitleLine] = []
    for group in groups:
        text = _line_text(group, source_text)
        if not text:
            continue
        lines.append(SubtitleLine(
            text=text,
            begin_ms=group[0].begin_ms,
            end_ms=group[-1].end_ms,
        ))

    # A caption stays on screen until the next one appears, so a short pause
    # between two lines does not make the text flicker.
    for i, line in enumerate(lines):
        if i + 1 < len(lines):
            line.end_ms = max(line.end_ms, lines[i + 1].begin_ms)
        if line.end_ms - line.begin_ms < MIN_LINE_MS:
            line.end_ms = line.begin_ms + MIN_LINE_MS

    return lines


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

def clean_caption_text(text: str) -> str:
    """Make a caption safe to hand to ffmpeg's drawtext."""
    text = re.sub(r"\s+", " ", (text or "").replace("\u200b", "")).strip()
    return text


def to_srt(lines: Iterable[SubtitleLine]) -> str:
    """Render the lines as an SRT document (handy for debugging)."""
    def stamp(ms: float) -> str:
        total = int(max(0.0, ms))
        hours, rest = divmod(total, 3_600_000)
        minutes, rest = divmod(rest, 60_000)
        seconds, millis = divmod(rest, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    out: list[str] = []
    for index, line in enumerate(lines, start=1):
        out.append(str(index))
        out.append(f"{stamp(line.begin_ms)} --> {stamp(line.end_ms)}")
        out.append(line.text)
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Approximate cue generation (fallback for providers without timed cues)
# --------------------------------------------------------------------------- #

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？；])\s*")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:，；：、])\s*")

MAX_CHUNK_WIDTH = 50


def _split_into_chunks(text: str, max_width: int = MAX_CHUNK_WIDTH) -> list[str]:
    """Split text into caption-sized chunks on sentence/clause boundaries.

    Prefers sentence boundaries but falls back to clause boundaries for long
    sentences.  Very long runs without punctuation are split mid-text.
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    for sentence in sentences:
        if text_width(sentence) <= max_width:
            chunks.append(sentence)
            continue
        clauses = [c.strip() for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]
        if not clauses:
            clauses = [sentence]
        buf = ""
        for clause in clauses:
            if text_width(clause) > max_width:
                if buf:
                    chunks.append(buf)
                    buf = ""
                pos = 0
                while pos < len(clause):
                    piece_chars: list[str] = []
                    piece_width = 0
                    while pos < len(clause) and piece_width < max_width:
                        ch = clause[pos]
                        ch_w = 2 if is_wide(ch) else 1
                        if piece_width + ch_w > max_width and piece_chars:
                            break
                        piece_chars.append(ch)
                        piece_width += ch_w
                        pos += 1
                    if piece_chars:
                        chunks.append("".join(piece_chars))
                continue
            candidate = f"{buf} {clause}".strip() if buf else clause
            if text_width(candidate) <= max_width:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = clause
        if buf:
            chunks.append(buf)

    return chunks


def generate_approximate_cues(
    source_text: str, duration_ms: float
) -> list[SubtitleCue]:
    """Generate approximate subtitle cues when the TTS provider returns none.

    Splits the source text into caption-sized chunks and distributes them
    proportionally across the audio duration based on text width.
    """
    chunks = _split_into_chunks(source_text)
    if not chunks or duration_ms <= 0:
        return []

    total_width = sum(text_width(c) for c in chunks)
    if total_width == 0:
        return []

    cues: list[SubtitleCue] = []
    elapsed = 0.0
    for chunk in chunks:
        chunk_width = text_width(chunk)
        chunk_duration = duration_ms * (chunk_width / total_width)
        cues.append(SubtitleCue(
            text=chunk,
            begin_ms=elapsed,
            end_ms=elapsed + chunk_duration,
        ))
        elapsed += chunk_duration

    return cues
