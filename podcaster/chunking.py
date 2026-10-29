"""Splitting long source material into digestible chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    text: str
    char_start: int = 0
    char_end: int = 0

    @property
    def chars(self) -> int:
        return len(self.text)


def split_into_chunks(text: str, target_chars: int = 6000, overlap: int = 200) -> list[Chunk]:
    """Paragraph-aware chunking.

    Paragraphs are accumulated until ``target_chars`` is reached; oversized
    paragraphs are split on sentence boundaries, and as a last resort on
    hard character slices so a single giant block never breaks the flow.
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    def split_long_paragraph(para: str) -> list[str]:
        if len(para) <= target_chars * 1.5:
            return [para]
        sentences = re.split(r"(?<=[.!?。！？；;])\s+", para)
        pieces: list[str] = []
        buf = ""
        for sentence in sentences:
            if len(sentence) > target_chars:
                if buf:
                    pieces.append(buf)
                    buf = ""
                for i in range(0, len(sentence), target_chars):
                    pieces.append(sentence[i:i + target_chars])
                continue
            if len(buf) + len(sentence) + 1 > target_chars and buf:
                pieces.append(buf)
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf:
            pieces.append(buf)
        return pieces

    flat: list[str] = []
    for para in paragraphs:
        flat.extend(split_long_paragraph(para))

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    cursor = 0

    for para in flat:
        if buf_len + len(para) > target_chars and buf:
            body = "\n\n".join(buf)
            chunks.append(Chunk(len(chunks), body, cursor, cursor + len(body)))
            cursor = text.find(body, cursor)
            cursor = cursor + len(body) - overlap if cursor >= 0 else cursor
            buf = []
            buf_len = 0
        buf.append(para)
        buf_len += len(para) + 2

    if buf:
        body = "\n\n".join(buf)
        chunks.append(Chunk(len(chunks), body, cursor, cursor + len(body)))

    return chunks


def estimate_speech_units(text: str, language: str) -> int:
    """Rough count of spoken units (words for English, characters for Chinese)."""
    if language == "zh":
        han = len(re.findall(r"[\u4e00-\u9fff]", text))
        other = len(re.findall(r"[A-Za-z]+", text))
        return han + other
    return len(re.findall(r"[A-Za-z0-9']+", text))


# Measured against the live MiniMax TTS endpoint: 159 words/min for English
# and 257 Chinese characters/min at speed 1.0.
SPEECH_UNITS_PER_MINUTE = {"en": 157.0, "zh": 252.0}


def units_to_minutes(units: int, language: str) -> float:
    """Estimated spoken minutes for a given number of units."""
    rate = SPEECH_UNITS_PER_MINUTE.get(language, SPEECH_UNITS_PER_MINUTE["en"])
    return units / rate
