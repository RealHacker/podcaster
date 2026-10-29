"""Tests for the subtitle cue parsing and the one-line caption layout."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from podcaster.subtitles import (  # noqa: E402
    MAX_LINE_UNITS,
    SubtitleCue,
    build_lines,
    clean_caption_text,
    parse_subtitle_payload,
    text_width,
    to_srt,
)

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  OK   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def words(*items: tuple[str, float, float]) -> list[dict]:
    return [{"word": w, "time_begin": b, "time_end": e} for w, b, e in items]


def test_width() -> None:
    check(text_width("hello") == 5, "latin text counts one unit per character")
    check(text_width("你好") == 4, "CJK text counts two units per character")
    check(text_width("hi 你") == 5, "mixed text is counted per script")


def test_parse_word_level() -> None:
    payload = [{
        "text": "Hello there",
        "time_begin": 0.0,
        "time_end": 1000.0,
        "timestamped_words": words(
            ("Hel", 0.0, 200.0), ("lo", 200.0, 400.0), (" ", 400.0, 500.0),
            ("there", 500.0, 1000.0),
        ),
    }]
    cues = parse_subtitle_payload(payload)
    check(len(cues) == 4, f"word level payload yields one cue per word ({len(cues)})")
    check(cues[0].text == "Hel" and cues[0].begin_ms == 0.0, "the first cue keeps its text and time")


def test_parse_sentence_level() -> None:
    payload = [{"text": "Hello there.", "time_begin": 0.0, "time_end": 1000.0}]
    cues = parse_subtitle_payload(payload)
    check(len(cues) == 1 and cues[0].text == "Hello there.",
          "a sentence level payload yields a single cue")


def test_parse_shapes() -> None:
    check(parse_subtitle_payload({"subtitles": [{"text": "a", "time_begin": 0,
                                                 "time_end": 1}]})[0].text == "a",
          "an object with a subtitles list is understood")
    check(parse_subtitle_payload(None) == [], "a broken payload yields no cues")


def test_lines_fit_one_line() -> None:
    sentence = ("Ambient computing is the idea that the room around you quietly "
                "reacts to what you are doing without ever asking for your attention")
    cues = [SubtitleCue(text=word, begin_ms=i * 100.0, end_ms=i * 100.0 + 90.0)
            for i, word in enumerate(sentence.split(" "))]
    lines = build_lines(cues, sentence)
    check(len(lines) > 1, f"a long sentence is split into several lines ({len(lines)})")
    check(all(text_width(line.text) <= MAX_LINE_UNITS for line in lines),
          "every line fits the single-line width budget")
    check(all(line.text.strip() for line in lines), "no empty lines are produced")
    joined = " ".join(line.text for line in lines)
    check(joined.split() == sentence.split(), "no words are lost or duplicated")


def test_lines_keep_timing() -> None:
    cues = [
        SubtitleCue("Hello", 0.0, 400.0),
        SubtitleCue(" ", 400.0, 450.0),
        SubtitleCue("world", 450.0, 900.0),
    ]
    lines = build_lines(cues)
    check(len(lines) == 1, f"short text stays on one line ({len(lines)})")
    check(lines[0].text == "Hello world", f"the words are joined with spaces ({lines[0].text!r})")
    check(lines[0].begin_ms == 0.0 and lines[0].end_ms == 900.0,
          "the line spans the time of its words")


def test_lines_do_not_flicker() -> None:
    cues = [
        SubtitleCue("One", 0.0, 300.0),
        SubtitleCue(" ", 300.0, 320.0),
        SubtitleCue("Two", 1400.0, 1700.0),
    ]
    lines = build_lines(cues)
    check(len(lines) == 2, f"a pause splits the text into two lines ({len(lines)})")
    check(lines[0].end_ms == 1400.0,
          f"the first line stays visible until the next one appears ({lines[0].end_ms})")


def test_cjk_layout() -> None:
    sentence = ("环境计算是指周围的环境能够感知你的状态并作出回应，"
                "却从不打扰你正在做的事情")
    cues = [SubtitleCue(text=ch, begin_ms=i * 120.0, end_ms=i * 120.0 + 110.0)
            for i, ch in enumerate(sentence)]
    lines = build_lines(cues)
    check(len(lines) > 1, f"a long Chinese sentence is split ({len(lines)})")
    check(all(text_width(line.text) <= MAX_LINE_UNITS for line in lines),
          "every Chinese line fits the width budget")
    check("".join(line.text for line in lines) == sentence,
          "the Chinese text is preserved exactly")


def test_long_word_is_split() -> None:
    long_word = "supercalifragilisticexpialidocious" * 3
    cues = [SubtitleCue(long_word, 0.0, 3000.0)]
    lines = build_lines(cues)
    check(len(lines) > 1, f"an over-long word is split ({len(lines)})")
    check(all(text_width(line.text) <= MAX_LINE_UNITS for line in lines),
          "the split pieces fit the width budget")
    check("".join(line.text for line in lines) == long_word,
          "the split pieces keep every character")


def test_clean_text() -> None:
    check(clean_caption_text("  a\nb  ") == "a b", "line breaks collapse to spaces")
    check(clean_caption_text("a\u200bb") == "ab", "zero width spaces are removed")


def test_srt() -> None:
    lines = build_lines([SubtitleCue("Hi", 0.0, 1500.0)])
    srt = to_srt(lines)
    check("00:00:00,000 --> 00:00:01,500" in srt, "the SRT timestamp is formatted")


if __name__ == "__main__":
    print("=== width")
    test_width()
    print("\n=== parsing")
    test_parse_word_level()
    test_parse_sentence_level()
    test_parse_shapes()
    print("\n=== layout")
    test_lines_fit_one_line()
    test_lines_keep_timing()
    test_lines_do_not_flicker()
    test_cjk_layout()
    test_long_word_is_split()
    print("\n=== text helpers")
    test_clean_text()
    test_srt()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("ALL SUBTITLE CHECKS PASSED")
