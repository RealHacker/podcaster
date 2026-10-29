"""Tests for the slide planner, the style block and the illustrated video layout."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from podcaster import media as media_mod  # noqa: E402
from podcaster import slides as slides_mod  # noqa: E402
from podcaster.subtitles import SubtitleLine  # noqa: E402
from podcaster.slides import (  # noqa: E402
    MAX_SLIDES,
    STYLE_SUFFIX,
    Slide,
    TurnInfo,
    assign_turns,
    build_image_prompt,
    parse_plan,
    plan_slides,
    soften_prompt,
    target_slide_count,
)

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  OK   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


class StubLLM:
    """Stands in for the chat client so the planner can be tested offline."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def chat(self, messages, temperature=None, max_tokens=None, retries=4):
        self.prompts.append(messages[-1]["content"])
        return self.reply


def test_slide_count() -> None:
    check(target_slide_count(0) == 1, "an empty episode still gets one slide")
    check(target_slide_count(25) == 1, "25 seconds of speech is one slide")
    check(target_slide_count(150) == 6, "150 seconds of speech is six slides")
    check(target_slide_count(3600) == MAX_SLIDES, "the slide count is capped")
    check(target_slide_count(5) == 1, "a very short episode is not given zero slides")


def test_style_prompt() -> None:
    first = build_image_prompt("A quiet library at dawn")
    second = build_image_prompt("A crowded market at noon")
    check("A quiet library at dawn" in first, "the subject is kept in the prompt")
    check(STYLE_SUFFIX in first and STYLE_SUFFIX in second,
          "every slide gets the same style block")
    check("no text" in first.lower(), "the style block forbids text in the picture")
    check(len(build_image_prompt("x" * 4000)) <= 1500, "the prompt fits the API limit")
    check(len(STYLE_SUFFIX) < 700, "the style block stays compact")


def test_soften_prompt() -> None:
    softened = soften_prompt("A market, with people and a sign")
    check("A market" in softened, "the softened prompt keeps the first clause")
    check("sign" not in softened, "the softened prompt drops the detail that was flagged")


def test_parse_plan() -> None:
    payload = ('{"slides": [{"title": "Intro", "prompt": "A wide shot of a city", '
               '"turns": [1, 2]}, {"title": "Middle", "prompt": "A close-up of a hand", '
               '"turns": "3-5"}]}')
    items = parse_plan(payload)
    check(len(items) == 2, "two slides are parsed")
    check(items[0]["turns"] == [1, 2], "a list of turn numbers is read")
    check(items[1]["turns"] == [3, 4, 5], "a turn range is expanded")

    fenced = parse_plan("Here you go:\n```json\n" + payload + "\n```")
    check(len(fenced) == 2, "a fenced code block is parsed")

    bare = parse_plan('[{"prompt": "A lone tree"}]')
    check(len(bare) == 1 and bare[0]["turns"] == [], "a bare array is parsed")

    messy = parse_plan('Sure! {"slides": [{"prompt": "A red door", "turns": 4}]} hope that helps')
    check(len(messy) == 1 and messy[0]["turns"] == [4], "prose around the JSON is tolerated")

    check(parse_plan("") == [], "an empty reply parses to nothing")
    check(parse_plan('{"slides": [{"title": "no prompt"}]}') == [],
          "a slide without a prompt is dropped")


def test_assign_turns() -> None:
    items = [{"title": "a", "prompt": "p", "turns": [1, 2]},
             {"title": "b", "prompt": "p", "turns": [5]}]
    slides = assign_turns(items, turn_count=8, durations=[5] * 8, skip_turns={0})
    covered = [t for slide in slides for t in slide.turn_indexes]
    check(sorted(covered) == [1, 2, 3, 4, 5, 6, 7],
          "every turn but the title card lands on exactly one slide")
    check(0 not in covered, "the opening turn is left to the title card")
    check(len(covered) == len(set(covered)), "no turn is used twice")
    check(slides[0].turn_indexes == [1, 2, 3, 4], "a gap is filled with the nearest slide")

    # No references at all -> the turns are spread by speaking time.
    plain = [{"title": "a", "prompt": "p"}, {"title": "b", "prompt": "p"}]
    spread = assign_turns(plain, turn_count=6, durations=[10, 10, 10, 10, 10, 10])
    check(sum(len(slide.turn_indexes) for slide in spread) == 6,
          "every turn is placed when the model gives no references")
    check(all(slide.turn_indexes for slide in spread), "no slide is left empty")


def test_assign_turns_skips_everything() -> None:
    slides = assign_turns([{"title": "a", "prompt": "p", "turns": [0]}],
                          turn_count=1, durations=[5], skip_turns={0})
    check(slides == [], "nothing is assigned when only the title card turn exists")


def test_plan_slides() -> None:
    turns = [TurnInfo(index=i, speaker="Marcus", text=f"Line number {i} of the script.",
                      seconds=20.0) for i in range(6)]
    reply = ('{"slides": ['
             '{"title": "One", "prompt": "A wide street", "turns": [1, 2]},'
             '{"title": "Two", "prompt": "A quiet room", "turns": [3, 4, 5]}]}')
    client = StubLLM(reply)
    slides = plan_slides(client, turns, total_seconds=120.0, title="Test episode")
    check(len(slides) == 2, "the planned slides come back")
    check(all(slide.prompt for slide in slides), "every slide has a prompt")
    covered = sorted(t for slide in slides for t in slide.turn_indexes)
    check(covered == [1, 2, 3, 4, 5], "the plan covers the script but not the title card")
    check("turn 0" in client.prompts[0].lower(),
          "the planner is told to leave out turn 0")
    check("120" in client.prompts[0] or "2.0" in client.prompts[0],
          "the planner is told how long the episode is")

    broken = StubLLM("I could not do that.")
    check(plan_slides(broken, turns, total_seconds=120.0) == [],
          "a useless reply produces no slides")


def test_plan_slides_caps_the_count() -> None:
    turns = [TurnInfo(index=i, speaker="A", text=f"line {i}", seconds=20.0) for i in range(40)]
    reply = ('{"slides": [' + ",".join(
        '{"title": "%d", "prompt": "scene %d", "turns": [%d]}' % (i, i, i)
        for i in range(40)) + "]}")
    slides = plan_slides(StubLLM(reply), turns, total_seconds=900.0)
    check(len(slides) <= MAX_SLIDES, "an over-long plan is trimmed to the target count")
    covered = [t for slide in slides for t in slide.turn_indexes]
    check(len(covered) == len(set(covered)), "trimming does not duplicate a turn")


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args,
                   check=True, capture_output=True)


def test_render_slides_layout() -> None:
    if not shutil.which("ffmpeg"):
        print("  SKIP ffmpeg is not available")
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        image = work / "slide.jpg"
        audio = work / "clip.mp3"
        _ffmpeg(["-f", "lavfi", "-i", "color=c=0x3366AA:s=1280x720",
                 "-frames:v", "1", str(image)])
        _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                 "-c:a", "libmp3lame", str(audio)])

        segment = media_mod.VideoSegment(
            audio=audio, speaker="Marcus", color="0x38BDF8", label="Marcus",
            subtitles=[SubtitleLine(text="Hello there", begin_ms=0, end_ms=2500)],
            slide=image,
        )
        out = work / "slide.mp4"
        media_mod.render_segment_video(segment, out, title="The Episode",
                                       episode_seconds=3.0, mode="slides")
        check(out.exists() and out.stat().st_size > 5000, "a slide segment renders")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True)
        check(probe.stdout.strip() == "1280,720", "the slide fills a 1280x720 frame")

        card = media_mod.VideoSegment(
            audio=audio, speaker="Marcus", color="0x38BDF8", label="Marcus",
            title_card=True, hosts=["Marcus", "Elena"],
        )
        card_out = work / "card.mp4"
        media_mod.render_segment_video(card, card_out, title="The Episode",
                                       episode_seconds=3.0, mode="slides")
        check(card_out.exists() and card_out.stat().st_size > 5000, "the title card renders")


def test_waveform_layout_still_works() -> None:
    """The slides work must not have broken the original waveform layout."""
    if not shutil.which("ffmpeg"):
        print("  SKIP ffmpeg is not available")
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        audio = work / "clip.mp3"
        _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                 "-c:a", "libmp3lame", str(audio)])
        segment = media_mod.VideoSegment(
            audio=audio, speaker="Marcus", color="0x38BDF8", label="Marcus",
            subtitles=[SubtitleLine(text="Waveform check", begin_ms=0, end_ms=2500)],
        )
        out = work / "wave.mp4"
        media_mod.render_segment_video(segment, out, title="The Episode",
                                       episode_seconds=3.0, mode="waveform")
        check(out.exists() and out.stat().st_size > 5000, "a waveform segment still renders")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True)
        check(probe.stdout.strip() == "1280,720", "the waveform layout is still 1280x720")


def test_trim_audio() -> None:
    if not shutil.which("ffmpeg"):
        print("  SKIP ffmpeg is not available")
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        audio = work / "clip.mp3"
        _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=10",
                 "-c:a", "libmp3lame", str(audio)])
        cut = media_mod.trim_audio(audio, 0.0, 4.0, work / "cut.mp3")
        duration = media_mod.probe_duration(cut)
        check(3.7 <= duration <= 4.3, f"the title card is trimmed to 4 seconds ({duration:.2f})")
        tail = media_mod.trim_audio(audio, 4.0, 6.0, work / "tail.mp3")
        check(media_mod.probe_duration(tail) > 5.5, "the rest of the intro survives the cut")


if __name__ == "__main__":
    for name, function in sorted(list(globals().items())):
        if name.startswith("test_") and callable(function):
            print(f"\n-- {name}")
            function()
    print()
    if failures:
        print(f"{len(failures)} FAILED:")
        for item in failures:
            print("   -", item)
        sys.exit(1)
    print("all slide checks passed")
