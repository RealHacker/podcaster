"""Slide planning for video exports.

The episode script is broken into illustrated slides. An LLM decides what each
slide should show; a fixed style block is appended to every prompt so the
whole deck looks like one series rather than a pile of unrelated pictures.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .images import ImageClientProtocol, ImageError, ImageSafetyError
from .llm import LLMClient

# Roughly how much speech one slide should cover.
SECONDS_PER_SLIDE = 25.0
MAX_SLIDES = 20
MIN_SLIDES = 1

# How long the opening title card stays on screen.
TITLE_CARD_SECONDS = 6.0

# Appended to every prompt. Keeping this byte-identical across the deck is
# what makes the slides look like a single series.
STYLE_SUFFIX = (
    "Podcast video slide illustration, part of one cohesive illustrated series. "
    "Modern editorial flat-vector style with soft gradient shading and clean geometric forms; "
    "a restrained palette of deep navy, warm amber and magenta accents; subtle film grain; "
    "soft directional lighting; simple uncluttered composition with generous negative space; "
    "16:9 landscape. "
    "Absolutely no text, letters, numbers, captions, logos, watermarks, signatures, "
    "charts with labels or user-interface elements."
)

Logger = Callable[[str, str], None]
ProgressFn = Callable[[str, float], None]


@dataclass
class Slide:
    index: int
    title: str = ""
    prompt: str = ""
    turn_indexes: list[int] = field(default_factory=list)
    image: Path | None = None
    seconds: float = 0.0


@dataclass
class TurnInfo:
    index: int
    speaker: str
    text: str
    seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #

def build_image_prompt(subject: str) -> str:
    """Combine a slide's subject with the series style block."""
    subject = re.sub(r"\s+", " ", (subject or "").strip())
    subject = subject[:700]
    if not subject:
        subject = "An abstract editorial illustration evoking the episode topic"
    return f"{subject}\n\n{STYLE_SUFFIX}"


def soften_prompt(subject: str) -> str:
    """A blunter, safer subject for prompts the content filter rejects."""
    head = re.split(r"[,;:.]", subject or "")[0].strip()
    head = re.sub(r"\s+", " ", head)[:220]
    if not head:
        head = "an abstract concept"
    return f"A calm, simple editorial illustration of {head}"


def target_slide_count(total_seconds: float, per_slide: float = SECONDS_PER_SLIDE) -> int:
    """How many slides an episode of this length should have."""
    if total_seconds <= 0:
        return MIN_SLIDES
    return max(MIN_SLIDES, min(MAX_SLIDES, int(round(total_seconds / per_slide))))


# --------------------------------------------------------------------------- #
# Plan parsing
# --------------------------------------------------------------------------- #

def _turn_refs(value: Any) -> list[int]:
    """Read turn numbers from whatever shape the model produced."""
    out: list[int] = []

    def add(item: Any) -> None:
        try:
            number = int(str(item).strip())
        except (TypeError, ValueError):
            return
        if number >= 0 and number not in out:
            out.append(number)

    if value is None:
        return out
    if isinstance(value, (int, float)):
        add(value)
        return out
    if isinstance(value, str):
        for part in re.split(r"[,\s]+", value.strip()):
            if not part:
                continue
            match = re.match(r"^(\d+)\s*[-~—]\s*(\d+)$", part)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                if end < start:
                    start, end = end, start
                for number in range(start, end + 1):
                    add(number)
            else:
                add(part)
        return out
    if isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(_turn_refs(item))
    return out


def parse_plan(raw: str) -> list[dict[str, Any]]:
    """Pull the slide list out of the model's reply."""
    text = (raw or "").strip()
    if not text:
        return []
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()

    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                data = None
        if data is None:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    data = None

    if isinstance(data, dict):
        items = data.get("slides") or data.get("slide_plan") or data.get("plan") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []

    slides: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or item.get("image_prompt")
                     or item.get("description") or item.get("visual") or "").strip()
        if not prompt:
            continue
        title = str(item.get("title") or item.get("label") or item.get("name") or "").strip()
        turns = _turn_refs(item.get("turns") or item.get("turn")
                           or item.get("turn_indexes") or item.get("segment"))
        slides.append({"title": title[:120], "prompt": prompt[:900], "turns": turns})
    return slides


# --------------------------------------------------------------------------- #
# Turn assignment
# --------------------------------------------------------------------------- #

def assign_turns(items: list[dict[str, Any]], turn_count: int,
                 durations: list[float] | None = None,
                 skip_turns: set[int] | None = None) -> list[Slide]:
    """Give every turn to exactly one slide.

    Turns the model left out are attached to the slide whose turns sit
    closest to them; if the model gave no references at all, the turns are
    spread across the slides in proportion to how long they take to speak.
    """
    slides = [Slide(index=i, title=it.get("title", ""), prompt=it.get("prompt", ""),
                    turn_indexes=list(it.get("turns") or []))
              for i, it in enumerate(items)]
    if not slides:
        return []

    skip = skip_turns or set()
    allowed = [t for t in range(turn_count) if t not in skip]
    if not allowed:
        return []

    buckets: list[set[int]] = [set() for _ in slides]
    for slide, bucket in zip(slides, buckets):
        for turn in slide.turn_indexes:
            if turn in allowed:
                bucket.add(turn)
        slide.turn_indexes = []

    covered = set().union(*buckets) if buckets else set()
    missing = [t for t in allowed if t not in covered]

    if missing:
        if covered:
            for turn in missing:
                best = 0
                best_distance: float | None = None
                for i, bucket in enumerate(buckets):
                    distance = (min(abs(turn - t) for t in bucket) if bucket else 10 ** 9)
                    if best_distance is None or distance < best_distance:
                        best, best_distance = i, distance
                buckets[best].add(turn)
        else:
            _distribute_by_duration(buckets, allowed, durations or [])

    for slide, bucket in zip(slides, buckets):
        slide.turn_indexes = sorted(bucket)

    result = [s for s in slides if s.turn_indexes]
    for i, slide in enumerate(result):
        slide.index = i
    return result


def _distribute_by_duration(buckets: list[set[int]], allowed: list[int],
                            durations: list[float]) -> None:
    """Spread the turns over the slides so each one gets a similar share."""
    weights = [max(0.1, durations[t]) if t < len(durations) else 1.0 for t in allowed]
    total = sum(weights) or float(len(allowed))
    per_slide = total / len(buckets)

    slide_index = 0
    cumulative = 0.0
    for turn, weight in zip(allowed, weights):
        while slide_index < len(buckets) - 1 and cumulative >= per_slide * (slide_index + 1):
            slide_index += 1
        buckets[slide_index].add(turn)
        cumulative += weight


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

def _turn_listing(turns: list[TurnInfo]) -> str:
    lines = []
    for turn in turns:
        text = re.sub(r"\s+", " ", turn.text or "").strip()
        lines.append(f"[{turn.index}] ({turn.seconds:.0f}s) {turn.speaker}: {text[:260]}")
    return "\n".join(lines)


def _plan_prompt(turns: list[TurnInfo], count: int, language: str, title: str,
                 total_seconds: float) -> str:
    listing = _turn_listing(turns)
    if language == "zh":
        return (
            f"你是一档播客视频的视觉总监。下面这期节目被拆成了编号的台词片段，并标注了每段的时长。\n\n"
            f"节目名：{title}\n"
            f"总时长：{total_seconds / 60:.1f} 分钟\n"
            f"幻灯片数量：{count} 张（每张大约覆盖 {SECONDS_PER_SLIDE:.0f} 秒的对白）\n\n"
            f"台词：\n{listing}\n\n"
            "现在规划这些幻灯片：\n"
            "1. 每一段台词必须且只能属于一张幻灯片，把它的编号写进该幻灯片的 `turns` 数组。"
            "第 0 段是片头标题卡，不属于任何幻灯片，请不要写进 `turns`。\n"
            "2. `prompt` 只描述画面本身：主体、场景、构图和氛围。不要把台词写进画面描述。\n"
            "3. `prompt` 一律用英文写，即使节目本身是中文；专有名词保留原文。\n"
            "4. 画面里绝对不能出现文字、字母、数字、字幕、logo、带标注的图表或界面元素。\n"
            "5. 每张幻灯片的主体和视角要不同，但整体视觉风格要保持一致。\n"
            "6. `title` 是这张幻灯片的一个简短标签，只用于记录。\n\n"
            "只输出 JSON：\n"
            '{"slides": [{"title": "...", "prompt": "...", "turns": [1, 2]}]}'
        )
    return (
        f"You are the visual director of a podcast video. The episode below is split into "
        f"numbered spoken turns with their lengths.\n\n"
        f"EPISODE TITLE: {title}\n"
        f"TOTAL LENGTH: {total_seconds / 60:.1f} minutes\n"
        f"NUMBER OF SLIDES: {count} (about {SECONDS_PER_SLIDE:.0f} seconds of speech each)\n\n"
        f"SCRIPT:\n{listing}\n\n"
        "Plan the slides now.\n"
        "1. Every turn must belong to exactly one slide — put the turn numbers in that slide's "
        "\"turns\" array. Turn 0 is the opening title card and belongs to no slide, so leave it out.\n"
        "2. \"prompt\" describes only the PICTURE: the subject, the scene, the composition and the "
        "mood. Describe an image, never a line of dialogue.\n"
        "3. Write every \"prompt\" in English, even when the episode is in another language. Keep "
        "essential proper nouns in their original script.\n"
        "4. The picture must never contain words, letters, numbers, captions, logos, charts with "
        "labels or user-interface elements.\n"
        "5. Give every slide a different subject and viewpoint, but keep them visually consistent — "
        "the same world, palette and medium.\n"
        "6. \"title\" is a short label for the slide, used only for bookkeeping.\n\n"
        "Answer with JSON only:\n"
        '{"slides": [{"title": "...", "prompt": "...", "turns": [1, 2]}]}'
    )


def plan_slides(client: LLMClient, turns: list[TurnInfo], total_seconds: float, *,
                 language: str = "en", title: str = "",
                 progress: ProgressFn | None = None) -> list[Slide]:
    """Ask the model which slides the episode needs, and when they appear."""
    if not turns:
        return []
    count = target_slide_count(total_seconds)
    if progress:
        progress(f"Planning {count} slides", 0.1)

    prompt = _plan_prompt(turns, count, language, title, total_seconds)
    raw = ""
    try:
        raw = client.chat(
            [{"role": "system", "content": "You art-direct podcast videos and answer with JSON."},
             {"role": "user", "content": prompt}],
            temperature=0.55, max_tokens=3000)
    except Exception:
        return []

    items = parse_plan(raw)
    if not items:
        return []

    # More slides than we want is worse than a slightly coarser deck.
    if len(items) > count:
        keep = items[:count]
        spare = [t for item in items[count:] for t in item.get("turns") or []]
        if spare:
            keep[-1]["turns"] = list(keep[-1].get("turns") or []) + spare
        items = keep

    durations = [turn.seconds for turn in turns]
    slides = assign_turns(items, len(turns), durations, skip_turns={0})
    for slide in slides:
        slide.seconds = sum(durations[t] for t in slide.turn_indexes if t < len(durations))
    return slides


# --------------------------------------------------------------------------- #
# Image generation
# --------------------------------------------------------------------------- #

def generate_images(client: ImageClientProtocol, slides: list[Slide], outdir: Path, *,
                    log: Logger | None = None, progress: ProgressFn | None = None) -> list[Slide]:
    """Render one illustration per slide; failed slides are left without an image."""
    outdir.mkdir(parents=True, exist_ok=True)
    for i, slide in enumerate(slides):
        if progress:
            progress(f"Drawing slide {i + 1}/{len(slides)}", i / max(1, len(slides)))
        target = outdir / f"slide_{i + 1:02d}.jpg"
        try:
            result = client.generate(build_image_prompt(slide.prompt), target)
            slide.image = result.path
            continue
        except ImageSafetyError as exc:
            if log:
                log("warn", f"Slide {i + 1} was rejected by the content filter — retrying simply")
            try:
                result = client.generate(build_image_prompt(soften_prompt(slide.prompt)), target)
                slide.image = result.path
                continue
            except ImageError as retry_exc:
                if log:
                    log("warn", f"Slide {i + 1} failed: {retry_exc}")
                continue
        except ImageError as exc:
            if log:
                log("warn", f"Slide {i + 1} failed: {exc}")
            continue
    return slides
