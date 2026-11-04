"""Podcast script generation.

The flow is: plan → write segment by segment → parse → assemble.
Planning keeps long sources coherent; writing per chunk keeps each LLM call
inside a sane context window.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .chunking import SPEECH_UNITS_PER_MINUTE, estimate_speech_units, units_to_minutes
from .llm import LLMClient

ProgressFn = Callable[[str, float], None]

STYLES: dict[str, dict[str, Any]] = {
    "narrative": {
        "label": "Narrative",
        "en": "A story-driven single-narrator format with cinematic scene-setting, "
              "vivid detail and a clear arc. The host speaks to the audience, weaving "
              "in short asides from a co-host.",
    },
    "interview": {
        "label": "Interview",
        "en": "A classic interview: one host asks sharp, curious questions; the guest "
              "answers with personal experience and specifics. Follow-ups build on the "
              "previous answer.",
    },
    "chitchat": {
        "label": "Chitchat",
        "en": "Two friends chatting casually about the topic — light banter, reactions, "
              "short interruptions and warm humour, without sounding scripted.",
    },
    "deep_dive": {
        "label": "Deep dive",
        "en": "An analytical deep dive: two hosts reason through the material, cite "
              "concrete details from the source, debate nuance and explain mechanisms "
              "carefully.",
    },
    "news": {
        "label": "News",
        "en": "A crisp news briefing: the lead, the key facts in context, and a short "
              "sign-off. Tight, factual, no filler.",
    },
    "educational": {
        "label": "Educational",
        "en": "A clear explainer: one host teaches, the other asks the questions a "
              "listener would ask. Concepts are unpacked step by step.",
    },
    "storytelling": {
        "label": "Storytelling",
        "en": "Character-driven storytelling — people, moments and consequences, told "
              "with warmth and a strong sense of place.",
    },
    "comedy": {
        "label": "Comedy",
        "en": "A comedic podcast: playful banter, jokes that land off the subject "
              "matter, and a genuine comic arc. Never forced.",
    },
}

STYLE_ORDER = list(STYLES.keys())

ROLE_BY_STYLE = {
    "interview": ("host", "guest"),
    "educational": ("teacher", "curious listener"),
    "news": ("anchor", "co-anchor"),
}


@dataclass
class Host:
    name: str
    gender: str = "female"
    persona: str = ""
    voice_id: str = ""
    persona_en: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "gender": self.gender, "persona": self.persona,
                "voice_id": self.voice_id}


@dataclass
class Turn:
    speaker: str
    text: str
    segment: int = 0
    index: int = 0

    @property
    def units(self) -> int:
        return estimate_speech_units(self.text, "en")


@dataclass
class Script:
    title: str
    turns: list[Turn] = field(default_factory=list)
    hosts: list[Host] = field(default_factory=list)
    style: str = "narrative"
    language: str = "en"
    outline: list[str] = field(default_factory=list)
    target_minutes: float = 5.0
    notes: str = ""

    @property
    def text(self) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)

    def units(self, language: str) -> int:
        return sum(estimate_speech_units(t.text, language) for t in self.turns)


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #

def _language_block(language: str) -> str:
    if language == "zh":
        return (
            "语言：简体中文。所有台词必须用简体中文书写。\n"
            "不要使用拼音，不要中英混排到读起来别扭；专有名词可保留英文原名。\n"
        )
    return (
        "Language: English. Write every spoken line in natural, spoken English — "
        "contractions, short sentences, no jargon dumps, no bullet lists.\n"
    )


def _format_rule(language: str) -> str:
    if language == "zh":
        return (
            "格式硬性要求：\n"
            "1. 每一行都是一句台词，格式严格为 `说话人姓名: 台词内容`\n"
            "2. 说话人姓名必须且只能是下面给出的名字之一，原样使用，不要加编号、括号、角色说明\n"
            "3. 不要输出任何 markdown 标题、列表、括号里的舞台提示、旁白说明或空行\n"
            "4. 不要重复人名，不要写“主持人：”这种通用称谓\n"
        )
    return (
        "Strict output format:\n"
        "1. Every line is one spoken turn, formatted exactly as `SPEAKER_NAME: dialogue`\n"
        "2. SPEAKER_NAME must be one of the given names, used verbatim — no numbering, "
        "no role labels, no parentheses\n"
        "3. No markdown headings, no lists, no stage directions in brackets, no blank "
        "lines, no labels like 'Host:'\n"
        "4. Never repeat the name inside the dialogue text\n"
        "5. No word counts, counting, planning notes, or meta-commentary in dialogue "
        "(e.g. no '= 30 words', 'need more', 'Total 124', numbered-word drafts)\n"
    )


def _host_block(hosts: list[Host], language: str) -> str:
    lines = []
    for i, h in enumerate(hosts, 1):
        persona = h.persona or ("host" if language == "en" else "主持人")
        lines.append(f"- {h.name} ({h.gender}, {persona})")
    joined = "\n".join(lines)
    if language == "zh":
        return f"说话人：\n{joined}\n\n"
    return f"CAST:\n{joined}\n\n"


def _role_hint(style: str, hosts: list[Host], language: str) -> str:
    roles = ROLE_BY_STYLE.get(style)
    if not roles or len(hosts) < 2:
        return ""
    if language == "zh":
        return f"角色分工：{hosts[0].name} 是{roles[0]}，{hosts[1].name} 是{roles[1]}。\n"
    return f"ROLE ASSIGNMENT: {hosts[0].name} is the {roles[0]}; {hosts[1].name} is the {roles[1]}.\n"


def _units_word(language: str) -> str:
    return "字" if language == "zh" else "words"


def _units_target(total_units: int, language: str) -> str:
    return f"{total_units} {_units_word(language)}"


# --------------------------------------------------------------------------- #
# Planning / word-count artifact cleaning
# --------------------------------------------------------------------------- #

# Pattern: word count annotations at end of line, e.g. "= 30", "= 22ish.", "=20"
_WORD_COUNT_SUFFIX = re.compile(r"\s*=\s*\d+(?:ish)?\.?\s*$", re.I)

# Pattern: arithmetic / counting expressions, e.g. "Total 6+18+10+55+25=114. About target"
_COUNTING_EXPR = re.compile(
    r"(?:^|\s)(?:Total\s+)?\d+(?:\s*[+\-*/]\s*\d+)+\s*=\s*\d+|"
    r"(?:Need\s+)?(?:total|target)\s+(?:about\s+)?\d+|"
    r"About\s+target|"
    r"Let'?s\s+count\s+carefully|"
    r"That'?s\s+\d+\??",
    re.I
)

# Pattern: numbered-word drafts, e.g. "What's(1) the2 practical3" or "word1 word2 word3"
_NUMBERED_WORDS = re.compile(
    r"[\w']+\(\d+\)\s+\w+\d+|"  # word(1) word2, allowing apostrophe
    r"(?:\b[A-Za-z][A-Za-z\-_]*\d+\b\s*){3,}",  # word1 word2 word3 (at least 3, with hyphens)
    re.I
)

# Pattern: line number prefix from source dumps, e.g. "    10|Sofia:" or "20|Marcus:"
_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+\s*\|\s*")

# Pattern: meta planning instructions that should never appear in dialogue
_META_PLANNING = re.compile(
    r"(?:^|\s)(?:"
    r"Need\s+(?:maybe\s+)?(?:add\s+more|target|total)|"
    r"Then\s+\w+\s+answer\s+with|"
    r"Should\s+be\s+okay|"
    r"Could\s+(?:add|include|be)|"
    r"(?:content\s+)?pillars?,?\s+(?:unique\s+)?perspective|"
    r"mission\s+and\s+beliefs|"
    r"what\s+you'?re\s+against|"
    r"walks?\s+through|"
    r"before\s+\w+\s+closes|"
    r"Marcus\s+closes?\s+promo|"
    r"good\.?\s*Word\s+count|"
    r"(?:Brand|content)\s+(?:and\s+content\s+)?docs?\b"
    r")",
    re.I
)


def _is_planning_line(text: str) -> bool:
    """Return True if the line is primarily planning/meta text, not dialogue."""
    if _COUNTING_EXPR.search(text):
        return True
    if _META_PLANNING.search(text):
        return True
    numbered_match = _NUMBERED_WORDS.search(text)
    if numbered_match:
        ratio = len(numbered_match.group(0)) / len(text) if text else 0
        if ratio > 0.2 or len(numbered_match.group(0)) > 20:
            return True
    return False


_NUMBERED_SUFFIX = re.compile(
    r"[\w']+\(\d+\)(?:\s+[\w\-]+\d+)*\s*$|"  # word(1) word2 word3... at end
    r"(?:\s+[\w\-]+\d+){2,}\s*$",  # word1 word2 word3... at end (2+ consecutive)
    re.I
)


def _clean_dialogue_text(text: str) -> str:
    """Remove word-count annotations and planning artifacts from dialogue text.
    
    Keeps the core dialogue intact while stripping:
    - Word count suffixes: "= 30ish.", "= 22", "=20"
    - Line number prefixes from dumps: "10|"
    - Numbered-word suffixes: "What's(1) the2 practical3..." at end of line
    """
    if not text:
        return ""
    text = _LINE_NUMBER_PREFIX.sub("", text)
    text = _WORD_COUNT_SUFFIX.sub("", text)
    text = _NUMBERED_SUFFIX.sub("", text)
    text = text.strip()
    return text


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

_SPEAKER_LINE = re.compile(r"^\s*(?:\*\*)?([A-Za-z一-鿿][A-Za-z0-9一-鿿 ._'\-]{0,28}?)(?:\*\*)?\s*[:：]\s*(.+)$")


def parse_turns(raw: str, hosts: list[Host], segment: int = 0) -> list[Turn]:
    """Parse `NAME: text` lines into turns, tolerating minor format drift.
    
    Lines containing planning artifacts, word-count annotations, or meta
    instructions are cleaned or rejected to prevent them from reaching TTS.
    """
    exact = {h.name: h.name for h in hosts}
    lowered = {h.name.lower(): h.name for h in hosts}
    turns: list[Turn] = []

    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith(("- ", "* ", "• ")):
            line = line[2:].strip()
        line = _LINE_NUMBER_PREFIX.sub("", line)
        match = _SPEAKER_LINE.match(line)
        if not match:
            continue
        name, text = match.group(1).strip(), match.group(2).strip()
        text = text.strip().strip('"').strip("“”").strip()
        if not text:
            continue

        if _is_planning_line(text):
            continue

        text = _clean_dialogue_text(text)
        if not text:
            continue

        # Resolve to the canonical host *name* (a string), never the Host object.
        resolved = (exact.get(name)
                    or lowered.get(name.lower())
                    or lowered.get(name.title().lower()))
        if resolved is None:
            # Accept "Alice Smith" matching host "alice smith" or vice versa.
            for host in hosts:
                if name.lower() in host.name.lower() or host.name.lower() in name.lower():
                    resolved = host.name
                    break
        if resolved is None:
            # Unknown speaker — drop the line rather than speaking it blind.
            continue
        turns.append(Turn(speaker=resolved, text=text, segment=segment))

    # Merge consecutive turns from the same speaker so TTS gets natural phrases.
    merged: list[Turn] = []
    for turn in turns:
        if merged and merged[-1].speaker == turn.speaker:
            merged[-1].text = f"{merged[-1].text} {turn.text}".strip()
        else:
            merged.append(turn)
    for i, turn in enumerate(merged):
        turn.index = i
    return merged


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def _condensed_source(text: str, chunks: list, limit: int = 11000) -> str:
    """A compressed view of the source for the planning call."""
    parts = [text[:2500]]
    for chunk in chunks[:40]:
        head = chunk.text[:700].replace("\n\n", " ")
        parts.append(f"[part {chunk.index + 1}] {head}")
    combined = "\n\n".join(parts)
    return combined[:limit]


def generate_script(
    client: LLMClient,
    source_text: str,
    chunks: list,
    hosts: list[Host],
    *,
    language: str = "en",
    style: str = "narrative",
    target_minutes: float = 5.0,
    title_hint: str = "",
    extra_instructions: str = "",
    progress: ProgressFn | None = None,
) -> Script:
    def report(stage: str, pct: float) -> None:
        if progress:
            progress(stage, max(0.0, min(1.0, pct)))

    style_meta = STYLES.get(style, STYLES["narrative"])
    style_desc = style_meta["en"] if language == "en" else style_meta["label"]
    units_per_minute = SPEECH_UNITS_PER_MINUTE.get(language, SPEECH_UNITS_PER_MINUTE["en"])
    total_units = max(60, int(target_minutes * units_per_minute))

    script = Script(title=title_hint or "Podcast", hosts=hosts, style=style,
                    language=language, target_minutes=target_minutes)

    # ---- 1. Plan ---------------------------------------------------------- #
    report("Planning the episode", 0.05)
    condensed = _condensed_source(source_text, chunks)
    n_segments = max(1, min(len(chunks) or 1, max(2, int(round(target_minutes / 1.5)) + 1)))

    plan_prompt = _plan_prompt(language, style_desc, hosts, condensed,
                                n_segments, total_units, title_hint)
    try:
        outline_raw = client.chat(
            [{"role": "system", "content": "You are a senior podcast producer and show runner."},
             {"role": "user", "content": plan_prompt}],
            temperature=0.6, max_tokens=1500)
    except Exception:
        outline_raw = ""
    outline = _parse_outline(outline_raw, language)
    script.outline = outline

    if not outline:
        outline = [f"Part {i + 1}" for i in range(n_segments)]

    # ---- 2. Write each segment ------------------------------------------- #
    segments = _pair_chunks_with_outline(chunks, outline)
    # Reserve a slice for the hook and sign-off, and ask for a little under
    # budget per segment because models tend to overshoot.
    intro_budget = max(24, int(total_units * 0.10))
    segment_budget = max(60, total_units - intro_budget)
    per_segment_units = max(40, int(segment_budget * 0.88 / len(segments)))
    turns: list[Turn] = []

    for idx, (chunk, outline_line) in enumerate(segments):
        report(f"Writing segment {idx + 1}/{len(segments)}", 0.1 + 0.65 * (idx / max(1, len(segments))))
        prompt = _segment_prompt(
            language=language, style_desc=style_desc, hosts=hosts,
            outline_line=outline_line, chunk_text=chunk.text,
            unit_target=per_segment_units, segment_index=idx,
            total_segments=len(segments), title=title_hint or script.title,
            extra=extra_instructions, role_hint=_role_hint(style, hosts, language))
        raw = client.chat(
            [{"role": "system", "content": "You write broadcast-ready podcast dialogue."},
             {"role": "user", "content": prompt}],
            temperature=0.85, max_tokens=min(6000, int(per_segment_units * 3) + 600))
        parsed = parse_turns(raw, hosts, segment=idx)
        turns.extend(parsed)

    # ---- 3. Intro / outro ------------------------------------------------- #
    report("Adding intro and outro", 0.78)
    try:
        intro_outro = client.chat(
            [{"role": "system", "content": "You write broadcast-ready podcast dialogue."},
             {"role": "user", "content": _intro_outro_prompt(
                 language, style_desc, hosts, title_hint or script.title, outline,
                 intro_budget)}],
            temperature=0.8, max_tokens=max(400, int(intro_budget * 2.5)))
        lines = intro_outro.splitlines()
        head, tail = [], []
        mode = "head"
        for line in lines:
            stripped = line.strip()
            if re.match(r"^\s*#?\s*(OUTRO|END|结尾|片尾|尾声)\s*[:：]?\s*$", stripped, re.I):
                mode = "tail"
                continue
            if re.match(r"^\s*#\s*(OUTRO|END|结尾|片尾|尾声)", stripped, re.I):
                mode = "tail"
                stripped = re.sub(r"^\s*#\s*(OUTRO|END|结尾|片尾|尾声)\s*[:：]?\s*", "", stripped,
                                  flags=re.I)
            if not stripped:
                continue
            (head if mode == "head" else tail).append(stripped)
        intro_turns = parse_turns("\n".join(head), hosts, segment=-1)
        outro_turns = parse_turns("\n".join(tail), hosts, segment=len(segments))
        script.turns = intro_turns + turns + outro_turns
    except Exception:
        script.turns = turns

    script.turns = _trim_to_target(script.turns, total_units, language)
    for i, turn in enumerate(script.turns):
        turn.index = i

    report("Script complete", 0.85)
    return script


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #

def _parse_outline(raw: str, language: str) -> list[str]:
    """Turn the planner's reply into a clean list of segment summaries.

    The model often answers with one long paragraph containing "Segment 2:"
    style markers, so the text is split on those markers as well as on
    newlines.
    """
    if not raw:
        return []
    text = raw.strip()

    if language == "zh":
        marker = re.compile(r"(?:^|[\s，,。；;])(?:第\s*[0-9一二三四五六七八九十]+\s*段\s*[:：]|-)")
    else:
        marker = re.compile(r"(?:^|[\s])(?:Segment\s*\d+\s*[:.\-]|-)", re.I)

    pieces: list[str] = []
    for block in re.split(r"\n+", text):
        block = block.strip()
        if not block:
            continue
        parts = marker.split(block)
        if len(parts) > 1:
            for part in parts:
                part = part.strip(" \t-*•#0123456789.:" ).strip()
                if part:
                    pieces.append(part)
        else:
            pieces.append(block)

    cleaned: list[str] = []
    for piece in pieces:
        piece = piece.strip().strip("*#-• ").strip()
        if len(piece) < 3:
            continue
        if len(piece) > 320:
            piece = piece[:317].rstrip() + "…"
        cleaned.append(piece)
    return cleaned[:12]


def _trim_to_target(turns: list[Turn], total_units: int, language: str,
                    tolerance: float = 1.12) -> list[Turn]:
    """Drop turns when the script runs long.

    The hook and the sign-off are always kept. Removal starts at the end and
    takes every other turn first, so the exchange stays balanced rather than
    ending mid-conversation; a second, more aggressive pass runs only if that
    was not enough.
    """
    def units_of(items: list[Turn]) -> int:
        return sum(estimate_speech_units(t.text, language) for t in items)

    if not turns:
        return turns
    target = int(total_units * tolerance)
    if units_of(turns) <= target:
        return turns

    keep = list(turns)
    protected = {0, 1, len(keep) - 1, len(keep) - 2}

    for stride in (2, 1):
        index = len(keep) - 3
        while units_of(keep) > target and index > 2:
            if index not in protected:
                del keep[index]
            index -= stride
        if units_of(keep) <= target:
            break
    return keep


def _pair_chunks_with_outline(chunks: list, outline: list[str]) -> list[tuple[Any, str]]:
    """Pair each source chunk with the outline line it should cover.

    When there are more outline entries than chunks, chunks are reused so the
    outline drives the episode length; when there are fewer, chunks share a
    single outline line.
    """
    if not chunks:
        return [(None, line) for line in outline[:1]]
    if not outline:
        outline = [f"Part {i + 1}" for i in range(len(chunks))]

    pairs: list[tuple[Any, str]] = []
    if len(outline) >= len(chunks):
        step = len(outline) / len(chunks)
        for i, chunk in enumerate(chunks):
            index = min(len(outline) - 1, int(i * step))
            pairs.append((chunk, outline[index]))
    else:
        per_chunk = len(chunks) / len(outline)
        for i, chunk in enumerate(chunks):
            index = min(len(outline) - 1, int(i / per_chunk))
            pairs.append((chunk, outline[index]))
    return pairs


def _plan_prompt(language, style_desc, hosts, condensed, n_segments, total_units,
                 title_hint) -> str:
    if language == "zh":
        return (
            "你是一档播客节目的总策划。请通读下面的素材，为一期播客设计分段大纲。\n\n"
            f"节目风格：{style_desc}\n"
            f"说话人：{', '.join(h.name for h in hosts)}\n"
            f"目标时长：约 {_units_target(total_units, language)}\n"
            f"分段数量：{n_segments}\n\n"
            "【素材】\n" + condensed + "\n\n"
            "请只输出大纲，每行一段，格式为 `- 第N段：这一段讲什么`。不要写任何其他内容。"
        )
    return (
        "You are the show runner of a podcast. Read the material below and design a "
        "segment-by-segment outline for one episode.\n\n"
        f"SHOW STYLE: {style_desc}\n"
        f"CAST: {', '.join(h.name for h in hosts)}\n"
        f"TARGET LENGTH: about {_units_target(total_units, language)}\n"
        f"NUMBER OF SEGMENTS: {n_segments}\n\n"
        "MATERIAL:\n" + condensed + "\n\n"
        "Output ONLY the outline, one segment per line, in the form "
        "`- Segment N: what happens here`. Nothing else."
    )


def _segment_prompt(language, style_desc, hosts, outline_line, chunk_text, unit_target,
                    segment_index, total_segments, title, extra, role_hint) -> str:
    if language == "zh":
        return (
            f"你在写播客第 {segment_index + 1}/{total_segments} 段的台词。\n\n"
            f"节目风格：{style_desc}\n"
            f"本段大纲：{outline_line}\n"
            f"节目名：{title}\n"
            f"本段目标：约 {unit_target} 字\n"
            + (role_hint + "\n" if role_hint else "")
            + _host_block(hosts, language)
            + _format_rule(language)
            + ("额外要求：" + extra + "\n" if extra else "")
            + "\n【本段素材】\n" + chunk_text + "\n\n"
            "现在只输出台词本身，每行一句，不要任何解释、字数统计或规划备注。"
        )
    return (
        f"You are writing segment {segment_index + 1} of {total_segments} for a podcast.\n\n"
        f"SHOW STYLE: {style_desc}\n"
        f"THIS SEGMENT: {outline_line}\n"
        f"EPISODE TITLE: {title}\n"
        f"TARGET FOR THIS SEGMENT: about {unit_target} words\n"
        + (role_hint + "\n" if role_hint else "")
        + _host_block(hosts, language)
        + _format_rule(language)
        + (f"EXTRA REQUIREMENTS: {extra}\n" if extra else "")
        + "\nSOURCE MATERIAL FOR THIS SEGMENT:\n" + chunk_text + "\n\n"
        "Now output only the dialogue lines. No commentary, no headings, no notes, "
        "no word counts, no planning or meta-text."
    )


def _intro_outro_prompt(language, style_desc, hosts, title, outline, budget) -> str:
    outline_txt = "\n".join(f"- {o}" for o in outline[:8])
    if language == "zh":
        return (
            "你正在为一档播客写片头和片尾。\n\n"
            f"节目名：{title}\n风格：{style_desc}\n"
            f"片头加片尾总长度：约 {budget} 字\n"
            + _host_block(hosts, language)
            + "【本期大纲】\n" + outline_txt + "\n\n"
            "请输出：\n"
            "# INTRO\n"
            "（最多2句让人想听下去的片头，不要用“欢迎收听”这种套话）\n"
            "# OUTRO\n"
            "（最多1句简短的结尾和告别）\n\n"
            + _format_rule(language)
        )
    return (
        "You are writing the cold open and the sign-off for a podcast episode.\n\n"
        f"EPISODE TITLE: {title}\nSTYLE: {style_desc}\n"
        f"TOTAL LENGTH FOR INTRO AND OUTRO TOGETHER: about {budget} words\n"
        + _host_block(hosts, language)
        + "EPISODE OUTLINE:\n" + outline_txt + "\n\n"
        "Output:\n"
        "# INTRO\n"
        "(at most 2 hook lines that make people want to keep listening — no 'welcome to the show' boilerplate)\n"
        "# OUTRO\n"
        "(at most 1 short sign-off line)\n\n"
        + _format_rule(language)
    )


def estimate_minutes(script: Script) -> float:
    return units_to_minutes(script.units(script.language), script.language)
