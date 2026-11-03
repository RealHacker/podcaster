"""Voice catalogue and host→voice assignment.

The catalogue is a curated, API-verified list. It can be overridden by
``podcaster/voices_catalog.json`` (same schema) so the set can be refreshed
without touching code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).resolve().parent / "voices_catalog.json"

# Verified against the live MiniMax TTS endpoint. `tags` are free-form hints
# used to pick a good fit for the requested tone.
BUILTIN: list[dict[str, Any]] = [
    # ---- English-leaning voices ----
    {"id": "English_Trustworthy_Man", "label": "Trustworthy Man",
     "gender": "male", "tone": "trustworthy, warm, narrator"},
    {"id": "English_Upbeat_Woman", "label": "Upbeat Woman",
     "gender": "female", "tone": "bright, energetic, host"},
    {"id": "audiobook_male_1", "label": "Audiobook Man",
     "gender": "male", "tone": "measured, documentary"},
    {"id": "audiobook_female_1", "label": "Audiobook Woman",
     "gender": "female", "tone": "measured, documentary"},
    {"id": "presenter_male", "label": "Presenter Man",
     "gender": "male", "tone": "professional, anchor"},
    {"id": "presenter_female", "label": "Presenter Woman",
     "gender": "female", "tone": "professional, anchor"},
    # ---- Chinese-leaning voices ----
    {"id": "male-qn-qingse", "label": "青涩青年 / Youthful Man",
     "gender": "male", "tone": "youthful, sincere"},
    {"id": "male-qn-jingying", "label": "精英青年 / Elite Young Man",
     "gender": "male", "tone": "confident, sharp"},
    {"id": "male-qn-badao", "label": "霸道青年 / Commanding Man",
     "gender": "male", "tone": "decisive, magnetic"},
    {"id": "male-qn-daxuesheng", "label": "青年大学生 / Student Man",
     "gender": "male", "tone": "casual, friendly"},
    {"id": "female-shaonv", "label": "少女 / Young Woman",
     "gender": "female", "tone": "light, friendly"},
    {"id": "female-yujie", "label": "御姐 / Confident Woman",
     "gender": "female", "tone": "mature, assured"},
    {"id": "female-tianmei", "label": "甜美 / Sweet Woman",
     "gender": "female", "tone": "warm, sweet"},
    {"id": "female-chengshu", "label": "成熟女性 / Mature Woman",
     "gender": "female", "tone": "mature, composed"},
    {"id": "Chinese (Mandarin)_News_Anchor", "label": "新闻主播 / News Anchor",
     "gender": "male", "tone": "news, formal"},
    {"id": "Chinese (Mandarin)_Warm_Bestie", "label": "暖心好友 / Warm Bestie",
     "gender": "female", "tone": "warm, friendly"},
    {"id": "Chinese (Mandarin)_Wise_Women", "label": "知性女声 / Wise Woman",
     "gender": "female", "tone": "thoughtful, wise"},
    {"id": "Chinese (Mandarin)_Lyrical_Voice", "label": "抒情女声 / Lyrical Voice",
     "gender": "female", "tone": "poetic, gentle"},
]

# ModelVerse Qwen TTS voices (qwen3-tts-flash model).
# These support both Chinese and English with language_type metadata.
QWEN_VOICES: list[dict[str, Any]] = [
    {"id": "Cherry", "label": "Cherry (Qwen)",
     "gender": "female", "tone": "warm, expressive", "provider": "qwen"},
    {"id": "Ethan", "label": "Ethan (Qwen)",
     "gender": "male", "tone": "confident, clear", "provider": "qwen"},
    {"id": "Serena", "label": "Serena (Qwen)",
     "gender": "female", "tone": "calm, professional", "provider": "qwen"},
    {"id": "Chelsie", "label": "Chelsie (Qwen)",
     "gender": "female", "tone": "friendly, upbeat", "provider": "qwen"},
]


def get_qwen_voices() -> list[dict[str, Any]]:
    """Return voices available for ModelVerse Qwen TTS."""
    return [dict(v) for v in QWEN_VOICES]


def get_default_qwen_voice() -> str:
    """Return the default voice id for Qwen TTS connection testing."""
    return "Cherry"

_LANG_HINTS = {
    "zh": ("male-qn", "female-", "chinese (mandarin)", "presenter", "audiobook",
           "english_upbeat", "english_trustworthy"),
    "en": ("english", "presenter", "audiobook", "male-qn", "female-",
           "chinese (mandarin)"),
}


def load_catalog() -> list[dict[str, Any]]:
    if CATALOG_PATH.exists():
        try:
            data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return [dict(v) for v in BUILTIN]


def save_catalog(voices: list[dict[str, Any]]) -> None:
    CATALOG_PATH.write_text(json.dumps(voices, indent=2, ensure_ascii=False), encoding="utf-8")


def list_voices(language: str | None = None, gender: str | None = None) -> list[dict[str, Any]]:
    voices = load_catalog()
    out = []
    for voice in voices:
        if gender and voice.get("gender") != gender:
            continue
        if language:
            hints = _LANG_HINTS.get(language, ())
            blob = f"{voice.get('id', '')} {voice.get('label', '')}".lower()
            if not any(h in blob for h in hints):
                continue
        out.append(voice)
    return out


def assign_voices(hosts: list, language: str = "en",
                  overrides: dict[str, str] | None = None) -> list[str]:
    """Pick a distinct voice per host, respecting the requested gender.

    ``overrides`` maps a host name to an explicit voice id.
    """
    overrides = overrides or {}
    catalog = load_catalog()
    used: set[str] = set()
    assigned: list[str] = []

    def candidates(gender: str) -> list[dict[str, Any]]:
        preferred = [v for v in catalog if v.get("gender") == gender]
        if language == "zh":
            hints = ("male-qn", "female-", "chinese (mandarin)")
        else:
            hints = ("english", "presenter", "audiobook")
        preferred.sort(key=lambda v: 0 if any(h in str(v.get("id", "")).lower() for h in hints) else 1)
        return preferred

    for host in hosts:
        override = overrides.get(host.name)
        if override:
            host.voice_id = override
            assigned.append(override)
            used.add(override)
            continue
        gender = (host.gender or "female").lower()
        pool = candidates(gender) or catalog
        picked = None
        for voice in pool:
            if voice.get("id") not in used:
                picked = voice["id"]
                break
        if picked is None:
            picked = pool[0]["id"] if pool else catalog[0]["id"]
        host.voice_id = picked
        assigned.append(picked)
        used.add(picked)
    return assigned


def color_for(gender: str) -> str:
    from .media import PALETTE

    key = (gender or "").lower()
    if key in PALETTE:
        return PALETTE[key]
    return PALETTE["default"]
