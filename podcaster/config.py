"""Persistent settings for Podcaster.

Everything lives in a single JSON file next to the app so the tool stays a
"pure tool" with no database. API keys are stored locally in clear text; that
is an accepted trade-off for a single-user local utility.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
CONFIG_PATH = DATA_DIR / "config.json"

MINIMAX_CN = "https://api.minimaxi.com/v1"
MINIMAX_INTL = "https://api.minimax.io/v1"

DEFAULTS: dict[str, Any] = {
    "llm_base_url": MINIMAX_CN,
    "llm_model": "MiniMax-M2.5",
    "llm_api_key": "",
    "llm_temperature": 0.7,
    "llm_max_tokens": 4096,
    "tts_provider": "minimax",
    "tts_base_url": MINIMAX_CN,
    "tts_model": "speech-2.5-hd-preview",
    "tts_api_key": "",
    "tts_speed": 1.0,
    "image_base_url": "",
    "image_api_key": "",
    "image_model": "image-01",
    "output_dir": str(APP_DIR / "output"),
    "source_chunk_chars": 6000,
    "keep_workdir": True,
}

_lock = threading.Lock()


def _read_env_key() -> str:
    """Pick up the MiniMax key from the environment or from env.ps1."""
    for var in ("MINIMAX_APIKEY", "MINIMAX_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    # Fall back to env.ps1 sitting next to the app.
    env_file = APP_DIR / "env.ps1"
    if env_file.exists():
        try:
            text = env_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        m = re.search(r"MINIMAX_API_?KEY\s*=\s*['\"]([^'\"]+)['\"]", text)
        if m:
            return m.group(1).strip()
    return ""


def load() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                cfg.update({k: v for k, v in stored.items() if k in DEFAULTS})
        except (OSError, json.JSONDecodeError):
            pass
    env_key = _read_env_key()
    if not cfg.get("llm_api_key"):
        cfg["llm_api_key"] = env_key
    if not cfg.get("tts_api_key"):
        cfg["tts_api_key"] = env_key
    if not cfg.get("image_api_key"):
        cfg["image_api_key"] = env_key
    return cfg


def save(cfg: dict[str, Any]) -> dict[str, Any]:
    clean = {k: cfg.get(k, DEFAULTS.get(k)) for k in DEFAULTS}
    clean["llm_api_key"] = str(clean.get("llm_api_key") or "").strip()
    clean["tts_api_key"] = str(clean.get("tts_api_key") or "").strip()
    clean["image_api_key"] = str(clean.get("image_api_key") or "").strip()
    clean["llm_base_url"] = str(clean.get("llm_base_url") or DEFAULTS["llm_base_url"]).rstrip("/")
    clean["tts_base_url"] = str(clean.get("tts_base_url") or DEFAULTS["tts_base_url"]).rstrip("/")
    clean["image_base_url"] = str(clean.get("image_base_url") or "").rstrip("/")
    clean["output_dir"] = str(clean.get("output_dir") or DEFAULTS["output_dir"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        CONFIG_PATH.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    return clean


def public_view(cfg: dict[str, Any]) -> dict[str, Any]:
    """Mask the secrets so the UI can show whether a key is configured."""
    view = dict(cfg)
    for field in ("llm_api_key", "tts_api_key", "image_api_key"):
        key = view.get(field) or ""
        view[field] = ""
        view[f"{field}_set"] = bool(key)
        view[f"{field}_hint"] = (key[:6] + "…" + key[-4:]) if len(key) > 12 else ("••••" if key else "")
    return view


def merge_patch(patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial update without wiping a key the UI sent back masked."""
    cfg = load()
    for key, value in (patch or {}).items():
        if key not in DEFAULTS:
            continue
        if key in ("llm_api_key", "tts_api_key", "image_api_key") and not value:
            continue  # masked / blank means "keep existing"
        cfg[key] = value
    return save(cfg)
