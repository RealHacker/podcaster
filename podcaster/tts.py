"""Text-to-speech clients.

Two providers are supported:

* ``minimax`` — the native ``/v1/t2a_v2`` endpoint (default).
* ``openai``  — any OpenAI-compatible ``/audio/speech`` endpoint.
"""
from __future__ import annotations

import base64
import binascii
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .subtitles import SubtitleCue, parse_subtitle_payload


class TTSError(RuntimeError):
    pass


@dataclass
class ClipResult:
    path: Path
    duration_ms: int = 0
    chars: int = 0
    format: str = "mp3"
    subtitles: list[SubtitleCue] = field(default_factory=list)


QWEN_MAX_CHARS = 600


def split_for_tts(text: str, max_chars: int = 4000) -> list[str]:
    """Split a long turn into API-sized pieces on sentence boundaries."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: list[str] = []
    buf = ""
    for sentence in re.split(r"(?<=[.!?。！？；;])\s*", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if buf:
                pieces.append(buf)
                buf = ""
            for i in range(0, len(sentence), max_chars):
                pieces.append(sentence[i:i + max_chars])
            continue
        if len(buf) + len(sentence) + 1 > max_chars and buf:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def is_qwen_dialect(model: str, base_url: str) -> bool:
    """Detect if using ModelVerse Qwen TTS which has a different API schema."""
    model_lower = (model or "").lower()
    url_lower = (base_url or "").lower()
    return "qwen" in model_lower or "modelverse" in url_lower


def _convert_wav_to_mp3(wav_bytes: bytes, out_path: Path) -> Path:
    """Convert WAV audio bytes to MP3 and write to out_path using ffmpeg.
    
    This ensures callers receive the file at the exact path they requested,
    rather than a different extension based on what the API returned.
    """
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        out_path.write_bytes(wav_bytes)
        return out_path

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_wav = Path(tmp.name)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg, "-y", "-i", str(tmp_wav),
            "-c:a", "libmp3lame", "-b:a", "192k",
            "-ar", "44100", "-ac", "1",
            str(out_path)
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0:
            out_path.write_bytes(wav_bytes)
            return out_path
        return out_path
    finally:
        tmp_wav.unlink(missing_ok=True)


class _RateLimiter:
    """Keeps us under the provider's requests-per-minute ceiling."""

    def __init__(self, min_interval: float = 1.1):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


class MiniMaxTTS:
    provider = "minimax"

    def __init__(self, base_url: str, api_key: str, model: str = "speech-2.5-hd-preview",
                 speed: float = 1.0, timeout: float = 300.0, min_interval: float = 1.2):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = model
        self.speed = speed
        self.timeout = timeout
        self.limiter = _RateLimiter(min_interval)

    def _endpoint(self) -> str:
        base = self.base_url
        if not base:
            raise TTSError("TTS base URL is not configured")
        if base.endswith("/t2a_v2"):
            return base
        return base + "/t2a_v2"

    def _fetch_subtitles(self, url: str, source_text: str = "") -> list[SubtitleCue]:
        """Download the subtitle file the API returned and parse it."""
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                response = client.get(url)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            # Subtitles are a nice-to-have; a failed download must not fail
            # the whole episode.
            return []
        return parse_subtitle_payload(payload, source_text)

    def synthesize(self, text: str, voice_id: str, out_path: Path,
                   language: str = "en", speed: float | None = None,
                   subtitle_enable: bool = False,
                   subtitle_type: str = "word") -> ClipResult:
        if not self.api_key:
            raise TTSError("TTS API key is not configured")
        if not text.strip():
            raise TTSError("Cannot synthesize empty text")

        payload: dict[str, Any] = {
            "model": self.model,
            "text": text,
            "stream": False,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed if speed is not None else self.speed,
                "vol": 1.0,
                "pitch": 0,
            },
            "language_boost": "Chinese" if language == "zh" else "English",
            "output_format": "hex",
        }
        if subtitle_enable:
            payload["subtitle_enable"] = True
            payload["subtitle_type"] = subtitle_type
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error: Exception | None = None
        for attempt in range(5):
            self.limiter.wait()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self._endpoint(), headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_error = TTSError(f"TTS network error: {exc}")
                time.sleep(min(2 ** attempt, 20))
                continue

            if response.status_code >= 400:
                detail = response.text[:600]
                if response.status_code in (429, 500, 502, 503, 504) or "rate limit" in detail.lower():
                    last_error = TTSError(f"TTS HTTP {response.status_code}: {detail}")
                    time.sleep(min(3 * (attempt + 1), 30))
                    continue
                raise TTSError(f"TTS HTTP {response.status_code}: {detail}")

            try:
                data = response.json()
            except Exception as exc:
                raise TTSError(f"TTS returned invalid JSON: {exc}") from exc

            base_resp = data.get("base_resp") or {}
            if base_resp.get("status_code", 0) != 0:
                message = base_resp.get("status_msg", "unknown error")
                if "rate limit" in str(message).lower():
                    last_error = TTSError(f"TTS rate limited: {message}")
                    time.sleep(min(5 * (attempt + 1), 45))
                    continue
                raise TTSError(f"TTS error: {message}")

            audio_hex = (data.get("data") or {}).get("audio") or ""
            if not audio_hex:
                raise TTSError("TTS returned no audio data")

            try:
                audio_bytes = bytes.fromhex(audio_hex)
            except (ValueError, binascii.Error):
                try:
                    audio_bytes = base64.b64decode(audio_hex)
                except Exception as exc:
                    raise TTSError(f"Could not decode TTS audio payload: {exc}") from exc

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(audio_bytes)
            info = data.get("extra_info") or {}

            cues: list[SubtitleCue] = []
            if subtitle_enable:
                subtitle_url = (data.get("data") or {}).get("subtitle_file") or ""
                if subtitle_url:
                    cues = self._fetch_subtitles(subtitle_url, text)

            return ClipResult(path=out_path, duration_ms=int(info.get("audio_length") or 0),
                              chars=int(info.get("usage_characters") or len(text)),
                              format=str(info.get("audio_format") or "mp3"),
                              subtitles=cues)

        raise TTSError(f"TTS request failed after retries: {last_error}")


class OpenAICompatTTS:
    provider = "openai"

    def __init__(self, base_url: str, api_key: str, model: str = "tts-1",
                 speed: float = 1.0, timeout: float = 300.0, min_interval: float = 0.4,
                 voice_id: str = "alloy"):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = model
        self.speed = speed
        self.timeout = timeout
        self.voice_id = voice_id
        self.limiter = _RateLimiter(min_interval)
        self._is_qwen = is_qwen_dialect(model, base_url)

    def _endpoint(self) -> str:
        base = self.base_url
        if not base:
            raise TTSError("TTS base URL is not configured")
        if base.endswith("/audio/speech"):
            return base
        return base + "/audio/speech"

    def _build_payload(self, text: str, voice_id: str, language: str,
                       speed: float | None) -> dict[str, Any]:
        """Build request payload based on dialect (Qwen vs standard OpenAI)."""
        if self._is_qwen:
            payload: dict[str, Any] = {
                "model": self.model,
                "input": text,
                "voice": voice_id or self.voice_id,
            }
            lang_map = {"en": "English", "zh": "Chinese"}
            lang_type = lang_map.get(language, "Auto")
            payload["metadata"] = {"language_type": lang_type}
            return payload
        return {
            "model": self.model,
            "voice": voice_id or self.voice_id,
            "input": text,
            "speed": speed if speed is not None else self.speed,
            "response_format": "mp3",
        }

    def _handle_response(self, response: httpx.Response, out_path: Path,
                         source_text: str = "") -> ClipResult:
        """Handle response: binary audio (standard OpenAI) or JSON+URL (Qwen).
        
        For Qwen dialect, the API returns JSON with an audio URL (often .wav).
        If the caller requests .mp3 but the API returns .wav, we convert using
        ffmpeg to ensure the file is written at the exact path the caller expects.
        """
        content_type = response.headers.get("content-type", "").lower()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        text_chars = len(source_text) if source_text else 0

        if self._is_qwen or "application/json" in content_type:
            try:
                data = response.json()
            except Exception as exc:
                raise TTSError(f"Expected JSON response but got invalid data: {exc}") from exc

            audio_url = (data.get("output") or {}).get("audio", {}).get("url")
            if not audio_url:
                raise TTSError(f"Qwen TTS response missing output.audio.url: {data}")

            try:
                with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                    audio_resp = client.get(audio_url)
                audio_resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise TTSError(f"Failed to download Qwen audio from URL: {exc}") from exc

            audio_bytes = audio_resp.content
            audio_type = audio_resp.headers.get("content-type", "").lower()
            is_wav = "wav" in audio_type
            wants_mp3 = out_path.suffix.lower() == ".mp3"

            if is_wav and wants_mp3:
                result = _convert_wav_to_mp3(audio_bytes, out_path)
                return ClipResult(path=result, chars=text_chars, format="mp3")

            out_path.write_bytes(audio_bytes)
            fmt = "wav" if is_wav else "mp3"
            return ClipResult(path=out_path, chars=text_chars, format=fmt)

        out_path.write_bytes(response.content)
        return ClipResult(path=out_path, chars=text_chars, format="mp3")

    def max_chars(self) -> int:
        """Return the maximum characters per request for this dialect."""
        return QWEN_MAX_CHARS if self._is_qwen else 4000

    def synthesize(self, text: str, voice_id: str, out_path: Path,
                   language: str = "en", speed: float | None = None,
                   subtitle_enable: bool = False,
                   subtitle_type: str = "word") -> ClipResult:
        if not self.api_key:
            raise TTSError("TTS API key is not configured")
        if not text.strip():
            raise TTSError("Cannot synthesize empty text")

        payload = self._build_payload(text, voice_id, language, speed)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None

        for attempt in range(4):
            self.limiter.wait()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self._endpoint(), headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_error = TTSError(f"TTS network error: {exc}")
                time.sleep(min(2 ** attempt, 15))
                continue
            if response.status_code >= 400:
                detail = response.text[:500]
                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = TTSError(f"TTS HTTP {response.status_code}: {detail}")
                    time.sleep(min(3 * (attempt + 1), 30))
                    continue
                raise TTSError(f"TTS HTTP {response.status_code}: {detail}")

            return self._handle_response(response, out_path, source_text=text)

        raise TTSError(f"TTS request failed after retries: {last_error}")


def build_tts(cfg: dict[str, Any]) -> MiniMaxTTS | OpenAICompatTTS:
    provider = (cfg.get("tts_provider") or "minimax").lower()
    if provider == "openai":
        return OpenAICompatTTS(
            base_url=cfg.get("tts_base_url", ""),
            api_key=cfg.get("tts_api_key", ""),
            model=cfg.get("tts_model") or "tts-1",
            speed=float(cfg.get("tts_speed") or 1.0),
        )
    return MiniMaxTTS(
        base_url=cfg.get("tts_base_url", ""),
        api_key=cfg.get("tts_api_key", ""),
        model=cfg.get("tts_model") or "speech-2.5-hd-preview",
        speed=float(cfg.get("tts_speed") or 1.0),
    )
