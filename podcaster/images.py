"""Image generation for podcast slides.

Talks to the MiniMax text-to-image endpoint (``POST /v1/image_generation``).
Images come back either as a URL or as a base64 blob; the base64 form is
preferred because the URLs only live for 24 hours.
"""
from __future__ import annotations

import base64
import binascii
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx


class ImageError(RuntimeError):
    pass


class ImageSafetyError(ImageError):
    """The provider's content filter rejected the prompt."""


@dataclass
class ImageResult:
    path: Path
    bytes: int = 0


class ImageClientProtocol(Protocol):
    """What the slide planner needs from an image backend."""

    def generate(self, prompt: str, out_path: Path, *, aspect_ratio: str = "16:9",
                 n: int = 1, retries: int = 4) -> ImageResult: ...


def sniff_extension(data: bytes) -> str:
    """Work out the real image format — the API does not always say."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


class MiniMaxImageClient:
    provider = "minimax"

    def __init__(self, base_url: str, api_key: str, model: str = "image-01",
                 timeout: float = 240.0, min_interval: float = 2.0):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = model or "image-01"
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_call = 0.0

    def _endpoint(self) -> str:
        if not self.base_url:
            raise ImageError("The image base URL is not configured")
        if self.base_url.endswith("/image_generation"):
            return self.base_url
        return self.base_url + "/image_generation"

    def _throttle(self) -> None:
        delta = time.monotonic() - self._last_call
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last_call = time.monotonic()

    def _download(self, url: str) -> bytes:
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            response = client.get(url)
        response.raise_for_status()
        return response.content

    def _blob(self, data: dict[str, Any]) -> bytes:
        payload = data.get("data") or {}
        encoded = payload.get("image_base64") or []
        if encoded:
            try:
                return base64.b64decode(encoded[0])
            except (ValueError, binascii.Error) as exc:
                raise ImageError(f"Could not decode the generated image: {exc}") from exc
        urls = payload.get("image_urls") or []
        if urls:
            return self._download(urls[0])
        raise ImageError("The image API returned no image data")

    def generate(self, prompt: str, out_path: Path, *, aspect_ratio: str = "16:9",
                 n: int = 1, retries: int = 4) -> ImageResult:
        """Render one image and write it next to ``out_path``."""
        if not self.api_key:
            raise ImageError("The image API key is not configured")
        prompt = (prompt or "").strip()
        if not prompt:
            raise ImageError("Cannot generate an image from an empty prompt")

        payload = {
            "model": self.model,
            "prompt": prompt[:1500],
            "aspect_ratio": aspect_ratio,
            "response_format": "base64",
            "n": max(1, min(9, int(n))),
            "prompt_optimizer": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        last_error: Exception | None = None
        for attempt in range(retries):
            self._throttle()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self._endpoint(), headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_error = ImageError(f"Image network error: {exc}")
                time.sleep(min(3 * (attempt + 1), 30))
                continue

            if response.status_code >= 400:
                detail = response.text[:500]
                if response.status_code in (429, 500, 502, 503, 504) or "rate limit" in detail.lower():
                    last_error = ImageError(f"Image HTTP {response.status_code}: {detail}")
                    time.sleep(min(4 * (attempt + 1), 40))
                    continue
                raise ImageError(f"Image HTTP {response.status_code}: {detail}")

            try:
                data = response.json()
            except Exception as exc:
                raise ImageError(f"The image API returned invalid JSON: {exc}") from exc

            base_resp = data.get("base_resp") or {}
            code = int(base_resp.get("status_code") or 0)
            message = str(base_resp.get("status_msg") or "unknown error")
            if code != 0:
                if code == 1002 or "rate limit" in message.lower():
                    last_error = ImageError(f"Image API rate limited: {message}")
                    time.sleep(min(5 * (attempt + 1), 45))
                    continue
                if code == 1026:
                    raise ImageSafetyError(f"The image prompt was rejected: {message}")
                raise ImageError(f"Image API error {code}: {message}")

            blob = self._blob(data)
            if not blob:
                raise ImageError("The image API returned an empty image")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            target = out_path.with_suffix(sniff_extension(blob))
            target.write_bytes(blob)
            return ImageResult(path=target, bytes=len(blob))

        raise ImageError(f"The image request failed after {retries} attempts: {last_error}")


def build_image_client(cfg: dict[str, Any]) -> MiniMaxImageClient:
    """Build a client from the app config, defaulting to the TTS credentials."""
    base_url = (cfg.get("image_base_url") or cfg.get("tts_base_url") or "").strip()
    api_key = (cfg.get("image_api_key") or cfg.get("tts_api_key") or "").strip()
    return MiniMaxImageClient(
        base_url=base_url,
        api_key=api_key,
        model=(cfg.get("image_model") or "image-01").strip(),
    )
