"""A small OpenAI-compatible chat client.

Works with any endpoint that speaks ``POST {base}/chat/completions`` —
MiniMax, OpenAI, DeepSeek, vLLM, Ollama's compat mode, etc.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.S | re.I)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>.*\Z", re.S | re.I)


class LLMError(RuntimeError):
    pass


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` scratchpads that reasoning models emit."""
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    return cleaned.strip()


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise LLMError(f"LLM returned no choices: {json.dumps(payload)[:400]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        content = choices[0].get("text") or ""
    if isinstance(content, list):  # some gateways return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return (content or "").strip()


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 0.7, max_tokens: int = 4096, timeout: float = 300.0):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _endpoint(self) -> str:
        base = self.base_url
        if not base:
            raise LLMError("LLM base URL is not configured")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None,
             max_tokens: int | None = None, retries: int = 4) -> str:
        if not self.api_key:
            raise LLMError("LLM API key is not configured")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self._endpoint(), headers=headers, json=payload)
                if response.status_code >= 400:
                    detail = response.text[:600]
                    if response.status_code in (429, 500, 502, 503, 504):
                        last_error = LLMError(f"LLM HTTP {response.status_code}: {detail}")
                        time.sleep(min(2 ** attempt, 20))
                        continue
                    raise LLMError(f"LLM HTTP {response.status_code}: {detail}")
                data = response.json()
            except httpx.TimeoutException as exc:
                last_error = LLMError(f"LLM request timed out: {exc}")
                time.sleep(min(2 ** attempt, 20))
                continue
            except httpx.HTTPError as exc:
                last_error = LLMError(f"LLM network error: {exc}")
                time.sleep(min(2 ** attempt, 20))
                continue
            except json.JSONDecodeError as exc:
                raise LLMError(f"LLM returned invalid JSON: {exc}") from exc

            text = strip_reasoning(_extract_text(data))
            if not text:
                last_error = LLMError("LLM returned an empty response")
                time.sleep(1.5)
                continue
            return text
        raise LLMError(f"LLM request failed after {retries} attempts: {last_error}")

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        raw = self.chat(messages, **kwargs)
        cleaned = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.S)
        if fenced:
            cleaned = fenced.group(1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError as exc:
                    raise LLMError(f"LLM response was not valid JSON: {exc}") from exc
            raise LLMError("LLM response was not valid JSON")

    def list_models(self) -> list[str]:
        """Best-effort model discovery; returns [] when unsupported."""
        url = self.base_url + "/models"
        if not self.base_url:
            return []
        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
            if response.status_code != 200:
                return []
            data = response.json()
            return [m.get("id") for m in data.get("data", []) if m.get("id")]
        except Exception:
            return []
