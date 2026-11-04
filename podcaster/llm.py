"""A small OpenAI-compatible chat client.

Works with any endpoint that speaks ``POST {base}/chat/completions`` —
MiniMax, OpenAI, DeepSeek, vLLM, Ollama's compat mode, ModelVerse, etc.
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


def _extract_content_from_parts(parts: list) -> str:
    """Extract text from OpenAI-style content parts list."""
    texts = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif "text" in part:
            texts.append(part.get("text", ""))
    return "".join(texts)


def _build_debug_context(payload: dict[str, Any]) -> str:
    """Build a debug string with non-sensitive response metadata."""
    ctx_parts = []
    choices = payload.get("choices") or []
    if choices:
        choice = choices[0]
        if "finish_reason" in choice:
            ctx_parts.append(f"finish_reason={choice['finish_reason']}")
        message = choice.get("message") or {}
        present_fields = [k for k in message.keys() if message.get(k) is not None]
        if present_fields:
            ctx_parts.append(f"message_fields={present_fields}")
    if "usage" in payload:
        usage = payload["usage"]
        ctx_parts.append(f"usage={{prompt={usage.get('prompt_tokens')}, "
                         f"completion={usage.get('completion_tokens')}}}")
    if "model" in payload:
        ctx_parts.append(f"model={payload['model']}")
    return "; ".join(ctx_parts) if ctx_parts else "no metadata"


def _extract_reasoning_details(details: list) -> str:
    """Extract text from reasoning_details array (structured thinking fragments)."""
    if not details or not isinstance(details, list):
        return ""
    texts = []
    for item in details:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            for key in ("text", "content", "value"):
                if key in item and isinstance(item[key], str):
                    texts.append(item[key])
                    break
    return " ".join(texts)


def _extract_text(payload: dict[str, Any]) -> str:
    """Extract response text from an OpenAI-compatible chat completion.
    
    Handles standard content, content parts lists, and reasoning fields
    from reasoning models (DeepSeek-R1, Qwen thinking variants, ModelVerse, etc.).
    
    Field priority:
    1. message.content (string or parts list)
    2. choices[].text (legacy)
    3. message.reasoning_content
    4. message.reasoning
    5. message.reasoning_details (array of fragments)
    """
    choices = payload.get("choices") or []
    if not choices:
        raise LLMError(f"LLM returned no choices: {json.dumps(payload)[:400]}")
    
    choice = choices[0]
    message = choice.get("message") or {}
    
    content = message.get("content")
    if content is None:
        content = choice.get("text") or ""
    
    if isinstance(content, list):
        content = _extract_content_from_parts(content)
    
    content = (content or "").strip()
    
    if not content:
        reasoning_content = message.get("reasoning_content")
        if reasoning_content and isinstance(reasoning_content, str):
            content = reasoning_content.strip()
    
    if not content:
        reasoning = message.get("reasoning")
        if reasoning and isinstance(reasoning, str):
            content = reasoning.strip()
    
    if not content:
        reasoning_details = message.get("reasoning_details")
        if reasoning_details:
            content = _extract_reasoning_details(reasoning_details).strip()
    
    return content


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 0.7, max_tokens: int = 4096, timeout: float = 300.0,
                 token_param: str | None = None):
        """Initialize an OpenAI-compatible chat client.
        
        Args:
            base_url: API endpoint (e.g. https://api.openai.com/v1).
            api_key: Bearer token for authentication.
            model: Model identifier to use.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.
            timeout: HTTP timeout in seconds.
            token_param: Which parameter to use for token limit:
                - "max_tokens": Traditional OpenAI parameter (default for most models).
                - "max_completion_tokens": Newer parameter for reasoning models.
                - None: Auto-detect based on model name and base URL.
        """
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._token_param = token_param

    def _get_token_param_name(self) -> str:
        """Determine which token parameter to use for this model/endpoint.
        
        Returns "max_completion_tokens" for reasoning models that require it,
        otherwise returns "max_tokens" for broader compatibility.
        """
        if self._token_param:
            return self._token_param
        
        model_lower = (self.model or "").lower()
        url_lower = (self.base_url or "").lower()
        
        reasoning_model_patterns = (
            "deepseek-r1", "deepseek-reasoner",
            "qwen-", "qwen2", "qwq",
            "o1-", "o1-mini", "o1-preview", "o3-",
        )
        reasoning_url_patterns = (
            "modelverse", "astraflow",
        )
        
        for pattern in reasoning_model_patterns:
            if pattern in model_lower:
                return "max_completion_tokens"
        
        for pattern in reasoning_url_patterns:
            if pattern in url_lower:
                return "max_completion_tokens"
        
        return "max_tokens"

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
        effective_max = max_tokens or self.max_tokens
        token_param_name = self._get_token_param_name()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            token_param_name: effective_max,
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

            raw_text = _extract_text(data)
            text = strip_reasoning(raw_text)
            if not text:
                debug_ctx = _build_debug_context(data)
                last_error = LLMError(
                    f"LLM returned an empty response after extraction "
                    f"(raw_len={len(raw_text)}, {debug_ctx})"
                )
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
