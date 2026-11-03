"""Tests for ModelVerse Qwen TTS dialect handling.

Covers:
- Dialect detection (is_qwen_dialect)
- Payload shape (no speed/response_format for Qwen)
- JSON+URL response handling
- Character limit (600 chars for Qwen)
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from podcaster.tts import (
    QWEN_MAX_CHARS,
    OpenAICompatTTS,
    is_qwen_dialect,
    split_for_tts,
)
from podcaster.voices import get_default_qwen_voice, get_qwen_voices

fails = []


# --- is_qwen_dialect detection ---
def test_dialect_detection():
    global fails

    if not is_qwen_dialect("qwen3-tts-flash", "https://api.example.com/v1"):
        fails.append("should detect qwen dialect from model name")

    if not is_qwen_dialect("QWEN3-TTS-FLASH", "https://api.example.com/v1"):
        fails.append("should detect qwen dialect case-insensitively")

    if not is_qwen_dialect("tts-1", "https://api.modelverse.cn/v1"):
        fails.append("should detect qwen dialect from modelverse URL")

    if is_qwen_dialect("tts-1", "https://api.openai.com/v1"):
        fails.append("should NOT detect qwen dialect for standard OpenAI")

    if is_qwen_dialect("gpt-tts", "https://api.deepseek.com/v1"):
        fails.append("should NOT detect qwen dialect for other providers")


test_dialect_detection()
print("dialect detection:", "PASS" if not fails else f"FAIL {fails}")


# --- Payload shape for Qwen vs OpenAI ---
def test_payload_shape():
    global fails

    qwen_tts = OpenAICompatTTS(
        base_url="https://api.modelverse.cn/v1",
        api_key="test-key",
        model="qwen3-tts-flash",
    )
    qwen_payload = qwen_tts._build_payload("Hello", "Cherry", "en", None)

    if "speed" in qwen_payload:
        fails.append("Qwen payload should NOT contain 'speed'")
    if "response_format" in qwen_payload:
        fails.append("Qwen payload should NOT contain 'response_format'")
    if qwen_payload.get("model") != "qwen3-tts-flash":
        fails.append(f"Qwen payload model mismatch: {qwen_payload.get('model')}")
    if qwen_payload.get("voice") != "Cherry":
        fails.append(f"Qwen payload voice mismatch: {qwen_payload.get('voice')}")
    if qwen_payload.get("input") != "Hello":
        fails.append(f"Qwen payload input mismatch: {qwen_payload.get('input')}")
    if qwen_payload.get("metadata", {}).get("language_type") != "English":
        fails.append(f"Qwen payload language_type mismatch: {qwen_payload}")

    qwen_zh_payload = qwen_tts._build_payload("你好", "Cherry", "zh", None)
    if qwen_zh_payload.get("metadata", {}).get("language_type") != "Chinese":
        fails.append(f"Qwen ZH payload language_type mismatch: {qwen_zh_payload}")

    openai_tts = OpenAICompatTTS(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="tts-1",
        speed=1.2,
    )
    openai_payload = openai_tts._build_payload("Hello", "alloy", "en", None)

    if openai_payload.get("speed") != 1.2:
        fails.append(f"OpenAI payload should contain speed: {openai_payload}")
    if openai_payload.get("response_format") != "mp3":
        fails.append(f"OpenAI payload should contain response_format: {openai_payload}")
    if "metadata" in openai_payload:
        fails.append("OpenAI payload should NOT contain metadata")


test_payload_shape()
print("payload shape:", "PASS" if not fails else f"FAIL {fails}")


# --- max_chars method ---
def test_max_chars():
    global fails

    qwen_tts = OpenAICompatTTS(
        base_url="https://api.modelverse.cn/v1",
        api_key="test-key",
        model="qwen3-tts-flash",
    )
    if qwen_tts.max_chars() != QWEN_MAX_CHARS:
        fails.append(f"Qwen max_chars should be {QWEN_MAX_CHARS}, got {qwen_tts.max_chars()}")

    openai_tts = OpenAICompatTTS(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="tts-1",
    )
    if openai_tts.max_chars() != 4000:
        fails.append(f"OpenAI max_chars should be 4000, got {openai_tts.max_chars()}")


test_max_chars()
print("max_chars:", "PASS" if not fails else f"FAIL {fails}")


# --- split_for_tts with Qwen limit ---
def test_split_for_qwen():
    global fails

    long_text = "A" * 1200
    pieces = split_for_tts(long_text, max_chars=QWEN_MAX_CHARS)
    if len(pieces) < 2:
        fails.append(f"1200 chars should split into at least 2 pieces at 600 limit, got {len(pieces)}")
    for piece in pieces:
        if len(piece) > QWEN_MAX_CHARS:
            fails.append(f"Piece exceeds Qwen limit: {len(piece)} > {QWEN_MAX_CHARS}")


test_split_for_qwen()
print("split_for_qwen:", "PASS" if not fails else f"FAIL {fails}")


# --- JSON+URL response handling (mocked) - WAV output when caller wants WAV ---
def test_json_url_response():
    """When caller requests .wav and API returns WAV, no conversion needed."""
    global fails

    qwen_tts = OpenAICompatTTS(
        base_url="https://api.modelverse.cn/v1",
        api_key="test-key",
        model="qwen3-tts-flash",
    )

    mock_json_response = MagicMock()
    mock_json_response.headers = {"content-type": "application/json"}
    mock_json_response.json.return_value = {
        "output": {
            "audio": {
                "url": "https://example.com/audio.wav"
            }
        }
    }

    mock_audio_response = MagicMock()
    mock_audio_response.content = b"RIFF" + b"\x00" * 100
    mock_audio_response.headers = {"content-type": "audio/wav"}
    mock_audio_response.raise_for_status = MagicMock()

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test.wav"

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_audio_response
            mock_client_cls.return_value = mock_client

            result = qwen_tts._handle_response(mock_json_response, out_path)

            if result.format != "wav":
                fails.append(f"Expected wav format, got {result.format}")
            if not result.path.exists():
                fails.append("Output file should exist")
            if result.path != out_path:
                fails.append(f"Expected path {out_path}, got {result.path}")


test_json_url_response()
print("json_url_response:", "PASS" if not fails else f"FAIL {fails}")


# --- Binary response handling (standard OpenAI) ---
def test_binary_response():
    global fails

    openai_tts = OpenAICompatTTS(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="tts-1",
    )

    mock_response = MagicMock()
    mock_response.headers = {"content-type": "audio/mpeg"}
    mock_response.content = b"\xff\xfb\x90\x00" + b"\x00" * 100

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test.mp3"
        result = openai_tts._handle_response(mock_response, out_path)

        if result.format != "mp3":
            fails.append(f"Expected mp3 format, got {result.format}")
        if not out_path.exists():
            fails.append("Output file should exist")
        if out_path.read_bytes() != mock_response.content:
            fails.append("File content mismatch")


test_binary_response()
print("binary_response:", "PASS" if not fails else f"FAIL {fails}")


# --- Qwen voices ---
def test_qwen_voices():
    global fails

    voices = get_qwen_voices()
    if len(voices) < 4:
        fails.append(f"Expected at least 4 Qwen voices, got {len(voices)}")

    voice_ids = {v["id"] for v in voices}
    required = {"Cherry", "Ethan", "Serena", "Chelsie"}
    missing = required - voice_ids
    if missing:
        fails.append(f"Missing required Qwen voices: {missing}")

    default = get_default_qwen_voice()
    if default not in voice_ids:
        fails.append(f"Default Qwen voice '{default}' not in voice list")


test_qwen_voices()
print("qwen_voices:", "PASS" if not fails else f"FAIL {fails}")


# --- Missing audio URL error handling ---
def test_missing_audio_url():
    global fails

    qwen_tts = OpenAICompatTTS(
        base_url="https://api.modelverse.cn/v1",
        api_key="test-key",
        model="qwen3-tts-flash",
    )

    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"output": {}}

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test.mp3"
        try:
            qwen_tts._handle_response(mock_response, out_path)
            fails.append("Should raise TTSError for missing audio URL")
        except Exception as e:
            if "missing output.audio.url" not in str(e).lower():
                fails.append(f"Unexpected error message: {e}")


test_missing_audio_url()
print("missing_audio_url:", "PASS" if not fails else f"FAIL {fails}")


# --- WAV to MP3 conversion - file exists at requested path ---
def test_wav_to_mp3_conversion_at_requested_path():
    """Qwen returns WAV but caller wants MP3 - file should exist at requested .mp3 path."""
    global fails

    qwen_tts = OpenAICompatTTS(
        base_url="https://api.modelverse.cn/v1",
        api_key="test-key",
        model="qwen3-tts-flash",
    )

    mock_json_response = MagicMock()
    mock_json_response.headers = {"content-type": "application/json"}
    mock_json_response.json.return_value = {
        "output": {
            "audio": {
                "url": "https://example.com/audio.wav"
            }
        }
    }

    wav_header = (
        b"RIFF" + b"\x24\x00\x00\x00" +
        b"WAVE" +
        b"fmt " + b"\x10\x00\x00\x00" +
        b"\x01\x00\x01\x00" +
        b"\x44\xac\x00\x00" +
        b"\x88\x58\x01\x00" +
        b"\x02\x00\x10\x00" +
        b"data" + b"\x00\x00\x00\x00"
    )
    mock_audio_response = MagicMock()
    mock_audio_response.content = wav_header + b"\x00" * 100
    mock_audio_response.headers = {"content-type": "audio/wav"}
    mock_audio_response.raise_for_status = MagicMock()

    with tempfile.TemporaryDirectory() as tmp:
        requested_path = Path(tmp) / "probe.mp3"

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_audio_response
            mock_client_cls.return_value = mock_client

            result = qwen_tts._handle_response(
                mock_json_response, requested_path, source_text="Test voice"
            )

            if not requested_path.exists():
                fails.append(
                    f"File should exist at requested path {requested_path}, "
                    f"but result.path is {result.path}"
                )

            if result.path != requested_path:
                fails.append(
                    f"result.path should be {requested_path}, got {result.path}"
                )

            if result.chars != len("Test voice"):
                fails.append(
                    f"result.chars should be text length (10), got {result.chars}"
                )

            if result.format != "mp3":
                fails.append(f"result.format should be 'mp3', got {result.format}")


test_wav_to_mp3_conversion_at_requested_path()
print("wav_to_mp3_conversion:", "PASS" if not fails else f"FAIL {fails}")


# --- chars should be text length, not audio bytes ---
def test_chars_is_text_length():
    """ClipResult.chars should reflect input text length, not audio byte count."""
    global fails

    openai_tts = OpenAICompatTTS(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="tts-1",
    )

    mock_response = MagicMock()
    mock_response.headers = {"content-type": "audio/mpeg"}
    mock_response.content = b"\xff\xfb\x90\x00" + b"\x00" * 5000

    test_text = "Hello, this is a test."

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test.mp3"
        result = openai_tts._handle_response(
            mock_response, out_path, source_text=test_text
        )

        if result.chars != len(test_text):
            fails.append(
                f"chars should be text length ({len(test_text)}), "
                f"got {result.chars}"
            )


test_chars_is_text_length()
print("chars_is_text_length:", "PASS" if not fails else f"FAIL {fails}")

print()
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL QWEN TTS CHECKS PASSED")
