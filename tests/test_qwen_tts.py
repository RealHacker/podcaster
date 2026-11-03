"""Tests for ModelVerse Qwen TTS dialect handling.

Covers:
- Dialect detection (is_qwen_dialect)
- Payload shape (no speed/response_format for Qwen)
- JSON+URL response handling
- Character limit (600 chars for Qwen)
- Qwen voice assignment and listing
- Subtitle fallback for OpenAI-compatible TTS
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
from podcaster.voices import (
    assign_voices,
    get_default_qwen_voice,
    get_qwen_voices,
    list_voices,
    load_catalog,
    QWEN_VOICES,
)
from podcaster.pipeline import build_hosts, PodcastRequest
from podcaster.script import Host

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


# --- Qwen voice assignment preference ---
def test_qwen_voice_assignment():
    """When is_qwen=True, assign_voices should use Qwen voices (Cherry, Ethan, etc.)."""
    global fails

    hosts = [
        Host(name="Alice", gender="female"),
        Host(name="Bob", gender="male"),
    ]

    assigned = assign_voices(hosts, language="en", is_qwen=True)
    qwen_ids = {v["id"] for v in QWEN_VOICES}

    for voice_id in assigned:
        if voice_id not in qwen_ids:
            fails.append(f"Qwen mode assigned non-Qwen voice: {voice_id}")

    if len(set(assigned)) != len(assigned):
        fails.append(f"Qwen mode assigned duplicate voices: {assigned}")


test_qwen_voice_assignment()
print("qwen_voice_assignment:", "PASS" if not fails else f"FAIL {fails}")


# --- Qwen voice listing ---
def test_qwen_voice_listing():
    """list_voices with is_qwen=True should return only Qwen voices."""
    global fails

    qwen_list = list_voices(is_qwen=True)
    qwen_ids = {v["id"] for v in QWEN_VOICES}

    for voice in qwen_list:
        if voice["id"] not in qwen_ids:
            fails.append(f"list_voices(is_qwen=True) returned non-Qwen voice: {voice['id']}")

    standard_list = list_voices(is_qwen=False)
    if standard_list == qwen_list:
        fails.append("Standard and Qwen voice lists should be different")


test_qwen_voice_listing()
print("qwen_voice_listing:", "PASS" if not fails else f"FAIL {fails}")


# --- load_catalog with is_qwen ---
def test_load_catalog_qwen():
    """load_catalog(is_qwen=True) should return Qwen voices."""
    global fails

    qwen_catalog = load_catalog(is_qwen=True)
    qwen_ids = {v["id"] for v in QWEN_VOICES}

    for voice in qwen_catalog:
        if voice["id"] not in qwen_ids:
            fails.append(f"load_catalog(is_qwen=True) returned non-Qwen voice: {voice['id']}")

    if len(qwen_catalog) != len(QWEN_VOICES):
        fails.append(f"Expected {len(QWEN_VOICES)} Qwen voices, got {len(qwen_catalog)}")


test_load_catalog_qwen()
print("load_catalog_qwen:", "PASS" if not fails else f"FAIL {fails}")


# --- build_hosts with Qwen config ---
def test_build_hosts_qwen_config():
    """build_hosts with Qwen TTS config should assign Qwen voices."""
    global fails

    qwen_cfg = {
        "tts_model": "qwen3-tts-flash",
        "tts_base_url": "https://api.modelverse.cn/v1",
    }

    req = PodcastRequest(host_count=2, host_genders=["male", "female"], language="en")
    hosts = build_hosts(req, qwen_cfg)
    qwen_ids = {v["id"] for v in QWEN_VOICES}

    for host in hosts:
        if host.voice_id not in qwen_ids:
            fails.append(f"build_hosts with Qwen config assigned non-Qwen voice: {host.voice_id}")

    if len({h.voice_id for h in hosts}) != len(hosts):
        fails.append(f"build_hosts assigned duplicate voices: {[h.voice_id for h in hosts]}")


test_build_hosts_qwen_config()
print("build_hosts_qwen_config:", "PASS" if not fails else f"FAIL {fails}")


# --- build_hosts without config still uses MiniMax voices ---
def test_build_hosts_no_config():
    """build_hosts without config should use MiniMax voices (backward compatibility)."""
    global fails

    req = PodcastRequest(host_count=2, host_genders=["male", "female"], language="en")
    hosts = build_hosts(req)
    qwen_ids = {v["id"] for v in QWEN_VOICES}

    has_minimax = False
    for host in hosts:
        if host.voice_id not in qwen_ids:
            has_minimax = True
            break

    if not has_minimax:
        fails.append("build_hosts without config should use MiniMax voices")


test_build_hosts_no_config()
print("build_hosts_no_config:", "PASS" if not fails else f"FAIL {fails}")


# --- Qwen max_chars is used in TTS client ---
def test_qwen_tts_max_chars_method():
    """OpenAICompatTTS.max_chars() should return 600 for Qwen dialect."""
    global fails

    qwen_tts = OpenAICompatTTS(
        base_url="https://api.modelverse.cn/v1",
        api_key="test-key",
        model="qwen3-tts-flash",
    )

    if qwen_tts.max_chars() != QWEN_MAX_CHARS:
        fails.append(f"Qwen TTS max_chars() should be {QWEN_MAX_CHARS}, got {qwen_tts.max_chars()}")

    standard_tts = OpenAICompatTTS(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="tts-1",
    )

    if standard_tts.max_chars() != 4000:
        fails.append(f"Standard TTS max_chars() should be 4000, got {standard_tts.max_chars()}")


test_qwen_tts_max_chars_method()
print("qwen_tts_max_chars_method:", "PASS" if not fails else f"FAIL {fails}")


# --- Gender preference in Qwen voice assignment ---
def test_qwen_gender_preference():
    """Qwen voice assignment should respect gender preferences."""
    global fails

    male_hosts = [Host(name="Bob", gender="male")]
    female_hosts = [Host(name="Alice", gender="female")]

    male_assigned = assign_voices(male_hosts, is_qwen=True)
    female_assigned = assign_voices(female_hosts, is_qwen=True)

    qwen_male_voices = {v["id"] for v in QWEN_VOICES if v.get("gender") == "male"}
    qwen_female_voices = {v["id"] for v in QWEN_VOICES if v.get("gender") == "female"}

    if male_assigned[0] not in qwen_male_voices:
        fails.append(f"Male host should get male Qwen voice, got {male_assigned[0]}")

    if female_assigned[0] not in qwen_female_voices:
        fails.append(f"Female host should get female Qwen voice, got {female_assigned[0]}")


test_qwen_gender_preference()
print("qwen_gender_preference:", "PASS" if not fails else f"FAIL {fails}")


# --- Subtitle fallback for OpenAI-compatible TTS ---
def test_subtitle_fallback_when_enabled():
    """When subtitle_enable=True and no provider cues, fallback generates approximate cues."""
    global fails

    qwen_tts = OpenAICompatTTS(
        base_url="https://api.modelverse.cn/v1",
        api_key="test-key",
        model="qwen3-tts-flash",
    )

    mock_json_response = MagicMock()
    mock_json_response.status_code = 200
    mock_json_response.headers = {"content-type": "application/json"}
    mock_json_response.json.return_value = {
        "output": {
            "audio": {
                "url": "https://example.com/audio.mp3"
            }
        }
    }

    mock_audio_response = MagicMock()
    mock_audio_response.content = b"\xff\xfb\x90\x00" + b"\x00" * 1000
    mock_audio_response.headers = {"content-type": "audio/mpeg"}
    mock_audio_response.raise_for_status = MagicMock()

    test_text = "Hello world. This is a test of subtitle fallback."

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test.mp3"

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_json_response
            mock_client.get.return_value = mock_audio_response
            mock_client_cls.return_value = mock_client

            with patch("podcaster.tts._probe_duration_ms", return_value=5000.0):
                result = qwen_tts.synthesize(
                    test_text, "Cherry", out_path,
                    subtitle_enable=True
                )

                if not result.subtitles:
                    fails.append("subtitles should be generated when subtitle_enable=True")
                elif len(result.subtitles) < 1:
                    fails.append(f"expected at least 1 subtitle cue, got {len(result.subtitles)}")
                else:
                    joined = "".join(c.text for c in result.subtitles)
                    if "Hello" not in joined:
                        fails.append(f"subtitle cues missing expected text, got: {joined}")


test_subtitle_fallback_when_enabled()
print("subtitle_fallback_enabled:", "PASS" if not fails else f"FAIL {fails}")


def test_subtitle_fallback_disabled():
    """When subtitle_enable=False, no subtitle cues should be generated."""
    global fails

    openai_tts = OpenAICompatTTS(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="tts-1",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "audio/mpeg"}
    mock_response.content = b"\xff\xfb\x90\x00" + b"\x00" * 1000

    test_text = "Hello world. This is a test."

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test.mp3"

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = openai_tts.synthesize(
                test_text, "alloy", out_path,
                subtitle_enable=False
            )

            if result.subtitles:
                fails.append(
                    f"subtitles should be empty when subtitle_enable=False, "
                    f"got {len(result.subtitles)} cues"
                )


test_subtitle_fallback_disabled()
print("subtitle_fallback_disabled:", "PASS" if not fails else f"FAIL {fails}")


def test_subtitle_fallback_zero_duration():
    """When audio duration is 0, no subtitle cues should be generated."""
    global fails

    openai_tts = OpenAICompatTTS(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="tts-1",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "audio/mpeg"}
    mock_response.content = b"\xff\xfb\x90\x00" + b"\x00" * 100

    test_text = "Hello world."

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test.mp3"

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            with patch("podcaster.tts._probe_duration_ms", return_value=0.0):
                result = openai_tts.synthesize(
                    test_text, "alloy", out_path,
                    subtitle_enable=True
                )

                if result.subtitles:
                    fails.append(
                        f"subtitles should be empty when duration is 0, "
                        f"got {len(result.subtitles)} cues"
                    )


test_subtitle_fallback_zero_duration()
print("subtitle_fallback_zero_duration:", "PASS" if not fails else f"FAIL {fails}")


print()
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL QWEN TTS CHECKS PASSED")
