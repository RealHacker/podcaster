"""Tests for LLM client text extraction and reasoning model support."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from podcaster.llm import (
    _extract_text,
    _extract_content_from_parts,
    _extract_reasoning_details,
    _build_debug_context,
    strip_reasoning,
    LLMError,
)

fails = []


def check(name: str, condition: bool, msg: str = "") -> None:
    if not condition:
        fails.append(f"{name}: {msg}" if msg else name)
        print(f"FAIL: {name} - {msg}")
    else:
        print(f"PASS: {name}")


# --------------------------------------------------------------------------- #
# strip_reasoning tests
# --------------------------------------------------------------------------- #

check(
    "strip_reasoning: empty string",
    strip_reasoning("") == "",
    f"got {strip_reasoning('')!r}"
)

check(
    "strip_reasoning: no tags",
    strip_reasoning("Hello world") == "Hello world",
    f"got {strip_reasoning('Hello world')!r}"
)

think_full = "<think>internal reasoning</think>Final answer here."
check(
    "strip_reasoning: removes complete think tag",
    strip_reasoning(think_full) == "Final answer here.",
    f"got {strip_reasoning(think_full)!r}"
)

think_multi = "<think>step 1</think>Middle<think>step 2</think>End"
check(
    "strip_reasoning: removes multiple think tags",
    strip_reasoning(think_multi) == "MiddleEnd",
    f"got {strip_reasoning(think_multi)!r}"
)

think_open = "Start<think>reasoning that never closes"
check(
    "strip_reasoning: removes unclosed think tag",
    strip_reasoning(think_open) == "Start",
    f"got {strip_reasoning(think_open)!r}"
)

think_only = "<think>Only reasoning, no answer</think>"
check(
    "strip_reasoning: think tag only leaves empty",
    strip_reasoning(think_only) == "",
    f"got {strip_reasoning(think_only)!r}"
)

think_with_attrs = "<think type='deep'>reasoning</think>Answer"
check(
    "strip_reasoning: handles think with attributes",
    strip_reasoning(think_with_attrs) == "Answer",
    f"got {strip_reasoning(think_with_attrs)!r}"
)

# --------------------------------------------------------------------------- #
# _extract_content_from_parts tests
# --------------------------------------------------------------------------- #

parts_simple = [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]
check(
    "_extract_content_from_parts: simple text parts",
    _extract_content_from_parts(parts_simple) == "Hello world",
    f"got {_extract_content_from_parts(parts_simple)!r}"
)

parts_mixed = [{"type": "text", "text": "A"}, {"type": "image_url", "url": "..."}, {"type": "text", "text": "B"}]
check(
    "_extract_content_from_parts: skips non-text parts",
    _extract_content_from_parts(parts_mixed) == "AB",
    f"got {_extract_content_from_parts(parts_mixed)!r}"
)

parts_legacy = [{"text": "Legacy"}, {"text": " format"}]
check(
    "_extract_content_from_parts: handles legacy text-only format",
    _extract_content_from_parts(parts_legacy) == "Legacy format",
    f"got {_extract_content_from_parts(parts_legacy)!r}"
)

check(
    "_extract_content_from_parts: empty list",
    _extract_content_from_parts([]) == "",
    f"got {_extract_content_from_parts([])!r}"
)

parts_non_dict = ["not a dict", {"text": "Valid"}]
check(
    "_extract_content_from_parts: skips non-dict entries",
    _extract_content_from_parts(parts_non_dict) == "Valid",
    f"got {_extract_content_from_parts(parts_non_dict)!r}"
)

# --------------------------------------------------------------------------- #
# _extract_reasoning_details tests
# --------------------------------------------------------------------------- #

details_strings = ["Step one.", "Step two.", "Conclusion."]
check(
    "_extract_reasoning_details: joins string fragments",
    _extract_reasoning_details(details_strings) == "Step one. Step two. Conclusion.",
    f"got {_extract_reasoning_details(details_strings)!r}"
)

details_dicts_text = [{"text": "First"}, {"text": "Second"}]
check(
    "_extract_reasoning_details: extracts text key from dicts",
    _extract_reasoning_details(details_dicts_text) == "First Second",
    f"got {_extract_reasoning_details(details_dicts_text)!r}"
)

details_dicts_content = [{"content": "A"}, {"content": "B"}]
check(
    "_extract_reasoning_details: extracts content key from dicts",
    _extract_reasoning_details(details_dicts_content) == "A B",
    f"got {_extract_reasoning_details(details_dicts_content)!r}"
)

details_dicts_value = [{"value": "X"}, {"value": "Y"}]
check(
    "_extract_reasoning_details: extracts value key from dicts",
    _extract_reasoning_details(details_dicts_value) == "X Y",
    f"got {_extract_reasoning_details(details_dicts_value)!r}"
)

details_mixed = ["Intro", {"text": "Middle"}, {"content": "End"}]
check(
    "_extract_reasoning_details: handles mixed types",
    _extract_reasoning_details(details_mixed) == "Intro Middle End",
    f"got {_extract_reasoning_details(details_mixed)!r}"
)

check(
    "_extract_reasoning_details: empty list",
    _extract_reasoning_details([]) == "",
    f"got {_extract_reasoning_details([])!r}"
)

check(
    "_extract_reasoning_details: None input",
    _extract_reasoning_details(None) == "",
    f"got {_extract_reasoning_details(None)!r}"
)

# --------------------------------------------------------------------------- #
# _extract_text tests - standard OpenAI format
# --------------------------------------------------------------------------- #

payload_standard = {
    "choices": [{
        "message": {"role": "assistant", "content": "Standard response"},
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: standard content string",
    _extract_text(payload_standard) == "Standard response",
    f"got {_extract_text(payload_standard)!r}"
)

payload_whitespace = {
    "choices": [{
        "message": {"role": "assistant", "content": "  Padded  "},
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: trims whitespace",
    _extract_text(payload_whitespace) == "Padded",
    f"got {_extract_text(payload_whitespace)!r}"
)

payload_text_field = {
    "choices": [{
        "text": "Legacy text field",
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: falls back to choices[].text",
    _extract_text(payload_text_field) == "Legacy text field",
    f"got {_extract_text(payload_text_field)!r}"
)

# --------------------------------------------------------------------------- #
# _extract_text tests - content parts list
# --------------------------------------------------------------------------- #

payload_parts = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Part one"}, {"type": "text", "text": " part two"}]
        },
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: handles content parts list",
    _extract_text(payload_parts) == "Part one part two",
    f"got {_extract_text(payload_parts)!r}"
)

# --------------------------------------------------------------------------- #
# _extract_text tests - reasoning_content (ModelVerse / DeepSeek-R1)
# --------------------------------------------------------------------------- #

payload_reasoning_only = {
    "model": "deepseek-r1",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": None,
            "reasoning_content": "The actual answer from reasoning model"
        },
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: extracts reasoning_content when content is None",
    _extract_text(payload_reasoning_only) == "The actual answer from reasoning model",
    f"got {_extract_text(payload_reasoning_only)!r}"
)

payload_reasoning_empty_string = {
    "model": "qwen-thinking",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Answer in reasoning_content field"
        },
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: extracts reasoning_content when content is empty string",
    _extract_text(payload_reasoning_empty_string) == "Answer in reasoning_content field",
    f"got {_extract_text(payload_reasoning_empty_string)!r}"
)

payload_both_fields = {
    "model": "deepseek-r1",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "Primary content",
            "reasoning_content": "Reasoning steps"
        },
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: prefers content over reasoning_content when both present",
    _extract_text(payload_both_fields) == "Primary content",
    f"got {_extract_text(payload_both_fields)!r}"
)

payload_whitespace_content_reasoning = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "   ",
            "reasoning_content": "Fallback answer"
        },
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: uses reasoning_content when content is whitespace-only",
    _extract_text(payload_whitespace_content_reasoning) == "Fallback answer",
    f"got {_extract_text(payload_whitespace_content_reasoning)!r}"
)

# --------------------------------------------------------------------------- #
# _extract_text tests - reasoning field (ModelVerse alternative)
# --------------------------------------------------------------------------- #

payload_reasoning_field = {
    "model": "modelverse-reasoning",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": None,
            "reasoning": "Answer from reasoning field"
        },
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: extracts reasoning field when content is None",
    _extract_text(payload_reasoning_field) == "Answer from reasoning field",
    f"got {_extract_text(payload_reasoning_field)!r}"
)

payload_reasoning_content_preferred = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "",
            "reasoning_content": "From reasoning_content",
            "reasoning": "From reasoning"
        },
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: prefers reasoning_content over reasoning",
    _extract_text(payload_reasoning_content_preferred) == "From reasoning_content",
    f"got {_extract_text(payload_reasoning_content_preferred)!r}"
)

# --------------------------------------------------------------------------- #
# _extract_text tests - reasoning_details (structured fragments)
# --------------------------------------------------------------------------- #

payload_reasoning_details = {
    "model": "modelverse-thinking",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": None,
            "reasoning_details": [
                {"text": "First thought."},
                {"text": "Second thought."},
                {"text": "Final answer."}
            ]
        },
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: extracts from reasoning_details array",
    _extract_text(payload_reasoning_details) == "First thought. Second thought. Final answer.",
    f"got {_extract_text(payload_reasoning_details)!r}"
)

payload_reasoning_details_strings = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "",
            "reasoning_details": ["Step 1", "Step 2", "Result"]
        },
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: extracts from reasoning_details string array",
    _extract_text(payload_reasoning_details_strings) == "Step 1 Step 2 Result",
    f"got {_extract_text(payload_reasoning_details_strings)!r}"
)

payload_all_reasoning_fields = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "Primary",
            "reasoning_content": "Secondary",
            "reasoning": "Tertiary",
            "reasoning_details": [{"text": "Last"}]
        },
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: content takes priority over all reasoning fields",
    _extract_text(payload_all_reasoning_fields) == "Primary",
    f"got {_extract_text(payload_all_reasoning_fields)!r}"
)

payload_reasoning_details_last = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": None,
            "reasoning_content": "",
            "reasoning": "",
            "reasoning_details": [{"text": "Only in details"}]
        },
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: falls back to reasoning_details when other fields empty",
    _extract_text(payload_reasoning_details_last) == "Only in details",
    f"got {_extract_text(payload_reasoning_details_last)!r}"
)

# --------------------------------------------------------------------------- #
# _extract_text tests - error cases
# --------------------------------------------------------------------------- #

try:
    _extract_text({"choices": []})
    fails.append("_extract_text: should raise on empty choices")
    print("FAIL: _extract_text: should raise on empty choices")
except LLMError as e:
    check(
        "_extract_text: raises LLMError on empty choices",
        "no choices" in str(e).lower(),
        f"wrong error message: {e}"
    )

try:
    _extract_text({})
    fails.append("_extract_text: should raise on missing choices")
    print("FAIL: _extract_text: should raise on missing choices")
except LLMError as e:
    check(
        "_extract_text: raises LLMError on missing choices",
        "no choices" in str(e).lower(),
        f"wrong error message: {e}"
    )

payload_truly_empty = {
    "choices": [{
        "message": {"role": "assistant", "content": None},
        "finish_reason": "stop"
    }]
}
check(
    "_extract_text: returns empty string when truly empty",
    _extract_text(payload_truly_empty) == "",
    f"got {_extract_text(payload_truly_empty)!r}"
)

# --------------------------------------------------------------------------- #
# _build_debug_context tests
# --------------------------------------------------------------------------- #

debug_full = {
    "model": "gpt-4",
    "choices": [{
        "message": {"role": "assistant", "content": "test"},
        "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20}
}
ctx = _build_debug_context(debug_full)
check(
    "_build_debug_context: includes finish_reason",
    "finish_reason=stop" in ctx,
    f"got {ctx!r}"
)
check(
    "_build_debug_context: includes model",
    "model=gpt-4" in ctx,
    f"got {ctx!r}"
)
check(
    "_build_debug_context: includes usage",
    "prompt=10" in ctx and "completion=20" in ctx,
    f"got {ctx!r}"
)

debug_minimal = {"choices": [{"message": {}}]}
ctx_min = _build_debug_context(debug_minimal)
check(
    "_build_debug_context: handles minimal payload",
    isinstance(ctx_min, str),
    f"got {ctx_min!r}"
)

debug_reasoning_fields = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": None,
            "reasoning_content": "answer"
        },
        "finish_reason": "stop"
    }]
}
ctx_reason = _build_debug_context(debug_reasoning_fields)
check(
    "_build_debug_context: shows reasoning_content in message_fields",
    "reasoning_content" in ctx_reason,
    f"got {ctx_reason!r}"
)

# --------------------------------------------------------------------------- #
# Integration: strip_reasoning + _extract_text
# --------------------------------------------------------------------------- #

payload_think_in_content = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "<think>Let me reason...</think>Here is the final answer."
        },
        "finish_reason": "stop"
    }]
}
raw = _extract_text(payload_think_in_content)
final = strip_reasoning(raw)
check(
    "integration: extracts and strips think from content",
    final == "Here is the final answer.",
    f"got {final!r}"
)

payload_think_in_reasoning = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "",
            "reasoning_content": "<think>Internal</think>The actual output"
        },
        "finish_reason": "stop"
    }]
}
raw2 = _extract_text(payload_think_in_reasoning)
final2 = strip_reasoning(raw2)
check(
    "integration: extracts from reasoning_content and strips think",
    final2 == "The actual output",
    f"got {final2!r}"
)

payload_all_think = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "<think>Only thinking, no answer</think>"
        },
        "finish_reason": "stop"
    }]
}
raw3 = _extract_text(payload_all_think)
final3 = strip_reasoning(raw3)
check(
    "integration: all-think content becomes empty after stripping",
    final3 == "",
    f"got {final3!r}"
)

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

print()
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL LLM EXTRACTION TESTS PASSED")
