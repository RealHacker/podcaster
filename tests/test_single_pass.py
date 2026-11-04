"""Tests for single-pass vs multi-segment generation path selection.

Also tests new parsing patterns for numbered-word artifacts from sample 2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from podcaster.script import (
    Host,
    parse_turns,
    _is_planning_line,
    _clean_dialogue_text,
    _should_use_single_pass,
    SINGLE_PASS_SOURCE_CHARS,
    SINGLE_PASS_TARGET_UNITS,
)

fails = []


def check(name: str, condition: bool, msg: str = "") -> None:
    if not condition:
        fails.append(f"{name}: {msg}" if msg else name)
        print(f"FAIL: {name} - {msg}")
    else:
        print(f"PASS: {name}")


hosts = [Host(name="Marcus", gender="male"), Host(name="Sofia", gender="female")]

# --------------------------------------------------------------------------- #
# Single-pass path selection tests
# --------------------------------------------------------------------------- #

print("\n--- Single-pass path selection ---")

check(
    "short_source_single_pass",
    _should_use_single_pass("A" * 5000, 800, "en"),
    "5000 chars source with 800 word target should use single-pass"
)

check(
    "short_source_long_target_single_pass",
    _should_use_single_pass("A" * 5000, 1200, "en"),
    "5000 chars source with 1200 word target should use single-pass"
)

check(
    "long_source_multi_segment",
    not _should_use_single_pass("A" * 15000, 800, "en"),
    "15000 chars source should use multi-segment"
)

check(
    "short_source_very_long_target_multi_segment",
    not _should_use_single_pass("A" * 5000, 2000, "en"),
    "5000 chars source with 2000 word target should use multi-segment"
)

check(
    "threshold_boundary_source",
    _should_use_single_pass("A" * SINGLE_PASS_SOURCE_CHARS, 800, "en"),
    f"exactly {SINGLE_PASS_SOURCE_CHARS} chars should use single-pass"
)

check(
    "over_threshold_source",
    not _should_use_single_pass("A" * (SINGLE_PASS_SOURCE_CHARS + 1), 800, "en"),
    f"{SINGLE_PASS_SOURCE_CHARS + 1} chars should use multi-segment"
)

check(
    "threshold_boundary_target",
    _should_use_single_pass("A" * 5000, SINGLE_PASS_TARGET_UNITS, "en"),
    f"exactly {SINGLE_PASS_TARGET_UNITS} unit target should use single-pass"
)

check(
    "over_threshold_target",
    not _should_use_single_pass("A" * 5000, SINGLE_PASS_TARGET_UNITS + 1, "en"),
    f"{SINGLE_PASS_TARGET_UNITS + 1} unit target should use multi-segment"
)

check(
    "empty_source_single_pass",
    _should_use_single_pass("", 800, "en"),
    "empty source should use single-pass"
)

# --------------------------------------------------------------------------- #
# Sample 2 numbered-word artifacts tests
# --------------------------------------------------------------------------- #

print("\n--- Sample 2 numbered-word patterns ---")

# From the actual sample: numbered words with trailing punctuation
# Note: When a line has both good content AND numbered cruft, we want to
# CLEAN the suffix (via _clean_dialogue_text) rather than reject the whole line.
# This test verifies the pure numbered-word sequence is detected.
pure_numbered_sequence = (
    "The(1) creator2 economy3 is4 forecast5 to6 grow7 from8 two9 hundred10 "
    "fifty11 to12 four13 hundred14 eighty15 billion16 dollars17."
)

check(
    "_is_planning_line: pure numbered sequence",
    _is_planning_line(pure_numbered_sequence),
    "pure numbered-word sequence should be flagged as planning"
)

# Mixed content is handled by _clean_dialogue_text, not _is_planning_line
sample2_mixed_line = (
    "Harder, but it puts the leverage back in your hands. The(1) creator2 "
    "economy3 is4 forecast5..."
)
cleaned = _clean_dialogue_text(sample2_mixed_line)

check(
    "_clean_dialogue_text: mixed content cleaned",
    "Harder, but it puts" in cleaned and "creator2" not in cleaned,
    f"mixed content should preserve good part, strip numbered suffix: {cleaned}"
)

# From the sample: truncated planning cruft "(16? Goldman1 Sachs2 forecasts"
sample2_truncated_cruft = "(16? Goldman1 Sachs2 forecasts"

check(
    "_is_planning_line: truncated planning cruft",
    _is_planning_line(sample2_truncated_cruft),
    "'(16? Goldman1 Sachs2' pattern should be flagged as planning"
)

# Numbered words with trailing period
check(
    "_is_planning_line: numbered with trailing period",
    _is_planning_line("Perfect1 Join2 the3 workshop4 now5."),
    "numbered sequence ending with period should be flagged"
)

# Clean dialogue mixed with numbered suffix
clean_with_numbered_suffix = "This is clean dialogue. word1 word2 word3"

check(
    "_clean_dialogue_text: strips numbered suffix",
    "word1" not in _clean_dialogue_text(clean_with_numbered_suffix),
    "numbered-word suffix should be stripped"
)

# Numbered suffix with trailing punctuation
numbered_with_punct = "The idea works word1 word2 word3."
cleaned = _clean_dialogue_text(numbered_with_punct)

check(
    "_clean_dialogue_text: strips numbered suffix with punctuation",
    "word1" not in cleaned and "idea works" in cleaned,
    f"numbered suffix with period should be stripped, got: {cleaned}"
)

# --------------------------------------------------------------------------- #
# parse_turns integration tests for sample 2 patterns
# --------------------------------------------------------------------------- #

print("\n--- Sample 2 full line parsing ---")

sample2_raw = """Marcus: Harder, but it puts the leverage back in your hands. The(1) creator2 economy3 is4 forecast5 to6 grow7 from8 two9 hundred10 fifty11 to12 four13 hundred14 eighty15 billion16 dollars17. Creators18 are19 now20 passing21 traditional22 media23 in24 ad25 revenue26, and27 brand28 spending29 on30 creators31 has32 tripled33.
Sofia: So what are the two practical paths? (16? Goldman1 Sachs2 forecasts
Marcus: Build your own creative team—personal brand, audience, products, email list.
"""

turns = parse_turns(sample2_raw, hosts)
texts = [t.text for t in turns]

# The numbered suffix is stripped, and the clean parts are preserved.
# Line 1: "Harder, but it puts..." is kept, numbered suffix stripped
# Line 2: Either filtered entirely OR truncated cruft stripped
# Line 3: Clean line preserved
check(
    "sample2: clean parts extracted",
    len(turns) >= 1 and len(turns) <= 3,
    f"expected 1-3 turns with numbered cruft stripped, got {len(turns)}: {texts}"
)

check(
    "sample2: no numbered words in output",
    not any("creator2" in t for t in texts),
    f"'creator2' should not appear in output: {texts}"
)

check(
    "sample2: no truncated cruft in output",
    not any("(16?" in t for t in texts),
    f"'(16?' should not appear in output: {texts}"
)

check(
    "sample2: clean line preserved",
    any("Build your own" in t or "personal brand" in t for t in texts),
    f"clean 'Build your own creative team' should be preserved: {texts}"
)

# Test that good content before the numbered suffix is handled
mixed_line = """Marcus: This is the core message. Then1 word2 continues3 here4."""
mixed_turns = parse_turns(mixed_line, hosts)

check(
    "mixed_line: either filtered or cleaned",
    len(mixed_turns) == 0 or not any("word2" in t.text for t in mixed_turns),
    "numbered words should be filtered or stripped"
)

# --------------------------------------------------------------------------- #
# Test repetitive content patterns from sample 2
# --------------------------------------------------------------------------- #

print("\n--- Repetition pattern detection ---")

# Lines that repeat the same statistic
repetitive_sample = """Marcus: The creator economy is forecast to grow from two hundred fifty to four hundred eighty billion dollars.
Sofia: Why does this matter right now?
Marcus: The creator economy is forecast to grow from two hundred fifty to four hundred eighty billion dollars. Creators are now passing traditional media in ad revenue.
"""

rep_turns = parse_turns(repetitive_sample, hosts)

check(
    "repetitive_sample: all lines parsed (filter does not block repetition)",
    len(rep_turns) >= 2,
    "repeated content lines should parse (anti-repetition is in generation, not parsing)"
)

# --------------------------------------------------------------------------- #
# Edge case: line starting with parenthesis number
# --------------------------------------------------------------------------- #

print("\n--- Edge cases ---")

paren_start = """Marcus: (1) First point here
Sofia: (2) Second point
Marcus: Normal dialogue follows."""

paren_turns = parse_turns(paren_start, hosts)

check(
    "parenthesis_numbered_lines",
    len(paren_turns) == 3,
    f"lines starting with (1), (2) should not be filtered: got {len(paren_turns)}"
)

# Ensure we don't over-filter simple numeric references
numeric_ref = """Marcus: The creator economy grew from 250 to 480 billion.
Sofia: Goldman Sachs forecasts similar growth."""

num_turns = parse_turns(numeric_ref, hosts)

check(
    "normal_numeric_references_preserved",
    len(num_turns) == 2 and "250" in num_turns[0].text,
    "normal numeric references should be preserved"
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
print("ALL SINGLE-PASS AND SAMPLE 2 TESTS PASSED")
