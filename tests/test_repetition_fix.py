"""Tests for repetition-prevention fixes in script generation.

These tests verify:
1. Prior-context block is included in segment prompts for N>1
2. Planning crumbs (e.g. "Count add.") are cleaned from dialogue
3. Truncated lines (e.g. "17 So") are filtered out
4. Near-duplicate turns are removed
5. Sample artifacts from the bug report are properly handled
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from podcaster.script import (
    Host,
    Turn,
    parse_turns,
    _segment_prompt,
    _prior_context_block,
    _clean_dialogue_text,
    _is_truncated_line,
    _text_similarity,
    _remove_near_duplicates,
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
# _prior_context_block tests
# --------------------------------------------------------------------------- #

check(
    "_prior_context_block: empty list returns empty string",
    _prior_context_block([], "en") == "",
    "empty prior_covered should return empty block"
)

prior_en = _prior_context_block(["Attention is the key skill", "Personal brands are growing"], "en")
check(
    "_prior_context_block: English block contains bullet points",
    "- Attention is the key skill" in prior_en and "- Personal brands are growing" in prior_en,
    "English prior block should contain bullet-pointed outline items"
)

check(
    "_prior_context_block: English block has anti-repetition instruction",
    "do not repeat" in prior_en.lower() or "do not restate" in prior_en.lower(),
    "English prior block should instruct model not to repeat"
)

prior_zh = _prior_context_block(["注意力是最重要的技能", "个人品牌正在崛起"], "zh")
check(
    "_prior_context_block: Chinese block contains bullet points",
    "- 注意力是最重要的技能" in prior_zh and "- 个人品牌正在崛起" in prior_zh,
    "Chinese prior block should contain bullet-pointed outline items"
)

check(
    "_prior_context_block: Chinese block has anti-repetition instruction",
    "不要" in prior_zh and "重" in prior_zh,
    "Chinese prior block should instruct model not to repeat"
)

# --------------------------------------------------------------------------- #
# _segment_prompt prior_covered tests
# --------------------------------------------------------------------------- #

prompt_seg1 = _segment_prompt(
    language="en", style_desc="chitchat", hosts=hosts,
    outline_line="Introduce the attention economy", chunk_text="Source material...",
    unit_target=100, segment_index=0, total_segments=3, title="Test Podcast",
    extra="", role_hint="", prior_covered=None
)

check(
    "_segment_prompt: segment 1 has no ALREADY COVERED block",
    "ALREADY COVERED" not in prompt_seg1,
    "first segment should not have prior context block"
)

prompt_seg2 = _segment_prompt(
    language="en", style_desc="chitchat", hosts=hosts,
    outline_line="Discuss personal branding", chunk_text="Source material...",
    unit_target=100, segment_index=1, total_segments=3, title="Test Podcast",
    extra="", role_hint="", prior_covered=["Introduce the attention economy"]
)

check(
    "_segment_prompt: segment 2 has ALREADY COVERED block",
    "ALREADY COVERED" in prompt_seg2,
    "segment 2+ should have prior context block"
)

check(
    "_segment_prompt: segment 2 contains prior outline",
    "Introduce the attention economy" in prompt_seg2,
    "segment 2 prompt should list previously covered outline"
)

check(
    "_segment_prompt: segment 2 has anti-repetition instruction",
    "do not restate" in prompt_seg2.lower() or "do not repeat" in prompt_seg2.lower(),
    "segment 2 prompt should instruct model not to repeat"
)

prompt_seg2_zh = _segment_prompt(
    language="zh", style_desc="闲聊", hosts=hosts,
    outline_line="讨论个人品牌", chunk_text="素材...",
    unit_target=100, segment_index=1, total_segments=3, title="测试播客",
    extra="", role_hint="", prior_covered=["介绍注意力经济"]
)

check(
    "_segment_prompt: Chinese segment 2 has prior block",
    "已讲过" in prompt_seg2_zh or "不要" in prompt_seg2_zh,
    "Chinese segment 2 should have prior context block"
)

# --------------------------------------------------------------------------- #
# _clean_dialogue_text tests for new patterns
# --------------------------------------------------------------------------- #

check(
    "_clean_dialogue_text: removes 'Count add.'",
    _clean_dialogue_text("That's the positioning exercise. Count add. Two paths.") ==
        "That's the positioning exercise. Two paths.",
    "'Count add.' crumb should be removed"
)

check(
    "_clean_dialogue_text: removes 'Add more.'",
    _clean_dialogue_text("This is a test. Add more. And continue.") ==
        "This is a test. And continue.",
    "'Add more.' crumb should be removed"
)

check(
    "_clean_dialogue_text: removes 'Check this.'",
    "Check this" not in _clean_dialogue_text("This needs work. Check this. And continue."),
    "'Check this.' should be removed"
)

check(
    "_clean_dialogue_text: collapses multiple spaces",
    "  " not in _clean_dialogue_text("Test.  Count add.  More text."),
    "multiple spaces should be collapsed to single space"
)

# --------------------------------------------------------------------------- #
# _is_truncated_line tests
# --------------------------------------------------------------------------- #

check(
    "_is_truncated_line: '17 So' is truncated",
    _is_truncated_line("17 So"),
    "'17 So' should be flagged as truncated"
)

check(
    "_is_truncated_line: '20 Th' is truncated",
    _is_truncated_line("20 Th"),
    "'20 Th' should be flagged as truncated"
)

check(
    "_is_truncated_line: 'Hi!' is not truncated",
    not _is_truncated_line("Hi!"),
    "short sentence ending with punctuation should not be flagged"
)

check(
    "_is_truncated_line: normal sentence is not truncated",
    not _is_truncated_line("This is a normal sentence."),
    "normal sentence should not be flagged"
)

check(
    "_is_truncated_line: very short without punctuation is truncated",
    _is_truncated_line("Ye"),
    "very short text without punctuation should be flagged"
)

# --------------------------------------------------------------------------- #
# _text_similarity tests
# --------------------------------------------------------------------------- #

check(
    "_text_similarity: identical texts have similarity 1.0",
    _text_similarity("Hello world", "Hello world") == 1.0,
    "identical texts should have similarity 1.0"
)

check(
    "_text_similarity: completely different texts have low similarity",
    _text_similarity("Hello world", "Goodbye universe") < 0.3,
    "completely different texts should have low similarity"
)

check(
    "_text_similarity: very similar texts have high similarity",
    _text_similarity(
        "Attention is the most lucrative skill right now",
        "Attention capture is the most lucrative skill right now"
    ) > 0.7,
    "very similar texts should have high similarity"
)

# --------------------------------------------------------------------------- #
# _remove_near_duplicates tests
# --------------------------------------------------------------------------- #

dup_turns = [
    Turn(speaker="Marcus", text="Attention is the most lucrative skill.", segment=0),
    Turn(speaker="Sofia", text="So great work alone isn't enough?", segment=0),
    Turn(speaker="Marcus", text="Attention capture is the most lucrative skill now.", segment=1),
    Turn(speaker="Sofia", text="So great work alone isn't enough?", segment=1),
]

deduped = _remove_near_duplicates(dup_turns)

check(
    "_remove_near_duplicates: removes highly similar Marcus turn",
    len([t for t in deduped if t.speaker == "Marcus"]) == 1,
    "should keep only one Marcus turn when both are similar"
)

check(
    "_remove_near_duplicates: removes duplicate Sofia question",
    len([t for t in deduped if t.speaker == "Sofia"]) == 1,
    "should keep only one Sofia turn when identical"
)

check(
    "_remove_near_duplicates: keeps first occurrence",
    deduped[0].text == "Attention is the most lucrative skill.",
    "should keep the first occurrence of duplicate content"
)

# Test that different content is preserved
diff_turns = [
    Turn(speaker="Marcus", text="Attention is key.", segment=0),
    Turn(speaker="Marcus", text="Personal brands matter.", segment=1),
    Turn(speaker="Marcus", text="Two paths exist.", segment=2),
]

diff_deduped = _remove_near_duplicates(diff_turns)
check(
    "_remove_near_duplicates: preserves different content",
    len(diff_deduped) == 3,
    "should preserve all turns when content is different"
)

# --------------------------------------------------------------------------- #
# parse_turns integration tests for new filters
# --------------------------------------------------------------------------- #

# Test planning crumb removal
crumb_raw = """Marcus: That's the positioning exercise. Count add. Two paths exist.
Sofia: Got it!
"""

crumb_turns = parse_turns(crumb_raw, hosts)
check(
    "parse_turns: removes 'Count add.' crumb",
    not any("Count add" in t.text for t in crumb_turns),
    "'Count add.' should be removed from dialogue"
)

check(
    "parse_turns: preserves surrounding content after crumb removal",
    any("positioning exercise" in t.text for t in crumb_turns),
    "content around the crumb should be preserved"
)

# Test truncated line filtering
truncated_raw = """Marcus: This is a full sentence.
Sofia: 17 So
Sofia: That makes sense.
"""

truncated_turns = parse_turns(truncated_raw, hosts)
check(
    "parse_turns: filters truncated '17 So' line",
    not any("17 So" in t.text for t in truncated_turns),
    "truncated '17 So' should be filtered out"
)

check(
    "parse_turns: preserves valid lines around truncated",
    len(truncated_turns) == 2,
    "should have 2 valid turns (Marcus + Sofia) after filtering truncated line"
)

# Test near-duplicate removal in parse_turns
dup_raw = """Marcus: Why is everyone suddenly building a personal brand?
Sofia: Because job security is falling. People face three choices.
Marcus: Why is everyone suddenly building a personal brand?
Sofia: Because job security is falling. People face three choices.
"""

dup_parsed = parse_turns(dup_raw, hosts)
check(
    "parse_turns: deduplicates identical exchanges",
    len(dup_parsed) == 2,
    f"should remove duplicate exchanges, keeping 2 turns, got {len(dup_parsed)}"
)

# --------------------------------------------------------------------------- #
# Regression test: real sample from bug report
# --------------------------------------------------------------------------- #

real_sample = """Marcus: The most lucrative skill has always been capturing attention. If nobody knows who you are or what you offer, how can they pay you?
Sofia: So great work alone isn't enough?
Marcus: No. You can build excellent software, spend years writing a book, or work hard to improve your career, but without awareness and customers, none of that translates into money. Attention and customer acquisition determine success. This applies to your business and theirs.
Sofia: And other people will pay for that?
Marcus: Exactly. Other businesses will pay top dollar for you to capture attention for them.
Marcus: Attention capture is the most lucrative skill right now. You can have brilliant work, but if nobody sees it, nobody pays for it.
Sofia: So great work alone isn't enough?
Marcus: Not even close. Quality software, a good book, solid career effort—none of it turns into money without awareness and customers. Attention and customer acquisition decide who wins, and everyone else stays invisible.
Sofia: Why is everyone suddenly building a personal brand?
    10|Marcus: Because job security is falling. People face three choices: complain, upskill, or start a business.
Sofia: And many pick business?
Marcus: Yes, and a personal brand is often the cheapest way in. Attention is finite and scarce, so owning an audience matters.
Sofia: Is that why creator businesses are growing?
Marcus: Exactly. Creator-economy revenue keeps climbing, and creator ad revenue has passed traditional media in some measures. A brand lets people turn trust into income.
Sofia: Why is everyone suddenly building a personal brand?
Marcus: Because job security is falling. People face three choices: complain, upskill, or start a business.
Sofia: And many pick business?
Marcus: Yes, and a personal brand is often the cheapest way in. Attention is finite and scarce, so owning an audience matters.
Sofia: Is that why creator businesses are growing?
    20|Marcus: Exactly. Creator-economy revenue keeps climbing, and creator ad revenue has passed traditional media in some measures. A brand lets people turn trust into income.
Sofia: So personal brands are becoming the new résumé?
Marcus: That's why they feel urgent: they convert scarce attention into opportunity and leverage.
Marcus: Two paths: build your own creative team and personal brand, or become someone else's creative director.
Sofia: So how do I land the first client?
Marcus: Do the permissionless apprentice move. JK Molina once DM'd a creator: "I wrote ten posts for you anyway. Post them; if you like them, let me write more—or give me a testimonial.
Sofia: That's bold.
Marcus: Then position yourself: what are you against, why does it matter now, and where do you show up? Public applications work too. That's the positioning exercise: what are you against, why does it matter now, and where are you most visible? Public applications work too." Count add. Two paths: build your own creative team and personal brand, or become someone else's creative director.
Marcus: Do the permissionless apprentice move. JK Molina once DM'd a creator: "I wrote ten posts for you anyway. Post them; if you like them, let me write more—or give me a testimonial.
Marcus: Then run the positioning exercise: what are you against, why does it matter now, and where are you most visible? Public applications work too.
    30|Sofia: 17 So"""

real_turns = parse_turns(real_sample, hosts)
real_texts = [t.text for t in real_turns]

check(
    "real_sample: removes 'Count add.' crumb",
    not any("Count add" in t for t in real_texts),
    "'Count add.' should be removed from sample"
)

check(
    "real_sample: filters truncated '17 So'",
    not any("17 So" in t for t in real_texts),
    "truncated '17 So' should be filtered"
)

check(
    "real_sample: line number prefixes stripped",
    not any("10|" in t or "20|" in t or "30|" in t for t in real_texts),
    "line number prefixes should be stripped"
)

check(
    "real_sample: near-duplicate attention skill removed",
    sum(1 for t in real_texts if "lucrative skill" in t.lower()) <= 2,
    "should not have many duplicate 'lucrative skill' mentions"
)

check(
    "real_sample: near-duplicate personal brand Q&A reduced",
    sum(1 for t in real_texts if "why is everyone suddenly building" in t.lower()) <= 1,
    "duplicate 'why is everyone building a personal brand' should be reduced"
)

check(
    "real_sample: reasonable number of turns",
    5 <= len(real_turns) <= 20,
    f"expected 5-20 clean turns after deduplication, got {len(real_turns)}"
)

check(
    "real_sample: preserves core dialogue content",
    any("attention" in t.lower() for t in real_texts),
    "core dialogue about attention should be preserved"
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
print("ALL REPETITION FIX TESTS PASSED")
