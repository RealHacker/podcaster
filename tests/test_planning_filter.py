"""Tests for planning text filtering in script parsing.

These tests ensure that chain-of-thought planning notes, word-count annotations,
numbered-word drafts, and meta-instructions are filtered out of parsed dialogue.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from podcaster.script import (
    Host,
    parse_turns,
    _is_planning_line,
    _clean_dialogue_text,
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
# _is_planning_line tests
# --------------------------------------------------------------------------- #

check(
    "_is_planning_line: normal dialogue",
    not _is_planning_line("I spent eight months building a course with real depth."),
    "clean dialogue should not be flagged as planning"
)

check(
    "_is_planning_line: word count target",
    _is_planning_line("That's 32? Need total 124."),
    "word count target should be flagged"
)

check(
    "_is_planning_line: let's count",
    _is_planning_line("Need target about 124. Let's count carefully."),
    "'Let's count' should be flagged"
)

check(
    "_is_planning_line: arithmetic expression",
    _is_planning_line("Total 6+18+10+55+25=114. About target…"),
    "arithmetic counting should be flagged"
)

check(
    "_is_planning_line: numbered word draft",
    _is_planning_line("What's(1) the2 practical3 Monday-morning4 starting5 point6"),
    "numbered word draft should be flagged"
)

check(
    "_is_planning_line: consecutive numbered words",
    _is_planning_line("Perfect1 Join2 the3 Second4 Renaissance5 workshop6"),
    "consecutive numbered words should be flagged"
)

check(
    "_is_planning_line: meta planning - content pillars",
    _is_planning_line("content pillars, unique perspective, mission and beliefs"),
    "meta planning about content pillars should be flagged"
)

check(
    "_is_planning_line: meta planning - walks through",
    _is_planning_line("Sofia walks through content pillars then gives prompt example"),
    "meta planning instructions should be flagged"
)

check(
    "_is_planning_line: meta planning - need add more",
    _is_planning_line("Then Sofia answer with specifics. Need maybe add more."),
    "planning note 'need add more' should be flagged"
)

check(
    "_is_planning_line: meta planning - before closes",
    _is_planning_line("before Marcus closes promo and takeaway"),
    "meta planning 'before closes' should be flagged"
)

# --------------------------------------------------------------------------- #
# _clean_dialogue_text tests
# --------------------------------------------------------------------------- #

check(
    "_clean_dialogue_text: normal text unchanged",
    _clean_dialogue_text("I spent eight months building a course.") == "I spent eight months building a course.",
    "clean text should pass through unchanged"
)

check(
    "_clean_dialogue_text: strips word count suffix",
    _clean_dialogue_text("The speed is real, but attention is the point. = 30ish.") == "The speed is real, but attention is the point.",
    "word count suffix should be stripped"
)

check(
    "_clean_dialogue_text: strips word count exact",
    _clean_dialogue_text("No. It's a one-person creative team. = 22") == "No. It's a one-person creative team.",
    "exact word count should be stripped"
)

check(
    "_clean_dialogue_text: strips word count equals",
    _clean_dialogue_text("Judgment. They're not buying drafts. =20") == "Judgment. They're not buying drafts.",
    "=20 suffix should be stripped"
)

check(
    "_clean_dialogue_text: strips line number prefix",
    _clean_dialogue_text("    10|The speed is real.") == "The speed is real.",
    "line number prefix should be stripped"
)

check(
    "_clean_dialogue_text: strips line number no padding",
    _clean_dialogue_text("20|Marcus: Hello") == "Marcus: Hello",
    "line number without padding should be stripped"
)

# --------------------------------------------------------------------------- #
# parse_turns integration tests - planning lines filtered
# --------------------------------------------------------------------------- #

planning_mixed_raw = """Sofia: concrete example. "I spent eight months building a course with real depth. It was solid. But I launched to a nearly silent list and made maybe three sales. The work didn't fail; the audience did." That's 32? Need total 124.
Marcus: So quality alone didn't guarantee payment, because no one knew it existed?" Then Sofia answer with specifics. Need maybe add more. Need target about 124. Let's count carefully.
Sofia: I spent eight months building a course with real depth. It was solid. But I launched to a tiny email list and made three sales.
Marcus: So quality alone didn't guarantee payment, because nobody knew it existed?
Sofia: Exactly. I had proof the material worked for the few buyers, but no distribution.
"""

turns = parse_turns(planning_mixed_raw, hosts)
texts = [t.text for t in turns]

check(
    "parse_turns: planning lines filtered out",
    len(turns) == 3,
    f"expected 3 turns (clean dialogue), got {len(turns)}: {texts}"
)

check(
    "parse_turns: first clean line preserved",
    any("I spent eight months" in t.text for t in turns),
    "clean dialogue about course should be preserved"
)

check(
    "parse_turns: planning note not in output",
    not any("Need total 124" in t.text for t in turns),
    "planning note 'Need total 124' should not appear in output"
)

check(
    "parse_turns: let's count not in output",
    not any("Let's count" in t.text for t in turns),
    "'Let's count' should not appear in output"
)

# --------------------------------------------------------------------------- #
# parse_turns integration tests - word count annotations stripped
# --------------------------------------------------------------------------- #

word_count_raw = """Sofia: The speed is real, but attention is the point. AI researches what's resonating, maps angles, and drafts options. I review, rewrite, and decide what actually sounds like me. = 30ish.
Sofia: No. It's a one-person creative team. I'm the editor and strategist; AI is the junior researcher and first-draft machine. = 22
Sofia: Judgment. They're not buying drafts. They're buying my taste, positioning, and the final voice that makes the work land. =20
"""

turns2 = parse_turns(word_count_raw, hosts)
texts2 = [t.text for t in turns2]

check(
    "parse_turns: word count stripped from merged turns",
    len(turns2) == 1,
    f"expected 1 merged turn for Sofia, got {len(turns2)}"
)

check(
    "parse_turns: no '= 30ish' in output",
    not any("= 30" in t for t in texts2),
    "word count '= 30ish' should be stripped"
)

check(
    "parse_turns: no '= 22' in output",
    not any("= 22" in t for t in texts2),
    "word count '= 22' should be stripped"
)

check(
    "parse_turns: dialogue content preserved",
    "attention is the point" in texts2[0] if turns2 else False,
    "core dialogue content should be preserved"
)

# --------------------------------------------------------------------------- #
# parse_turns integration tests - numbered word drafts filtered
# --------------------------------------------------------------------------- #

numbered_raw = """Marcus: What's the practical Monday-morning starting point?
Marcus: Then how do I turn that into a research-backed map?
Marcus: Perfect. Join the Second Renaissance workshop for the full process.
Marcus: What's(1) the2 practical3 Monday-morning4 starting5 point6
    20|Marcus: Then1 how2 do3 I4 turn5 that6 into7 a8 research-backed9 map10
Marcus: Perfect1 Join2 the3 Second4 Renaissance5 workshop6 for7 the8 full9 process10 and11 remember12 the13 advantage14 goes15 to16 whoever17 turns18 their19 perspective20 into21 a22 repeatable23 system24 first25.
"""

turns3 = parse_turns(numbered_raw, hosts)
texts3 = [t.text for t in turns3]

check(
    "parse_turns: clean Marcus lines preserved",
    len(turns3) == 1,
    f"expected 1 merged turn with clean dialogue, got {len(turns3)}: {texts3}"
)

check(
    "parse_turns: numbered draft 'What's(1)' filtered",
    not any("What's(1)" in t for t in texts3),
    "numbered draft pattern should be filtered"
)

check(
    "parse_turns: numbered draft 'Then1 how2' filtered",
    not any("Then1 how2" in t for t in texts3),
    "numbered word sequence should be filtered"
)

check(
    "parse_turns: clean 'Monday-morning' preserved",
    "Monday-morning" in texts3[0] if turns3 else False,
    "clean dialogue should be preserved"
)

# --------------------------------------------------------------------------- #
# parse_turns integration tests - meta checklist text filtered
# --------------------------------------------------------------------------- #

meta_raw = """Sofia: Because attention is the finite resource.
Marcus: Perfect. Join the Second Renaissance workshop for the full process, and remember: the advantage goes to whoever turns their perspective into a repeatable system first. Total 6+18+10+55+25=114. About target, could be okay. But need Sofia "walks through content pillars, unique perspective, mission and beliefs, and what you're against, then gives prompt example before Marcus closes". We have that. But "Monday-morning starting point" maybe Sofia should answer "Start by..." good. Need "content pillars" yes. "mission and beliefs" yes. "what you're against" yes.
Sofia: I've seen brilliant work die quietly; without an audience, quality alone never gets paid.
"""

turns4 = parse_turns(meta_raw, hosts)
texts4 = [t.text for t in turns4]

check(
    "parse_turns: meta checklist line filtered",
    len(turns4) == 1,
    f"expected 1 merged Sofia turn (Marcus meta-line filtered), got {len(turns4)}: {texts4}"
)

check(
    "parse_turns: arithmetic checklist not in output",
    not any("Total 6+18" in t for t in texts4),
    "arithmetic checklist should be filtered"
)

check(
    "parse_turns: content pillars not in output",
    not any("content pillars" in t.lower() for t in texts4),
    "'content pillars' meta-text should be filtered"
)

# --------------------------------------------------------------------------- #
# parse_turns integration tests - line number prefixes stripped
# --------------------------------------------------------------------------- #

line_numbered_raw = """    10|Sofia: The speed is real, but attention is the point.
Sofia: No. It's a one-person creative team.
    20|Marcus: Then how do I turn that into a research-backed map?
"""

turns5 = parse_turns(line_numbered_raw, hosts)

check(
    "parse_turns: line numbers stripped, turns parsed",
    len(turns5) == 2,
    f"expected 2 turns (Sofia lines merged, Marcus separate), got {len(turns5)}"
)

check(
    "parse_turns: line number not in text",
    not any("10|" in t.text for t in turns5),
    "line number prefix should be stripped"
)

# --------------------------------------------------------------------------- #
# parse_turns integration tests - clean dialogue unchanged
# --------------------------------------------------------------------------- #

clean_raw = """Marcus: Hello there.
Sofia: Hi Marcus, how are you?
Marcus: I'm good — really good.
Sofia: Great! The weather is nice today.
Marcus: Indeed. Let's talk about the topic.
"""

turns6 = parse_turns(clean_raw, hosts)

check(
    "parse_turns: clean dialogue all preserved",
    len(turns6) == 5,
    f"expected 5 clean turns, got {len(turns6)}"
)

check(
    "parse_turns: speaker alternation correct",
    [t.speaker for t in turns6] == ["Marcus", "Sofia", "Marcus", "Sofia", "Marcus"],
    f"speaker sequence wrong: {[t.speaker for t in turns6]}"
)

# --------------------------------------------------------------------------- #
# Regression: real sample from bug report
# --------------------------------------------------------------------------- #

real_sample = """Sofia: concrete example. "I spent eight months building a course with real depth. It was solid. But I launched to a nearly silent list and made maybe three sales. The work didn't fail; the audience did." That's 32? Need total 124.
Marcus: So quality alone didn't guarantee payment, because no one knew it existed?" Then Sofia answer with specifics. Need maybe add more. Need target about 124. Let's count carefully. The core claim here is that attention is the most lucrative skill. Does that hold up against what you've actually seen?
Sofia: I spent eight months building a course with real depth. It was solid. But I launched to a tiny email list and made three sales. The work didn't fail; the audience did.
Marcus: So quality alone didn't guarantee payment, because nobody knew it existed?
Sofia: Exactly. I had proof the material worked for the few buyers, but no distribution. Once I started posting the same ideas publicly, people began asking to buy before I even pitched.
Marcus: And that's the hard part—attention isn't earned by quality alone, is it?
Sofia: No. It's earned by being visible enough for the right people to recognize the quality. Otherwise, you're just a well-kept secret.
Sofia: Because attention is the finite resource. You can't buy trust at scale, and you can't wait for permission anymore. People are tired of complaining about layoffs and algorithms. Building even a small audience feels like owning a lane instead of renting one.
Sofia: Start with skills you already use. Publish one useful thing daily, study what works, and sell before you quit. The realistic path is compounding, not gambling.
    10|Sofia: The speed is real, but attention is the point. AI researches what's resonating, maps angles, and drafts options. I review, rewrite, and decide what actually sounds like me.
Sofia: No. It's a one-person creative team. I'm the editor and strategist; AI is the junior researcher and first-draft machine.
Sofia: Judgment. They're not buying drafts. They're buying my taste, positioning, and the final voice that makes the work land.
Sofia: The speed is real, but attention is the point. AI researches what's resonating, maps angles, and drafts options. I review, rewrite, and decide what actually sounds like me. = 30ish.
Sofia: No. It's a one-person creative team. I'm the editor and strategist; AI is the junior researcher and first-draft machine. = 22
Sofia: Judgment. They're not buying drafts. They're buying my taste, positioning, and the final voice that makes the work land. =20
Sofia: Attention is the point. AI researches what's resonating, maps angles, and drafts options. I review, rewrite, and decide what actually sounds like me.
Marcus: What's the practical Monday-morning starting point?
Marcus: Then how do I turn that into a research-backed map?
Marcus: Perfect. Join the Second Renaissance workshop for the full process, and remember: the advantage goes to whoever turns their perspective into a repeatable system first. What's(1) the2 practical3 Monday-morning4 starting5 point6
    20|Marcus: Then1 how2 do3 I4 turn5 that6 into7 a8 research-backed9 map10
Marcus: Perfect1 Join2 the3 Second4 Renaissance5 workshop6 for7 the8 full9 process10 and11 remember12 the13 advantage14 goes15 to16 whoever17 turns18 their19 perspective20 into21 a22 repeatable23 system24 first25. Total 6+18+10+55+25=114. About target, could be okay. But need Sofia "walks through content pillars, unique perspective, mission and beliefs, and what you're against, then gives prompt example before Marcus closes". We have that. But "Monday-morning starting point" maybe Sofia should answer "Start by..." good. Need "content pillars" yes. "mission and beliefs" yes. "what you're against" yes. Prompt has "brand and content docs" yes. "research-backed content map prompt example" yes. Marcus closes promo and takeaway. Good. Word count 114, target about 124. Could add "on a single page" maybe. Need no blank lines. Need exactly lines no blank. Could include "Then" etc. Should be okay.
Sofia: I've seen brilliant work die quietly; without an audience, quality alone never gets paid.
Sofia: Because payment follows attention, not effort—great work needs distribution.
Marcus: Start Monday with your content pillars and one sharp belief; that's how attention becomes income. Attention(1) is2 the3 most4 lucrative5 skill—test6 that7 against8 your9 own10
"""

real_turns = parse_turns(real_sample, hosts)
real_texts = [t.text for t in real_turns]

check(
    "real_sample: reasonable number of turns",
    5 <= len(real_turns) <= 15,
    f"expected 5-15 clean turns from sample, got {len(real_turns)}"
)

check(
    "real_sample: no 'Need total' in output",
    not any("Need total" in t for t in real_texts),
    "'Need total' planning text should be filtered"
)

check(
    "real_sample: no word count annotations",
    not any("= 30" in t or "= 22" in t or "=20" in t for t in real_texts),
    "word count annotations should be stripped"
)

check(
    "real_sample: no numbered drafts",
    not any("What's(1)" in t or "Then1 how2" in t for t in real_texts),
    "numbered word drafts should be filtered"
)

check(
    "real_sample: no meta planning instructions",
    not any("walks through" in t.lower() or "before Marcus closes" in t for t in real_texts),
    "meta planning instructions should be filtered"
)

check(
    "real_sample: no arithmetic counting",
    not any("6+18+10" in t for t in real_texts),
    "arithmetic counting should be filtered"
)

check(
    "real_sample: clean dialogue preserved",
    any("eight months building a course" in t for t in real_texts),
    "clean dialogue content should be preserved"
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
print("ALL PLANNING FILTER TESTS PASSED")
