"""Check that the script writer respects the requested length."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from podcaster.script import Host, Turn, _trim_to_target
from podcaster.chunking import estimate_speech_units, units_to_minutes, SPEECH_UNITS_PER_MINUTE

fails = []

# --- trim brings an over-long script inside the tolerance -------------------
lang = "en"
hosts = [Host(name="A", gender="male"), Host(name="B", gender="female")]
turns = [Turn(speaker="A" if i % 2 == 0 else "B", text=" ".join(["word"] * 60), segment=0)
         for i in range(20)]
target = 157 * 3  # 3 minutes of English
trimmed = _trim_to_target(turns, target, lang)
units = estimate_speech_units(" ".join(t.text for t in trimmed), lang)
print(f"before={estimate_speech_units(' '.join(t.text for t in turns), lang)} "
      f"after={units} target={target}")
if units > target * 1.12:
    fails.append(f"trim did not reach the target: {units} > {target * 1.12}")
if trimmed[0].speaker != "A" or len(trimmed) < 3:
    fails.append("trim removed the hook")

# --- the sign-off survives -------------------------------------------------
if len(trimmed) < 2 or trimmed[-1].text != turns[-1].text:
    fails.append("trim removed the sign-off")

# --- calibration against the measured rates --------------------------------
if abs(SPEECH_UNITS_PER_MINUTE["en"] - 157.0) > 1:
    fails.append("English rate drifted")
if abs(SPEECH_UNITS_PER_MINUTE["zh"] - 252.0) > 1:
    fails.append("Chinese rate drifted")
if abs(units_to_minutes(517, "en") - 3.29) > 0.05:
    fails.append(f"English estimate wrong: {units_to_minutes(517, 'en')}")
if abs(units_to_minutes(1028, "zh") - 4.08) > 0.05:
    fails.append(f"Chinese estimate wrong: {units_to_minutes(1028, 'zh')}")

# --- a short script is left alone -----------------------------------------
short = [Turn(speaker="A", text="hello there", segment=0), Turn(speaker="B", text="hi", segment=0)]
if len(_trim_to_target(short, target, lang)) != 2:
    fails.append("a short script was modified")

print()
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL LENGTH CHECKS PASSED")
