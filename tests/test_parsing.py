"""Regression checks for the script parsing / voice assignment path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from podcaster.script import Host, parse_turns
from podcaster.pipeline import build_hosts, PodcastRequest
from podcaster import voices as V

fails = []

# --- parse_turns must resolve to the host NAME (a str), not the Host object ---
hosts = [Host(name="Marcus", gender="male"), Host(name="Sofia", gender="female")]
raw = """Marcus: Hello there.
Sofia: Hi Marcus, how are you?
Marcus: I'm good — really good.
**Sofia**: Great!
Unknown Person: should be dropped
"""
turns = parse_turns(raw, hosts)
print("parsed speakers:", [(t.speaker, type(t.speaker).__name__) for t in turns])
if not turns:
    fails.append("no turns parsed")
for t in turns:
    if not isinstance(t.speaker, str):
        fails.append(f"speaker is {type(t.speaker).__name__}, expected str")
if [t.speaker for t in turns] != ["Marcus", "Sofia", "Marcus", "Sofia"]:
    fails.append(f"unexpected speaker sequence: {[t.speaker for t in turns]}")

# --- Chinese names ---
zh_hosts = [Host(name="李维", gender="male"), Host(name="苏菲", gender="female")]
zh = "李维：大家好。\n苏菲：你好，李维！\n"
zh_turns = parse_turns(zh, zh_hosts)
print("zh speakers:", [t.speaker for t in zh_turns])
if [t.speaker for t in zh_turns] != ["李维", "苏菲"]:
    fails.append(f"zh speaker resolution wrong: {[t.speaker for t in zh_turns]}")

# --- voice assignment must give different hosts different voices ---
req = PodcastRequest(host_count=2, host_genders=["male", "female"],
                     host_names=["Marcus", "Sofia"], language="en")
built = build_hosts(req)
print("voices:", [(h.name, h.gender, h.voice_id) for h in built])
if built[0].voice_id == built[1].voice_id:
    fails.append("both hosts got the same voice")
if built[0].voice_id != "English_Trustworthy_Man":
    fails.append(f"male host voice unexpected: {built[0].voice_id}")
if built[1].voice_id != "English_Upbeat_Woman":
    fails.append(f"female host voice unexpected: {built[1].voice_id}")

# --- three hosts, mixed genders, must still be distinct ---
req3 = PodcastRequest(host_count=3, host_genders=["male", "female", "male"], language="zh")
built3 = build_hosts(req3)
print("3-host voices:", [(h.name, h.gender, h.voice_id) for h in built3])
if len({h.voice_id for h in built3}) != 3:
    fails.append("3 hosts did not get 3 distinct voices")

print()
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
