"""Confirm that different hosts really are spoken by different voices.

Measures the fundamental frequency of each speaker's clips with an
autocorrelation pitch estimate.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

JOB = Path(sys.argv[1] if len(sys.argv) > 1 else
           r"E:\Voice\Podcaster\data\jobs\1e960aa176c7")

if not (JOB / "clips").exists():
    # No finished job to inspect — nothing to prove here.
    print("SKIP: no job work directory found at", JOB)
    print("       run a podcast job first, or pass a job directory as an argument")
    sys.exit(0)


def pitch(path: Path) -> float:
    data, rate = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    # work on a loud middle window
    if len(data) < rate // 4:
        return 0.0
    start = len(data) // 3
    window = data[start:start + rate // 2]
    window = window * np.hanning(len(window))
    corr = np.correlate(window, window, mode="full")[len(window) - 1:]
    corr /= (corr[0] + 1e-9)
    lo, hi = int(rate / 400), int(rate / 70)      # 70–400 Hz
    peak = lo + int(np.argmax(corr[lo:hi]))
    if corr[peak] < 0.3:
        return 0.0
    return rate / peak


script = (JOB / "script.txt").read_text(encoding="utf-8")
clips_dir = JOB / "clips"
clip_files = sorted(clips_dir.glob("clip_*.mp3"))

# Map clip index -> speaker, following the order turns were synthesised.
speakers = []
for line in script.splitlines():
    if ":" in line and not line.startswith("#") and not line.startswith("Style:"):
        name = line.split(":", 1)[0].strip()
        if name:
            speakers.append(name)

per_speaker = {}
for idx, clip in enumerate(clip_files):
    if idx >= len(speakers):
        break
    spk = speakers[idx]
    p = pitch(clip)
    if p > 0:
        per_speaker.setdefault(spk, []).append(p)

print(f"{'host':12s} {'clips':>6s} {'median f0':>10s} {'min':>7s} {'max':>7s}")
medians = {}
for spk, values in per_speaker.items():
    med = float(np.median(values))
    medians[spk] = med
    print(f"{spk:12s} {len(values):6d} {med:10.1f} {min(values):7.1f} {max(values):7.1f}")

if len(medians) >= 2:
    names = list(medians)
    a, b = medians[names[0]], medians[names[1]]
    ratio = max(a, b) / max(1e-6, min(a, b))
    print(f"\n{names[0]} vs {names[1]}: f0 ratio = {ratio:.2f}")
    if ratio < 1.12:
        print("WARNING: the two hosts sound like the same voice")
        sys.exit(1)
    print("OK: the hosts use clearly different voices")
else:
    print("not enough speakers to compare")
