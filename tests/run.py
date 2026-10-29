"""Run every check in this folder.

    python tests/run.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    failures = []
    for test in sorted(HERE.glob("test_*.py")):
        print(f"\n=== {test.name} " + "=" * (46 - len(test.name)))
        result = subprocess.run([sys.executable, str(test)])
        if result.returncode != 0:
            failures.append(test.name)
    print("\n" + "=" * 56)
    if failures:
        print("FAILED:", ", ".join(failures))
        sys.exit(1)
    print("ALL TESTS PASSED")
