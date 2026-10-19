"""Assert the Python metrics in reports/metrics.json equal the official Java evaluator's.

Usage (from repo root, needs `java`):
    python scripts/check_java_evaluator.py              # HOG reports in reports/
    python scripts/check_java_evaluator.py reports/cnn  # CNN reports
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = {
    "single": ("Evaluator", "trueLocations.txt", "foundLocations.txt"),
    "scale": ("Evaluator_Scale", "trueLocations_Scale.txt", "foundLocations_Scale.txt"),
}


def main() -> int:
    report_dir = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "reports")
    metrics = json.loads((report_dir / "metrics.json").read_text())
    ok = True
    for key, (cls, truth, found) in CASES.items():
        out = subprocess.run(
            ["java", "-cp", str(ROOT / "CarData"), cls, str(ROOT / "CarData" / truth), found],
            cwd=report_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tp = int(re.search(r"Correct detections\s*:\s*(\d+)", out).group(1))
        fp = int(re.search(r"False detections\s*:\s*(\d+)", out).group(1))
        py = metrics[key]
        match = (tp, fp) == (py["tp"], py["fp"])
        ok &= match
        print(
            f"{key:6s} java tp={tp} fp={fp} | python tp={py['tp']} fp={py['fp']} "
            f"-> {'OK' if match else 'MISMATCH'}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
