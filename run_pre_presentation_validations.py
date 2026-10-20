from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "src/models/feature_ablation.py",
    "src/models/watch_time_sensitivity.py",
    "src/models/probability_calibration.py",
    "src/models/counterfactual.py",
]

for script in STEPS:
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)

print("Validation enhancements complete.")
