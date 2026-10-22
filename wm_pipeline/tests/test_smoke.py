"""Step 3 smoke test: the package imports, and importing it keeps the Phase A rule."""

import subprocess
import sys
from pathlib import Path

# wm_pipeline/tests/test_smoke.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_import_does_not_pull_in_carla_or_car_dreamer():
    # Import in a fresh Python process: inside pytest, other test files may already have
    # imported modules, which could hide a violation (or fake one).
    # On the laptop a violation crashes with "No module named 'carla'"; on the GPU machine,
    # where carla is installed, the sys.modules check is what catches it.
    check = (
        "import sys, wm_pipeline; "
        "leaked = [m for m in ('carla', 'car_dreamer') if m in sys.modules]; "
        "assert not leaked, f'importing wm_pipeline also imported {leaked}'"
    )
    result = subprocess.run([sys.executable, "-c", check], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
