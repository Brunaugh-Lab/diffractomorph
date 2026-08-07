"""Import configuration for the standalone manuscript study test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = STUDY_ROOT / "analysis"
os.environ.setdefault("DFM_DATA_ROOT", "/tmp/diffractomorph-no-private-corpus")
for path in (str(STUDY_ROOT), str(ANALYSIS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
