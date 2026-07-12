from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_STATE_ROOT = REPO_ROOT / ".test-runtime"

# App imports create configuration and logging state. Keep that state in an
# ignored test-only tree so a test collection can never mutate the live tree.
os.environ["PERSONALITYRAG_STATE_ROOT"] = str(TEST_STATE_ROOT)
os.environ.setdefault("PERSONALITYRAG_SUPPRESS_BROWSER", "1")
