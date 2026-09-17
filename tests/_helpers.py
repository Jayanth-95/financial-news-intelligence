"""Shared helpers for test files in this directory.

Not a test file itself (no TESTS list, nothing runs if executed directly).
"""
from __future__ import annotations

from pathlib import Path

# The exact bug this project shipped and then fixed (see tests/test_phase9.py
# and pipeline.py's PROJECT_ROOT/DEFAULT_TICKER_MAP): a bare relative string
# "data/ticker_mapping.csv" silently resolves against whatever directory a
# command happens to be invoked from. Test files calling load_ticker_mapping
# directly must not repeat that mistake -- use this instead of a literal
# string so the test suite passes regardless of the invoking working
# directory, not only when run from the project root.
TICKER_MAPPING_PATH = str(Path(__file__).resolve().parent.parent / "data" / "ticker_mapping.csv")
