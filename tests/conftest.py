"""Make `src/` importable and expose config.json to the tests (no raw data is read)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    with (ROOT / "config.json").open(encoding="utf-8") as handle:
        return json.load(handle)
