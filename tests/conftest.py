"""Test configuration: make ``src/`` importable and provide shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for path in (str(SRC), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path of the project root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config() -> dict:
    """The real project configuration (read-only use)."""
    from hormuz_common import load_config

    return load_config()
