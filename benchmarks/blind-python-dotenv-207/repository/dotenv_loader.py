"""Minimal third-party-issue reproduction; not copied upstream source code."""

from __future__ import annotations

import locale
from pathlib import Path

DEFAULT_ENCODING: str | None = None


def read_env(path: str | Path, *, encoding: str | None = None) -> str:
    """Read one dotenv-style text file."""

    selected = encoding if encoding is not None else DEFAULT_ENCODING
    if selected is None:
        selected = locale.getpreferredencoding(False)
    return Path(path).read_text(encoding=selected)
