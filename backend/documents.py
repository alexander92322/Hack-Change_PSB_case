"""Мост к корневому модулю kb.py для обратной совместимости."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb import (  # noqa: E402, F401
    find_chunk_position,
    find_document,
    get_chunk,
    get_document,
    load_chunks,
    load_documents,
    parse_document_chunks,
    zip_path,
)