"""Чтение базы знаний из data/knowledge-base.zip для эндпоинтов /documents.

Без FastAPI: модуль можно использовать и из ядра.
Путь к папке с данными берётся из переменной окружения DATA_DIR (по умолчанию data).
"""

from __future__ import annotations

import os
import re
import zipfile
from functools import lru_cache
from pathlib import Path

_TRANSLIT = dict(zip(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "j", "k", "l", "m", "n", "o",
     "p", "r", "s", "t", "u", "f", "h", "c", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"],
))


def _slug(text: str) -> str:
    text = "".join(_TRANSLIT.get(ch, ch) for ch in text.lower())
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _doc_type(file_stem: str) -> str:
    s = file_stem.lower()
    if s.startswith("законы"):
        return "law"
    title = s.split("__", 1)[-1]
    # порядок важен: «договоров» встречается в названиях регламента и записки
    if title.startswith("доп соглашение"):
        return "addendum"
    if title.startswith("регламент"):
        return "regulation"
    if title.startswith("служебная записка"):
        return "memo"
    if title.startswith("письмо"):
        return "letter"
    if title.startswith("договор"):
        return "contract"
    return "other"


def zip_path() -> Path:
    return Path(os.getenv("DATA_DIR", "data")) / "knowledge-base.zip"


@lru_cache(maxsize=1)
def load_documents() -> list[dict]:
    """Список документов: doc_id, doc_title, doc_type, is_active, text. Пусто, если архива нет."""
    path = zip_path()
    if not path.exists():
        return []
    docs = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".txt"):
                continue
            raw = zf.read(name)
            text = raw.decode("utf-8-sig", errors="replace")
            stem = Path(name).stem
            # "документы компании__Договор ..." -> "Договор ..."
            title = stem.split("__", 1)[-1].replace("_действующая", "").replace("_", " ")
            docs.append({
                "doc_id": _slug(title),
                "doc_title": title,
                "doc_type": _doc_type(stem),
                "is_active": "утратил силу" not in stem.lower(),
                "text": text,
            })
    return docs


def get_document(doc_id: str) -> dict | None:
    return next((d for d in load_documents() if d["doc_id"] == doc_id), None)


def find_document(*keywords: str) -> dict | None:
    """Первый документ, в id или названии которого есть все ключевые слова (без учёта регистра)."""
    kws = [k.lower() for k in keywords]
    return next(
        (d for d in load_documents()
         if all(k in f'{d["doc_id"]} {d["doc_title"]}'.lower() for k in kws)),
        None,
    )