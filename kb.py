"""Чтение и структурный парсинг базы знаний (для ядра RAG и эндпоинтов /documents, /chunks)."""

from __future__ import annotations

import os
import re
import unicodedata
import zipfile
from functools import lru_cache
from pathlib import Path

_TRANSLIT = dict(zip(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "j", "k", "l", "m", "n", "o",
     "p", "r", "s", "t", "u", "f", "h", "c", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"],
))

# Ловит разделы вида "1. Предмет договора", "РАЗДЕЛ 2...", "Статья 164..."
_SECTION_RE = re.compile(
    r"^(?:РАЗДЕЛ\s+\d+[^\n]*|Статья\s+\d+(?:\.\d+)?[^\n]*|\d+\.\s+[А-ЯЁA-Z][^\n.]{2,80})$",
    re.MULTILINE,
)
# Ловит пункты вида "1.1.", "3.2.1.", числовые пункты статей "1. " и подпункты "1)"
_POINT_RE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)+(?:\.)?|\d+\.(?=\s+[А-ЯЁ«\"])|\d+\))\s+",
    re.MULTILINE,
)

MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS = 200


def _slug(text: str) -> str:
    text = "".join(_TRANSLIT.get(ch, ch) for ch in text.lower())
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _doc_type(file_stem: str) -> str:
    s = file_stem.lower()
    if s.startswith("законы"):
        return "law"
    title = s.split("__", 1)[-1]
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


def zip_path() -> Path | None:
    """Ищет архив базы знаний по всем возможным именам и путям."""
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    root_dir = Path(__file__).resolve().parent
    candidates = ("knowledge-base.zip", "kb.zip", "knowledge_base.zip")

    for base in (data_dir, root_dir / data_dir, Path.cwd() / data_dir, Path.cwd(), root_dir):
        for name in candidates:
            p = base / name
            if p.is_file():
                return p
    return None


def _extract_law_article(doc_title: str, text: str) -> str:
    m = re.search(r"(?:ст(?:атья)?\.?\s*|_)(\d+(?:\.\d+)?)", doc_title, re.IGNORECASE)
    if not m:
        m = re.search(r"Статья\s+(\d+(?:\.\d+)?)", text[:500], re.IGNORECASE)
    return f"ст. {m.group(1)}" if m else ""


def parse_document_chunks(doc: dict) -> list[dict]:
    """Разбивает документ на структурные фрагменты (чанки) с сохранением раздела, пункта и координат."""
    text: str = doc["text"]
    doc_id: str = doc["doc_id"]
    doc_title: str = doc["doc_title"]
    doc_type: str = doc["doc_type"]
    law_article = _extract_law_article(doc_title, text) if doc_type == "law" else ""

    sections = [(m.start(), " ".join(m.group(0).split())) for m in _SECTION_RE.finditer(text)]
    section_starts = {s_pos for s_pos, _ in sections}

    splits: list[tuple[int, str]] = []
    last_main_point = ""

    for m in _POINT_RE.finditer(text):
        # Если это заголовок раздела в договоре (например, "7. Ответственность сторон"), не считаем его пунктом
        if doc_type != "law" and m.start() in section_starts:
            continue
        raw = m.group("num").strip()
        if raw.endswith(")"):
            sub_num = raw[:-1]
            if last_main_point:
                label = f"{last_main_point}, подп. {sub_num}"
            else:
                label = f"{law_article}, подп. {sub_num}".lstrip(", ") if law_article else f"подп. {sub_num}"
        else:
            clean_num = raw.rstrip(".")
            label = f"{law_article}, п. {clean_num}".lstrip(", ") if law_article else f"п. {clean_num}"
            last_main_point = label
        splits.append((m.start(), label))

    if not splits:
        splits = [(0, law_article or "текст документа")]

    if splits[0][0] > 20:
        splits.insert(0, (0, "преамбула"))

    def current_section(pos: int) -> str:
        sec = ""
        for s_pos, s_title in sections:
            if s_pos <= pos:
                sec = s_title
            else:
                break
        return sec

    chunks: list[dict] = []
    chunk_idx = 0

    for i, (start, point_label) in enumerate(splits):
        end = splits[i + 1][0] if i + 1 < len(splits) else len(text)
        block = text[start:end].strip()
        if not block:
            continue

        sec = current_section(start)
        sub_start = start
        while sub_start < end:
            sub_end = min(sub_start + MAX_CHUNK_CHARS, end)
            if sub_end < end:
                nl = text.rfind("\n", sub_start + MAX_CHUNK_CHARS // 2, sub_end)
                dot = text.rfind(". ", sub_start + MAX_CHUNK_CHARS // 2, sub_end)
                cut = max(nl, dot + 1 if dot != -1 else -1)
                if cut > sub_start:
                    sub_end = cut

            piece = text[sub_start:sub_end].strip()
            if piece:
                chunk_idx += 1
                chunk_id = f"{doc_id}__c{chunk_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "doc_type": doc_type,
                    "is_active": doc["is_active"],
                    "section": sec,
                    "point": point_label,
                    "text": piece,
                    "start_char": sub_start,
                    "end_char": sub_end,
                })
            if sub_end >= end:
                break
            sub_start = max(sub_start + 1, sub_end - OVERLAP_CHARS)

    return chunks


def doc_ref_for_file(filename: str) -> tuple[str, str]:
    """(doc_id, doc_title) по имени файла из архива. Единый источник id для /documents и ответов агента."""
    stem = unicodedata.normalize("NFC", Path(filename).stem)
    title = stem.split("__", 1)[-1].replace("_действующая", "").replace("_", " ").strip()
    return _slug(title), title


@lru_cache(maxsize=1)
def load_documents() -> list[dict]:
    path = zip_path()
    if path is None or not path.exists():
        return []
    docs = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/") or name.startswith("__MACOSX/") or not name.lower().endswith(".txt"):
                continue
            raw = zf.read(name)
            text = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
            stem = unicodedata.normalize("NFC", Path(name).stem)
            doc_id, title = doc_ref_for_file(name)
            docs.append({
                "doc_id": doc_id,
                "doc_title": title,
                "doc_type": _doc_type(stem),
                "is_active": "утратил силу" not in stem.lower(),
                "text": text,
            })
    return docs


@lru_cache(maxsize=1)
def load_chunks() -> list[dict]:
    all_chunks: list[dict] = []
    for doc in load_documents():
        all_chunks.extend(parse_document_chunks(doc))
    return all_chunks


def get_document(doc_id: str) -> dict | None:
    return next((d for d in load_documents() if d["doc_id"] == doc_id), None)


def get_chunk(chunk_id: str) -> dict | None:
    return next((c for c in load_chunks() if c["chunk_id"] == chunk_id), None)


def find_chunk_position(doc_id: str, point: str = "", fragment_text: str = "") -> dict | None:
    doc = get_document(doc_id)
    if not doc:
        return None
    full_text = doc["text"]
    if fragment_text:
        idx = full_text.find(fragment_text[:120].strip())
        if idx != -1:
            return {
                "doc_id": doc_id,
                "doc_title": doc["doc_title"],
                "point": point,
                "text": fragment_text,
                "start_char": idx,
                "end_char": min(len(full_text), idx + len(fragment_text)),
            }
    for c in load_chunks():
        if c["doc_id"] == doc_id and (not point or c["point"] == point or point in c["point"]):
            return c
    return None


def find_document(*keywords: str) -> dict | None:
    kws = [k.lower() for k in keywords]
    return next(
        (d for d in load_documents()
         if all(k in f'{d["doc_id"]} {d["doc_title"]}'.lower() for k in kws)),
        None,
    )