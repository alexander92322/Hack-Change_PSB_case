"""Самопроверка ответа ассистента кодом (без лишних вызовов LLM).

Проверяет, что все упоминания статей, пунктов и номеров договоров в тексте ответа
действительно присутствуют в переданных найденных фрагментах (sources).
Корректно сопоставляет номера договоров через слэш (А-08/2024) и дефис (А-08-2024).
"""

from __future__ import annotations

import re
from typing import Any

import kb

# Вытаскивает упоминания статей: "ст. 164", "статьи 284.1", "ст 13.11", "ст9"
_ARTICLE_RE = re.compile(r"\bст(?:ать[еиюяь]|атья)?\.?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

# Вытаскивает упоминания пунктов: "п. 3.1", "пункта 7.4", "подп. 2", "пп. 1.2"
_POINT_RE = re.compile(r"\b(?:под)?п(?:ункт[ауеом]?|п)?\.?\s*(\d+(?:\.\d+)*)", re.IGNORECASE)

# Вытаскивает номера договоров и со слэшем (А-08/2024), и с дефисом (А-08-2024)
_DOC_NUM_RE = re.compile(
    r"\b([А-ЯЁA-Z]{1,4})\s*-\s*(\d{1,4})\s*[/\\-]\s*(20\d{2}|\d{2})\b",
    re.IGNORECASE,
)

# Перевод латинских букв-двойников в кириллицу (если модель написала A-08 латинской A)
_LAT_TO_CYR = str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ")


def _normalize_doc_num(prefix: str, num: str, year: str) -> str:
    """Приводит номер договора к единому виду: 'А-08-2024' и 'A-08/2024' -> 'А-08/2024'."""
    clean_prefix = prefix.upper().translate(_LAT_TO_CYR)
    clean_num = num.lstrip("0") or "0"
    return f"{clean_prefix}-{clean_num}/{year}"


def extract_citations(text: str) -> dict[str, set[str]]:
    """Извлекает из текста все юридические ссылки: статьи, пункты и нормализованные номера договоров."""
    articles = {m.group(1) for m in _ARTICLE_RE.finditer(text)}
    points = {m.group(1) for m in _POINT_RE.finditer(text)}
    doc_nums = {
        _normalize_doc_num(m.group(1), m.group(2), m.group(3))
        for m in _DOC_NUM_RE.finditer(text)
    }
    return {
        "articles": articles,
        "points": points,
        "doc_nums": doc_nums,
    }


def verify_answer(
    answer_text: str,
    status: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Проверяет ответ на соответствие найденным фрагментам (sources)."""
    if status in {"refused_no_data", "refused_offtopic", "refused_illegal"}:
        return {
            "is_valid": True,
            "answer": answer_text,
            "status": status,
            "unverified_citations": [],
        }

    citations = extract_citations(answer_text)
    has_any_citation = any(citations.values())

    if not sources:
        if has_any_citation:
            return {
                "is_valid": False,
                "answer": "В предоставленной базе знаний недостаточно подтверждённых данных для точного ответа на этот вопрос.",
                "status": "refused_no_data",
                "unverified_citations": sorted(
                    [f"ст. {a}" for a in citations["articles"]]
                    + [f"п. {p}" for p in citations["points"]]
                    + list(citations["doc_nums"])
                ),
            }
        return {
            "is_valid": True,
            "answer": answer_text,
            "status": status,
            "unverified_citations": [],
        }

    # Собираем текст из всех переданных фрагментов + преамбул этих документов из базы (по doc_id)
    parts: list[str] = []
    seen_docs: set[str] = set()
    for s in sources:
        doc_id = s.get("doc_id", "")
        parts.append(
            f"{doc_id} {s.get('doc_title', '')} {s.get('section', '')} {s.get('point', '')} {s.get('text', '')}"
        )
        if doc_id and doc_id not in seen_docs:
            seen_docs.add(doc_id)
            full_doc = kb.get_document(doc_id)
            if full_doc:
                # Добавляем заголовок и первые 600 символов (преамбулу) документа, где указаны номер и дата
                parts.append(f"{full_doc['doc_title']} {full_doc['text'][:600]}")

    allowed_blob = "\n".join(parts)
    allowed_citations = extract_citations(allowed_blob)
    allowed_numbers = set(re.findall(r"\b\d+(?:\.\d+)*\b", allowed_blob))

    unverified: list[str] = []

    for art in sorted(citations["articles"]):
        if art not in allowed_citations["articles"] and art not in allowed_numbers:
            unverified.append(f"ст. {art}")

    for pt in sorted(citations["points"]):
        if pt not in allowed_citations["points"] and pt not in allowed_numbers:
            unverified.append(f"п. {pt}")

    for dnum in sorted(citations["doc_nums"]):
        if dnum not in allowed_citations["doc_nums"]:
            unverified.append(f"документ {dnum}")

    if not unverified:
        return {
            "is_valid": True,
            "answer": answer_text,
            "status": status,
            "unverified_citations": [],
        }

    new_status = "partial" if status == "answered" else status
    warning = (
        f" (Примечание: ссылки на {', '.join(unverified)} не подтверждены "
        f"найденными фрагментами базы знаний и требуют дополнительной проверки)."
    )
    fixed_answer = answer_text.rstrip() + warning

    return {
        "is_valid": False,
        "answer": fixed_answer,
        "status": new_status,
        "unverified_citations": unverified,
    }