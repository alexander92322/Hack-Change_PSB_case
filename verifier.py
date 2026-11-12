"""Самопроверка ответа ассистента кодом (без лишних вызовов LLM).

Проверяет, что все упоминания статей, пунктов и номеров договоров в тексте ответа
действительно присутствуют в переданных найденных фрагментах (sources).
Защищает от галлюцинаций, за которые обнуляется балл на проверке.
"""

from __future__ import annotations

import re
from typing import Any

# Вытаскивает упоминания статей: "ст. 164", "статьи 284.1", "ст 13.11"
_ARTICLE_RE = re.compile(r"\bст(?:ать[еиюяь]|атья)?\.?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

# Вытаскивает упоминания пунктов: "п. 3.1", "пункта 7.4", "подп. 2", "пп. 1.2"
_POINT_RE = re.compile(r"\b(?:под)?п(?:ункт[ауеом]?|п)?\.?\s*(\d+(?:\.\d+)*)", re.IGNORECASE)

# Вытаскивает номера договоров/документов: "№ П-17/2024", "ПТ-05/2023", "А-08/2024"
_DOC_NUM_RE = re.compile(r"\b([А-ЯЁA-Z]{1,4}-\d{1,4}/\d{2,4})\b")


def _normalize_doc_num(num: str) -> str:
    return num.upper().replace("\\", "/").strip()


def extract_citations(text: str) -> dict[str, set[str]]:
    """Извлекает из текста все юридические ссылки: статьи, пункты и номера договоров."""
    articles = {m.group(1) for m in _ARTICLE_RE.finditer(text)}
    points = {m.group(1) for m in _POINT_RE.finditer(text)}
    doc_nums = {_normalize_doc_num(m.group(1)) for m in _DOC_NUM_RE.finditer(text)}
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
    """Проверяет ответ на соответствие найденным фрагментам (sources).

    Возвращает словарь:
    {
        "is_valid": bool,               # True, если все ссылки подтверждены фрагментами
        "answer": str,                  # проверенный (или скорректированный) текст ответа
        "status": str,                  # проверенный (или скорректированный) статус
        "unverified_citations": list[str],  # список неподтверждённых ссылок (если нашлись)
    }
    """
    # Если бот уже отказался отвечать, проверять ссылки на источники не требуется
    if status in {"refused_no_data", "refused_offtopic", "refused_illegal"}:
        return {
            "is_valid": True,
            "answer": answer_text,
            "status": status,
            "unverified_citations": [],
        }

    citations = extract_citations(answer_text)
    has_any_citation = any(citations.values())

    # Если источников вообще нет, а бот пытается утверждать что-то со ссылками на законы/договоры
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

    # Собираем единый корпус текста из всех найденных фрагментов (заголовки + пункты + текст)
    allowed_blob = "\n".join(
        f"{s.get('doc_title', '')} {s.get('section', '')} {s.get('point', '')} {s.get('text', '')}"
        for s in sources
    )
    allowed_citations = extract_citations(allowed_blob)
    # Также допускаем, что номер пункта/статьи встречается в тексте источника как обычное число
    allowed_numbers = set(re.findall(r"\b\d+(?:\.\d+)*\b", allowed_blob))
    allowed_blob_upper = allowed_blob.upper().replace("\\", "/")

    unverified: list[str] = []

    for art in sorted(citations["articles"]):
        if art not in allowed_citations["articles"] and art not in allowed_numbers:
            unverified.append(f"ст. {art}")

    for pt in sorted(citations["points"]):
        if pt not in allowed_citations["points"] and pt not in allowed_numbers:
            unverified.append(f"п. {pt}")

    for dnum in sorted(citations["doc_nums"]):
        if dnum not in allowed_citations["doc_nums"] and dnum not in allowed_blob_upper:
            unverified.append(f"документ {dnum}")

    if not unverified:
        return {
            "is_valid": True,
            "answer": answer_text,
            "status": status,
            "unverified_citations": [],
        }

    # Если найдены неподтвержденные ссылки — понижаем статус до partial и добавляем честную оговорку
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