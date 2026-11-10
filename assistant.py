"""Ядро ассистента: общий контракт для solution.py и веб-API.

answer(question, history) -> dict:
    {
        "answer":  str,          # текст ответа со ссылками на положения, до ~150 слов
        "status":  str,          # "answered" | "partial" | "refused_no_data"
                                 # | "refused_offtopic" | "refused_illegal"
        "sources": [             # фрагменты, на которые опирается ответ
            {"doc_id": str, "doc_title": str, "point": str, "text": str, "score": float}
        ],
    }

Сейчас это ЗАГЛУШКА с правильным форматом - Backend и Frontend работают на ней,
пока AI/ML собирает поиск и промпты. Сигнатуру и поля не меняем без согласования с CTO.
"""

from __future__ import annotations

import re

STATUSES = ("answered", "partial", "refused_no_data", "refused_offtopic", "refused_illegal")

_TURN_RE = re.compile(r"^(Пользователь|Ассистент)\s*:\s*", re.MULTILINE)


def split_dialog(raw_question: str) -> tuple[str, list[dict]]:
    """Разбирает вопрос с платформы на последнюю реплику пользователя и историю.

    На платформе история приходит внутри текста вопроса:
        Пользователь: ...
        Ассистент: ...
        Пользователь: ...
    Возвращает (последний вопрос, [{"role": "user"|"assistant", "content": ...}, ...]).
    """
    parts = _TURN_RE.split(raw_question.strip())
    if len(parts) < 3:  # обычный одиночный вопрос
        return raw_question.strip(), []
    turns = []
    for speaker, text in zip(parts[1::2], parts[2::2]):
        role = "user" if speaker == "Пользователь" else "assistant"
        turns.append({"role": role, "content": text.strip()})
    if turns and turns[-1]["role"] == "user":
        return turns[-1]["content"], turns[:-1]
    return raw_question.strip(), turns


def answer(question: str, history: list[dict] | None = None) -> dict:
    """Ответ на вопрос с учётом истории. ЗАГЛУШКА - заменить реальным пайплайном:
    поиск по индексу -> один вызов LLM -> самопроверка ссылок."""
    history = history or []
    return {
        "answer": f"[заглушка] Вопрос получен: {question}",
        "status": "answered",
        "sources": [
            {
                "doc_id": "demo",
                "doc_title": "Демо-документ",
                "point": "п. 1.1",
                "text": "Здесь будет фрагмент документа, подтверждающий ответ.",
                "score": 1.0,
            }
        ],
    }
