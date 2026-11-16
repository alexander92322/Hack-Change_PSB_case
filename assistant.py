"""Публичный контракт ассистента, общий для solution.py и HTTP API.

answer(question, history) -> {"answer": str, "status": str, "sources": list[dict]}
status: answered | partial | refused_no_data | refused_offtopic | refused_illegal
"""

from __future__ import annotations

from agent import LegalAgent, get_agent, parse_history

STATUSES = ("answered", "partial", "refused_no_data", "refused_offtopic", "refused_illegal")


def warmup() -> LegalAgent:
    """Загрузить базу один раз до запуска параллельных запросов."""
    return get_agent()


def split_dialog(raw_question: str) -> tuple[str, list[dict[str, str]]]:
    """Разбирает платформенную историю и возвращает последний вопрос и предыдущие ходы."""
    question, history = parse_history(raw_question)
    return question, history


def answer(question: str, history: list[dict] | None = None) -> dict:
    """Вернуть пользовательский ответ и подтверждающие фрагменты. Контракт: answer/status/sources."""
    return warmup().answer(question, history)
