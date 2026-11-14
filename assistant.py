"""Публичный контракт ассистента, общий для solution.py и HTTP API."""

from __future__ import annotations

from agent import LegalAgent, parse_history

STATUSES = ("answered", "partial", "refused_no_data", "refused_offtopic", "refused_illegal")

_DEFAULT_AGENT: LegalAgent | None = None


def warmup() -> LegalAgent:
    """Загрузить индекс один раз до запуска параллельных запросов."""
    global _DEFAULT_AGENT
    if _DEFAULT_AGENT is None:
        _DEFAULT_AGENT = LegalAgent()
    return _DEFAULT_AGENT


def split_dialog(raw_question: str) -> tuple[str, list[dict[str, str]]]:
    """Разбирает платформенную историю и возвращает последний вопрос и предыдущие ходы."""
    question, history = parse_history(raw_question)
    return question, history


def answer(question: str, history: list[dict] | None = None) -> dict:
    """Вернуть пользовательский ответ и найденные подтверждающие фрагменты.

    Контракт для backend сохраняется: answer/status/sources.
    """
    return warmup().answer(question, history)
