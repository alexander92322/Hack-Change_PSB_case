"""Единая конфигурация: всё читается из переменных окружения.

Используется и solution.py (платформа), и веб-бэкендом. Ничего не хардкодим:
на платформе значения подставляются автоматически, локально - из .env.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:  # локально удобно держать значения в .env; на платформе файла нет
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv может отсутствовать в окружении сабмита
    pass


def env(*names: str, default: str = "") -> str:
    """Первое непустое значение из списка имён (регламент допускает синонимы)."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return default


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float = 0.0, high: float = 2.0) -> float:
    try:
        return max(low, min(high, float(env(name))))
    except ValueError:
        return default


DATA_DIR = Path(env("DATA_DIR", default="data"))

# Локальный офлайн-путь используется только при явной настройке; сабмит работает через API.
EMB_BACKEND = env("EMB_BACKEND", default="api")
LOCAL_EMB_PATH = env("LOCAL_EMB_PATH")

API_BASE = env("LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL").rstrip("/")
LLM_KEY = env("LLM_API_KEY", "OPENAI_API_KEY")
EMBED_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY

LLM_MODEL = env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL", default="deepseek/deepseek-v4.1-flash")
EMB_MODEL = env("EMB_MODEL", default="baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL")  # имя реранкера берём только из окружения

TOP_K = env_int("TOP_K", 8, 1, 50)
RERANKER_TOP_N = env_int("RERANKER_TOP_N", 5, 1, 20)
MAX_TOKENS = env_int("MAX_TOKENS", 1024, 16, 400000)
MAX_TOKENS_FROM_ENV = bool(env("MAX_TOKENS"))
MAX_WORKERS = env_int("MAX_WORKERS", 6, 1, 32)
TEMPERATURE = env_float("TEMPERATURE", 0.0)

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 90.0, 5.0, 600.0)  # таймаут запроса веб-клиента (api_client.py)
RETRIES = 2  # ограниченные повторы при 429 / 5xx / таймауте (требование регламента)


# ===================================================================== переключатели решения
# Всё ниже можно менять либо переменной окружения, либо прямо здесь (константой) перед сборкой ZIP.
# Значения по умолчанию - то, что мы считаем лучшим вариантом; откат к более простому поведению
# делается одной строкой (см. отчёт).

def env_flag(name: str, default: bool) -> bool:
    value = env(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on", "да"}


# Что всегда лежит в контексте модели:
#   "full"    - все документы компании + короткие статьи законов (≈40 тыс. символов), поиск только
#               по большим статьям (КоАП 13.11, НК 145/164/224/284);
#   "company" - всегда только документы компании (≈15 тыс. символов), все законы - через поиск.
CONTEXT_MODE = env("CONTEXT_MODE", default="full").lower()
# Сколько символов больших статей можно добавить к одному вопросу.
DYN_BUDGET_CHARS = env_int("DYN_BUDGET_CHARS", 48000, 4000, 400000)

# Самопроверка: "off" - один вызов модели; "targeted" - второй вызов только если код нашёл в ответе
# число или ссылку, которых нет в документах; "full" - второй вызов-проверка для каждого ответа.
SELF_CHECK = env("SELF_CHECK", default="targeted").lower()
# Короткий разбор перед ответом (<analysis>...</analysis>): помогает слабой модели с датами,
# редакциями и похожими контрагентами. Выключается сам, если лимит токенов маленький.
USE_ANALYSIS = env_flag("USE_ANALYSIS", True)
# Разрешить модели запросить непоказанную статью (<need>NK164</need>) - ещё один вызов.
ALLOW_NEED = env_flag("ALLOW_NEED", True)
# Эмбеддинги вопроса через прокси для поиска по большим статьям (при сбое - только BM25).
USE_EMBEDDINGS = env_flag("USE_EMBEDDINGS", True)

# Лимит токенов ответа.
# По умолчанию используем значение платформы MAX_TOKENS как есть (если она его не задала - ANSWER_TOKENS_DEFAULT).
# Модель DeepSeek V4.1 Flash - рассуждающая: если прокси считает рассуждения в тот же лимит, ответ может
# прийти пустым или оборванным. Только в этом случае вопрос повторяется с лимитом ANSWER_TOKENS_RETRY
# («спасательный» повтор), а после трёх таких случаев большой лимит используется сразу.
# RESPECT_PLATFORM_MAX_TOKENS=1 запрещает превышать платформенное значение даже при обрывах.
ANSWER_TOKENS_DEFAULT = env_int("ANSWER_TOKENS_DEFAULT", 1200, 64, 100000)
ANSWER_TOKENS_RETRY = env_int("ANSWER_TOKENS_RETRY", 4000, 64, 200000)
RESPECT_PLATFORM_MAX_TOKENS = env_flag("RESPECT_PLATFORM_MAX_TOKENS", False)
# Температуру берём с платформы, но не выше этого значения: для фактов нужна стабильность.
TEMPERATURE_CAP = env_float("TEMPERATURE_CAP", 0.3)
# Необязательный параметр reasoning_effort ("low" | "high" | "max"); пусто - не отправляем вовсе.
REASONING_EFFORT = env("REASONING_EFFORT")
# То же, но только в быстром режиме (когда прогон не укладывается во время): короче рассуждения - быстрее ответ.
# Если прокси параметр не принимает, он автоматически перестаёт отправляться. Пусто - не использовать.
FAST_REASONING_EFFORT = env("FAST_REASONING_EFFORT", default="low")

# Время (секунды от старта solution.py).
RUN_SOFT_LIMIT = env_int("RUN_SOFT_LIMIT", 16 * 60 + 30, 30, 100000)   # после - новые вызовы модели не начинаем
RUN_HARD_LIMIT = env_int("RUN_HARD_LIMIT", 18 * 60 + 15, 40, 100000)   # после - пишем CSV и выходим
LLM_TIMEOUT = env_float("LLM_TIMEOUT", 75.0, 3.0, 600.0)               # ожидание одного ответа модели
EMB_TIMEOUT = env_float("EMB_TIMEOUT", 8.0, 1.0, 120.0)
# Если не успеваем - разрешаем поднять параллельность выше MAX_WORKERS (не более этого множителя).
EMERGENCY_WORKERS_FACTOR = env_float("EMERGENCY_WORKERS_FACTOR", 2.0, 1.0, 4.0)
