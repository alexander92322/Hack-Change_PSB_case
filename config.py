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
MAX_TOKENS = env_int("MAX_TOKENS", 1024, 128, 8192)
MAX_WORKERS = env_int("MAX_WORKERS", 4, 1, 16)
TEMPERATURE = env_float("TEMPERATURE", 0.0)

HTTP_TIMEOUT = 90.0
RETRIES = 2  # ограниченные повторы при 429 / 5xx / таймауте (требование регламента)
