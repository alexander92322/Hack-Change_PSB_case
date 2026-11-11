"""Надёжный HTTP-клиент для обращения к API платформы (LLM, Embeddings, Reranker).

Реализует таймауты, повторные попытки (retries) при 429/5xx и сетевых сбоях,
а также семафор для ограничения числа одновременных запросов.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Any

import httpx

import config

# Ограничиваем число одновременных HTTP-запросов к API
_API_SEMAPHORE = threading.Semaphore(config.MAX_WORKERS)

# Коды ответа, при которых нужно повторить запрос
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# Единый клиент с пулом соединений (потокобезопасен в httpx)
_CLIENT: httpx.Client | None = None
_CLIENT_LOCK = threading.Lock()


def get_client() -> httpx.Client:
    """Возвращает переиспользуемый потокобезопасный httpx.Client."""
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        with _CLIENT_LOCK:
            if _CLIENT is None or _CLIENT.is_closed:
                limits = httpx.Limits(
                    max_connections=config.MAX_WORKERS * 2,
                    max_keepalive_connections=config.MAX_WORKERS,
                )
                _CLIENT = httpx.Client(
                    timeout=httpx.Timeout(config.HTTP_TIMEOUT, connect=10.0),
                    limits=limits,
                )
    return _CLIENT


def post_with_retry(path: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Отправляет POST-запрос к API_BASE + path с таймаутом, семафором и ретраями."""
    if not config.API_BASE:
        raise ValueError("Не задан LLM_BASE_URL (или BASE_URL / OPENAI_BASE_URL)")
    if not api_key:
        raise ValueError(f"Отсутствует API-ключ для запроса к {path}")

    url = f"{config.API_BASE}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {api_key}"}
    client = get_client()

    max_attempts = config.RETRIES + 1
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            with _API_SEMAPHORE:
                response = client.post(url, headers=headers, json=payload)

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_attempts:
                retry_after = response.headers.get("Retry-After")
                wait_sec = float(retry_after) if (retry_after and retry_after.isdigit()) else (1.5 * (2 ** (attempt - 1)))
                wait_sec = min(wait_sec, 15.0)
                print(
                    f"[API] Код {response.status_code} на {path} (попытка {attempt}/{max_attempts}), пауза {wait_sec:.1f}с",
                    file=sys.stderr,
                )
                time.sleep(wait_sec)
                continue

            response.raise_for_status()
            return response.json()

        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                wait_sec = 1.5 * (2 ** (attempt - 1))
                print(
                    f"[API] Сетевой сбой {type(exc).__name__} на {path} (попытка {attempt}/{max_attempts}), пауза {wait_sec:.1f}с",
                    file=sys.stderr,
                )
                time.sleep(wait_sec)
                continue
            raise
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            raise

    raise RuntimeError(f"Не удалось выполнить запрос к {path}") from last_exc


def chat_completion(
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Вызов LLM (/chat/completions) с возвратом непустого текста ответа."""
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.TEMPERATURE if temperature is None else temperature,
        "max_tokens": config.MAX_TOKENS if max_tokens is None else max_tokens,
    }
    result = post_with_retry("/chat/completions", config.LLM_KEY, payload)
    content = result["choices"][0]["message"].get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Модель вернула пустой ответ")
    return content.strip()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Вызов эмбеддера (/embeddings) с L2-нормализацией векторов."""
    if not texts:
        return []
    result = post_with_retry(
        "/embeddings",
        config.EMBED_KEY,
        {
            "model": config.EMB_MODEL,
            "input": texts,
            "encoding_format": "float",
        },
    )
    vectors: list[list[float] | None] = [None] * len(texts)
    for pos, item in enumerate(result["data"]):
        idx = int(item.get("index", pos))
        if 0 <= idx < len(texts):
            vec = item["embedding"]
            norm = math.sqrt(sum(float(x) ** 2 for x in vec)) or 1.0
            vectors[idx] = [float(x) / norm for x in vec]
    if any(v is None for v in vectors):
        raise ValueError("Получены не все embeddings от API")
    return vectors  # type: ignore[return-value]


def rerank_documents(query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """Вызов реранкера (/rerank). Возвращает список (индекс_документа, relevance_score)."""
    if not docs or not config.RERANKER_MODEL:
        return [(i, 0.0) for i in range(min(top_n or config.RERANKER_TOP_N, len(docs)))]

    limit = min(top_n or config.RERANKER_TOP_N, len(docs))
    result = post_with_retry(
        "/rerank",
        config.RERANK_KEY,
        {
            "model": config.RERANKER_MODEL,
            "query": query,
            "documents": docs,
            "top_n": limit,
        },
    )
    items = result.get("results", result.get("data", []))
    ranked: list[tuple[int, float]] = []
    for item in items:
        idx = int(item["index"])
        score = float(item.get("relevance_score", item.get("score", 0.0)))
        if 0 <= idx < len(docs):
            ranked.append((idx, score))
    return ranked[:limit]