"""Надёжный HTTP-клиент для API платформы с таймаутами, ретраями, семафором и кэшем.

Реализует:
1. Таймауты, повторные попытки (retries) при 429/5xx и сетевых сбоях.
2. Семафор для ограничения числа одновременных запросов к API.
3. Потокобезопасный LRU-кэш для эмбеддингов запросов и готовых ответов.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import math
import sys
import threading
import time
from typing import Any, Callable

import httpx

import config

_API_SEMAPHORE = threading.Semaphore(config.MAX_WORKERS)
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

_CLIENT: httpx.Client | None = None
_CLIENT_LOCK = threading.Lock()

# Потокобезопасные кэши (ограничены по размеру, чтобы не съесть всю RAM)
_CACHE_LOCK = threading.Lock()
_EMBED_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_ANSWER_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MAX_CACHE_SIZE = 2048


def _cache_get(cache: OrderedDict[str, Any], key: str) -> Any | None:
    with _CACHE_LOCK:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
    return None


def _cache_set(cache: OrderedDict[str, Any], key: str, value: Any) -> None:
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _MAX_CACHE_SIZE:
            cache.popitem(last=False)


def clear_caches() -> None:
    """Очищает кэш эмбеддингов и ответов (полезно для тестов)."""
    with _CACHE_LOCK:
        _EMBED_CACHE.clear()
        _ANSWER_CACHE.clear()


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
                wait_sec = (
                    float(retry_after)
                    if (retry_after and retry_after.isdigit())
                    else (1.5 * (2 ** (attempt - 1)))
                )
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
    """Вызов эмбеддера (/embeddings) с L2-нормализацией и кэшированием по тексту."""
    if not texts:
        return []

    results: list[list[float] | None] = [None] * len(texts)
    missing_indices: list[int] = []
    missing_texts: list[str] = []

    # 1. Проверяем, какие тексты уже есть в кэше
    for idx, text in enumerate(texts):
        cache_key = f"{config.EMB_MODEL}:{text.strip()}"
        cached_vec = _cache_get(_EMBED_CACHE, cache_key)
        if cached_vec is not None:
            results[idx] = cached_vec
        else:
            missing_indices.append(idx)
            missing_texts.append(text)

    # 2. По сети запрашиваем только те тексты, которых не было в кэше
    if missing_texts:
        api_result = post_with_retry(
            "/embeddings",
            config.EMBED_KEY,
            {
                "model": config.EMB_MODEL,
                "input": missing_texts,
                "encoding_format": "float",
            },
        )
        fetched: list[list[float] | None] = [None] * len(missing_texts)
        for pos, item in enumerate(api_result["data"]):
            i = int(item.get("index", pos))
            if 0 <= i < len(missing_texts):
                vec = item["embedding"]
                norm = math.sqrt(sum(float(x) ** 2 for x in vec)) or 1.0
                fetched[i] = [float(x) / norm for x in vec]

        if any(v is None for v in fetched):
            raise ValueError("Получены не все embeddings от API")

        for orig_idx, text, vec in zip(missing_indices, missing_texts, fetched):
            assert vec is not None
            cache_key = f"{config.EMB_MODEL}:{text.strip()}"
            _cache_set(_EMBED_CACHE, cache_key, vec)
            results[orig_idx] = vec

    return results  # type: ignore[return-value]


def cached_answer(
    question: str,
    history: list[dict[str, str]] | None,
    compute_fn: Callable[[str, list[dict[str, str]]], dict[str, Any]],
) -> dict[str, Any]:
    """Кэширует итоговый словарь ответа ассистента по нормализованному вопросу и истории."""
    norm_history = history or []
    key_payload = json.dumps(
        {"q": " ".join(question.split()).lower(), "h": norm_history},
        ensure_ascii=False,
        sort_keys=True,
    )
    cached = _cache_get(_ANSWER_CACHE, key_payload)
    if cached is not None:
        return dict(cached)

    fresh = compute_fn(question, norm_history)
    # Не кэшируем аварийные заглушки при сбое сети
    if fresh.get("answer") and "Не удалось сформировать ответ" not in fresh["answer"]:
        _cache_set(_ANSWER_CACHE, key_payload, dict(fresh))
    return fresh


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