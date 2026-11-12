"""Клиент прокси платформы: /chat/completions, /embeddings, /rerank. Только httpx, с ретраями на 429/5xx."""
import random
import re
import sys
import time

import httpx
import numpy as np

import config

_RETRY_STATUS = {429, 500, 502, 503, 504}
_client = httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0),
                       limits=httpx.Limits(max_connections=64))


def _post(path: str, payload: dict, attempts: int = 5, api_key: str | None = None) -> dict:
    if not config.API_BASE:
        raise RuntimeError("LLM_BASE_URL не задан")
    url = f"{config.API_BASE}{path}"
    headers = {"Authorization": f"Bearer {api_key or config.LLM_KEY}"}
    last: Exception | None = None
    for i in range(attempts):
        wait = min(2 ** i, 20) + random.random()
        try:
            r = _client.post(url, json=payload, headers=headers)
            if r.status_code in _RETRY_STATUS:
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                ra = r.headers.get("retry-after", "")
                if ra.replace(".", "", 1).isdigit():
                    wait = min(float(ra), 30.0)
            else:
                r.raise_for_status()  # прочие 4xx — не ретраим
                return r.json()
        except httpx.TransportError as e:  # сеть, таймауты
            last = e
        if i < attempts - 1:
            time.sleep(wait)
    raise last  # type: ignore[misc]


def chat(messages: list[dict], max_tokens: int | None = None) -> str:
    data = _post("/chat/completions", {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else config.MAX_TOKENS,
    })
    text = (data["choices"][0]["message"].get("content") or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # на случай reasoning-модели
    return text.strip()


def embed(texts: list[str], batch: int = 32) -> np.ndarray:
    """Нормированные эмбеддинги (float32)."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        data = _post("/embeddings", {"model": config.EMB_MODEL, "input": texts[i:i + batch]},
                     api_key=config.EMBED_KEY)
        out.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0)))
    arr = np.asarray(out, dtype="float32")
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def rerank(query: str, docs: list[str], top_n: int) -> list[tuple[int, float]] | None:
    """[(индекс документа, оценка)] по убыванию; None, если реранкер недоступен (тогда работаем без него)."""
    if not config.RERANKER_MODEL or not docs:
        return None
    try:
        data = _post("/rerank", {"model": config.RERANKER_MODEL, "query": query,
                                 "documents": docs, "top_n": top_n}, attempts=3,
                     api_key=config.RERANK_KEY)
        items = data.get("results") or data.get("data") or []
        res = [(int(it["index"]), float(it.get("relevance_score", it.get("score", 0.0)))) for it in items]
        return sorted(res, key=lambda x: -x[1]) or None
    except Exception as e:  # реранк — улучшение, а не обязательный шаг
        print(f"[rerank] пропущен: {e}", file=sys.stderr)
        return None
