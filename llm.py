"""Клиент прокси платформы: /chat/completions, /embeddings, /rerank. Только httpx, с ретраями на 429/5xx."""
import random
import re
import sys
import time

import httpx
import numpy as np

import config

_RETRY_STATUS = {429, 500, 502, 503, 504}
_client = httpx.Client(timeout=httpx.Timeout(config.HTTP_TIMEOUT, connect=15.0),
                       limits=httpx.Limits(max_connections=64))


def _post(path: str, payload: dict, attempts: int | None = None,
          api_key: str | None = None, timeout: float | None = None) -> dict:
    if not config.API_BASE:
        raise RuntimeError("LLM_BASE_URL не задан")
    url = f"{config.API_BASE}{path}"
    key = api_key or config.LLM_KEY
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    attempts = max(1, attempts if attempts is not None else config.RETRIES + 1)
    last: Exception | None = None
    for i in range(attempts):
        wait = min(2 ** i, 20) + random.random()
        try:
            r = (_client.post(url, json=payload, headers=headers, timeout=timeout) if timeout
                 else _client.post(url, json=payload, headers=headers))
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
    # Рассуждение, оборванное лимитом токенов (без </think>), - не ответ: его черновые
    # утверждения не должны попасть пользователю и судье. Пустой текст -> запасной ответ.
    text = re.sub(r"<think>.*\Z", "", text, flags=re.S)
    return text.strip()


class _Breaker:
    """Предохранитель: после нескольких неудач подряд (и ни одного успеха) сервис больше не зовём.

    Иначе недоступный /embeddings или /rerank съедает по минуте-две на каждый вопрос,
    и прогон не укладывается в 20 минут. Поиск при этом работает без него (BM25 + правила).
    """

    PROBE_AFTER = 120.0  # раз в 2 минуты одна пробная попытка: кратковременный сбой не отключает навсегда

    def __init__(self, name: str, limit: int = 3):
        self.name, self.limit = name, limit
        self.failures = 0
        self.successes = 0
        self.opened_at = 0.0

    @property
    def open(self) -> bool:
        if not (self.successes == 0 and self.failures >= self.limit):
            return False
        if time.monotonic() - self.opened_at >= self.PROBE_AFTER:
            self.opened_at = time.monotonic()  # пропускаем одну пробу, остальные ждут следующего окна
            return False
        return True

    def ok(self) -> None:
        self.successes += 1
        self.failures = 0

    def fail(self, exc: Exception) -> None:
        self.failures += 1
        if self.successes == 0 and self.failures == self.limit:
            self.opened_at = time.monotonic()
            print(f"[{self.name}] {self.limit} неудач подряд — отключаю (проба раз в 2 мин): {exc}",
                  file=sys.stderr, flush=True)


EMBED_BREAKER = _Breaker("embeddings", limit=5)
RERANK_BREAKER = _Breaker("rerank", limit=3)


def embed(texts: list[str], batch: int = 32, timeout: float | None = None) -> np.ndarray:
    """Нормированные эмбеддинги (float32)."""
    if timeout is not None and EMBED_BREAKER.open:
        raise RuntimeError("эмбеддинги отключены после повторных ошибок")
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        try:
            data = _post("/embeddings", {"model": config.EMB_MODEL, "input": texts[i:i + batch]},
                         api_key=config.EMBED_KEY, timeout=timeout,
                         attempts=2 if timeout is not None else None)
        except Exception as exc:
            if timeout is not None:
                EMBED_BREAKER.fail(exc)
            raise
        if timeout is not None:
            EMBED_BREAKER.ok()
        out.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0)))
    arr = np.asarray(out, dtype="float32")
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def rerank(query: str, docs: list[str], top_n: int) -> list[tuple[int, float]] | None:
    """[(индекс документа, оценка)] по убыванию; None, если реранкер недоступен (тогда работаем без него)."""
    if not config.RERANKER_MODEL or not docs or RERANK_BREAKER.open:
        return None
    try:
        # Реранк - улучшение: долго не ждём, чтобы зависший сервис не съел лимит прогона.
        data = _post("/rerank", {"model": config.RERANKER_MODEL, "query": query,
                                 "documents": docs, "top_n": top_n}, attempts=2,
                     api_key=config.RERANK_KEY, timeout=30.0)
        items = data.get("results") or data.get("data") or []
        res = [(int(it["index"]), float(it.get("relevance_score", it.get("score", 0.0)))) for it in items]
        RERANK_BREAKER.ok()
        return sorted(res, key=lambda x: -x[1]) or None
    except Exception as e:  # реранк — улучшение, а не обязательный шаг
        RERANK_BREAKER.fail(e)
        print(f"[rerank] пропущен: {e}", file=sys.stderr)
        return None
