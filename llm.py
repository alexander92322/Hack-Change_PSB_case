"""Клиент прокси платформы: /chat/completions и /embeddings. Только httpx.

Главная задача этого файла - чтобы сбой или медлительность прокси никогда не роняли прогон:
- у каждого вызова есть крайний срок (deadline), после которого он не начинается и не повторяется;
- повторы только на 429 / 5xx / сетевые ошибки, с ограниченной паузой;
- «выключатель» (breaker): после серии сбоев подряд вызовы какое-то время отказывают мгновенно,
  чтобы очередь вопросов не простаивала на таймаутах; затем пробуем снова;
- параметры запроса подстраиваются под прокси: если он отвергает запрос с ошибкой 400/422,
  по очереди убираем необязательные поля и возвращаемся к платформенному MAX_TOKENS.
"""
from __future__ import annotations

import random
import re
import sys
import threading
import time
from dataclasses import dataclass

import httpx
import numpy as np

import config

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
_client = httpx.Client(
    timeout=httpx.Timeout(config.LLM_TIMEOUT, connect=10.0),
    limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
)


class LLMError(RuntimeError):
    """Модель не дала ответа (сеть, таймаут, отказ прокси, истёк срок)."""


@dataclass
class ChatResult:
    text: str
    finish_reason: str = ""
    truncated: bool = False       # ответ оборван лимитом токенов
    had_reasoning: bool = False   # прокси вернул отдельные рассуждения модели
    max_tokens: int = 0


# ------------------------------------------------------------------ статистика и выключатель
_LOCK = threading.Lock()
STATS = {
    "chat_calls": 0, "chat_ok": 0, "chat_retries": 0, "chat_failures": 0, "chat_timeouts": 0,
    "http_429": 0, "http_5xx": 0, "http_4xx": 0, "breaker_trips": 0, "breaker_fastfail": 0,
    "truncated": 0, "empty_content": 0, "reasoning_seen": 0, "param_fallbacks": 0,
    "emb_calls": 0, "emb_ok": 0, "emb_failures": 0, "emb_disabled": 0,
    "chat_seconds": 0.0, "first_error": "", "last_429_at": 0.0,
}
_BREAKER = {"fails": 0, "open_until": 0.0, "probe_at": 0.0}
BREAKER_THRESHOLD = 6        # сбоев подряд (по всем потокам), после которых «размыкаем»
BREAKER_OPEN_SECONDS = 12.0  # сколько не трогаем прокси перед пробным вызовом

# Что прокси согласился принять (меняется по ходу прогона при ошибках 400/422).
_NEGOTIATED = {"extras": True, "temperature": True, "own_max_tokens": True}
_EMB = {"fails": 0, "disabled": False}


def _stat(key: str, value: float = 1) -> None:
    with _LOCK:
        STATS[key] = STATS.get(key, 0) + value


def _note_error(text: str) -> None:
    with _LOCK:
        if not STATS["first_error"]:
            STATS["first_error"] = text[:300]


def breaker_open() -> bool:
    with _LOCK:
        return time.monotonic() < _BREAKER["open_until"]


def _breaker_allow() -> bool:
    """True - можно звать прокси. Когда выключатель разомкнут, пропускаем один пробный вызов за период."""
    now = time.monotonic()
    with _LOCK:
        if now >= _BREAKER["open_until"]:
            return True
        return False


def _breaker_result(ok: bool) -> None:
    now = time.monotonic()
    with _LOCK:
        if ok:
            _BREAKER["fails"] = 0
            _BREAKER["open_until"] = 0.0
            return
        _BREAKER["fails"] += 1
        if _BREAKER["fails"] >= BREAKER_THRESHOLD and now >= _BREAKER["open_until"]:
            _BREAKER["open_until"] = now + BREAKER_OPEN_SECONDS
            # после паузы достаточно одного нового сбоя, чтобы снова разомкнуть
            _BREAKER["fails"] = BREAKER_THRESHOLD - 1
            STATS["breaker_trips"] += 1


def recent_429(window: float = 30.0) -> bool:
    with _LOCK:
        return bool(STATS["last_429_at"]) and time.monotonic() - STATS["last_429_at"] < window


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"} if key else {}


def effective_max_tokens(retry: bool = False) -> int:
    """Лимит токенов ответа: платформенный; больший - только для повтора после обрыва (см. config.py)."""
    base = config.MAX_TOKENS if config.MAX_TOKENS_FROM_ENV else config.ANSWER_TOKENS_DEFAULT
    if config.RESPECT_PLATFORM_MAX_TOKENS or not _NEGOTIATED["own_max_tokens"]:
        return base
    if retry or _RAISED["on"]:
        return max(base, config.ANSWER_TOKENS_RETRY)
    return base


_RAISED = {"count": 0, "on": False}


def note_truncation_recovered() -> None:
    """Ответ был оборван лимитом, а с большим лимитом получился. После трёх таких случаев

    сразу просим большой лимит: иначе при «рассуждающей» модели каждый вопрос стоил бы двух вызовов."""
    with _LOCK:
        _RAISED["count"] += 1
        if _RAISED["count"] >= 3 and not _RAISED["on"]:
            _RAISED["on"] = True
            print(f"[llm] лимит токенов ответа поднят до {config.ANSWER_TOKENS_RETRY}: ответы обрывались",
                  file=sys.stderr, flush=True)


FAST = threading.Event()  # быстрый режим прогона (включает solution.py)


def _reasoning_effort() -> str:
    if FAST.is_set() and config.FAST_REASONING_EFFORT:
        return config.FAST_REASONING_EFFORT
    return config.REASONING_EFFORT


def _payload(messages: list[dict], max_tokens: int, opts: dict) -> dict:
    payload: dict = {"model": config.LLM_MODEL, "messages": messages, "max_tokens": int(max_tokens)}
    if opts["temperature"]:
        payload["temperature"] = min(config.TEMPERATURE, config.TEMPERATURE_CAP)
    if opts["extras"] and _reasoning_effort():
        payload["reasoning_effort"] = _reasoning_effort()
    return payload


def _downgrade_params(opts: dict, max_tokens: int, body: str) -> tuple[bool, int]:
    """Прокси отверг запрос (400/422): убираем один необязательный параметр - только для этого вызова.

    Что убирать, подсказывает текст ошибки; если подсказки нет - идём по порядку. Глобально решение
    закрепляется лишь после того, как урезанный запрос прошёл (см. chat), чтобы случайная ошибка 400
    не меняла параметры всего прогона.
    """
    low = body.lower()
    can_extras = opts["extras"] and bool(_reasoning_effort())
    base = config.MAX_TOKENS if config.MAX_TOKENS_FROM_ENV else config.ANSWER_TOKENS_DEFAULT
    can_tokens = opts["own_max_tokens"] and max_tokens > base
    can_temp = opts["temperature"]
    if can_extras and ("reasoning" in low or not (can_tokens and "token" in low) and not (can_temp and "temperature" in low)):
        opts["extras"] = False
    elif can_tokens and ("token" in low or not (can_temp and "temperature" in low)):
        opts["own_max_tokens"] = False
        max_tokens = base
    elif can_temp and ("temperature" in low or not ("context" in low or "too long" in low or "length" in low)):
        opts["temperature"] = False
    else:
        return False, max_tokens
    return True, max_tokens


def _content_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):  # некоторые прокси отдают список частей
        content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    text = str(content or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S) if "</think>" in text else text
    if "<think>" in text:  # рассуждение не закрыто - ответ оборван внутри него
        text = text.split("<think>", 1)[0]
    return text.strip()


def chat(messages: list[dict], max_tokens: int | None = None, *, deadline: float | None = None,
         attempts: int = 3, timeout: float | None = None) -> ChatResult:
    """Один ответ модели. deadline - момент time.monotonic(), после которого вызов не имеет смысла."""
    if not config.API_BASE:
        _stat("chat_failures")
        _note_error("LLM_BASE_URL не задан")
        raise LLMError("LLM_BASE_URL не задан")
    url = f"{config.API_BASE}/chat/completions"
    max_tokens = int(max_tokens or effective_max_tokens())
    _stat("chat_calls")
    last = "нет попыток"
    attempt = 0
    downgrades = 0
    with _LOCK:
        opts = dict(_NEGOTIATED)
    if not opts["own_max_tokens"]:
        max_tokens = min(max_tokens, config.MAX_TOKENS if config.MAX_TOKENS_FROM_ENV else config.ANSWER_TOKENS_DEFAULT)
    while attempt < max(1, attempts):
        now = time.monotonic()
        left = (deadline - now) if deadline is not None else 1e9
        if left < 4.0:
            last = f"истёк срок вопроса ({last})"
            break
        if not _breaker_allow():
            # Прокси только что отказал много раз подряд: не долбим его, а ждём конца паузы.
            # Если до срока вопроса пауза не кончится - сдаёмся сразу (вопрос уйдёт в повторный проход).
            with _LOCK:
                pause = _BREAKER["open_until"] - time.monotonic()
            if deadline is not None and time.monotonic() + pause > deadline - 4.0:
                _stat("breaker_fastfail")
                last = f"прокси временно отключён выключателем ({last})"
                break
            time.sleep(max(0.05, pause) + random.random() * 0.5)
            continue
        read_timeout = max(3.0, min(timeout or config.LLM_TIMEOUT, left - 1.0))
        t0 = time.monotonic()
        retry_after = None
        try:
            r = _client.post(url, json=_payload(messages, max_tokens, opts), headers=_headers(config.LLM_KEY),
                             timeout=httpx.Timeout(read_timeout, connect=min(10.0, read_timeout)))
            status = r.status_code
            if status == 200:
                data = r.json()
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = _content_text(message)
                finish = str(choice.get("finish_reason") or "")
                reasoning = bool(message.get("reasoning_content") or message.get("reasoning"))
                _breaker_result(True)
                if downgrades:  # урезанный запрос прошёл: запоминаем, что именно прокси не принимает
                    with _LOCK:
                        changed = [k for k in _NEGOTIATED if _NEGOTIATED[k] and not opts[k]]
                        for k in changed:
                            _NEGOTIATED[k] = False
                        STATS["param_fallbacks"] += len(changed)
                    if changed:
                        print(f"[llm] прокси не принимает параметры {changed}: дальше работаем без них",
                              file=sys.stderr, flush=True)
                _stat("chat_ok")
                _stat("chat_seconds", time.monotonic() - t0)
                if reasoning:
                    _stat("reasoning_seen")
                truncated = finish == "length"
                if truncated:
                    _stat("truncated")
                if not text:
                    _stat("empty_content")
                return ChatResult(text=text, finish_reason=finish, truncated=truncated,
                                  had_reasoning=reasoning, max_tokens=max_tokens)
            body = r.text[:300]
            last = f"HTTP {status}: {body}"
            if status in (400, 422) and downgrades < 3:
                changed, max_tokens = _downgrade_params(opts, max_tokens, body)
                if changed:  # это не сбой прокси, а несовместимый параметр: повторяем сразу
                    downgrades += 1
                    continue
            if status == 429:
                _stat("http_429")
                with _LOCK:
                    STATS["last_429_at"] = time.monotonic()
                ra = r.headers.get("retry-after", "")
                if ra.replace(".", "", 1).isdigit():
                    retry_after = min(float(ra), 20.0)
            elif status >= 500:
                _stat("http_5xx")
            else:
                _stat("http_4xx")
            _note_error(last)
            if status not in _RETRY_STATUS:
                _breaker_result(False)
                break  # 401/403/404 и прочие - повтор не поможет
            if status != 429:
                _breaker_result(False)
        except httpx.TimeoutException as exc:
            _stat("chat_timeouts")
            last = f"таймаут {read_timeout:.0f} с ({type(exc).__name__})"
            _note_error(last)
            _breaker_result(False)
        except (httpx.TransportError, ValueError, KeyError, TypeError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            _note_error(last)
            _breaker_result(False)
        attempt += 1
        if attempt < attempts:
            pause = retry_after if retry_after is not None else min(1.5 * (2 ** (attempt - 1)), 8.0) + random.random()
            if deadline is not None and time.monotonic() + pause > deadline - 4.0:
                break
            _stat("chat_retries")
            time.sleep(pause)
    _stat("chat_failures")
    raise LLMError(last)


def embed_query(text: str) -> np.ndarray | None:
    """Нормированный вектор вопроса или None (эмбеддинги - улучшение поиска, а не обязательный шаг)."""
    if not config.USE_EMBEDDINGS or not config.API_BASE or _EMB["disabled"]:
        return None
    _stat("emb_calls")
    try:
        r = _client.post(f"{config.API_BASE}/embeddings",
                         json={"model": config.EMB_MODEL, "input": [text[:4000]]},
                         headers=_headers(config.EMBED_KEY),
                         timeout=httpx.Timeout(config.EMB_TIMEOUT, connect=min(5.0, config.EMB_TIMEOUT)))
        r.raise_for_status()
        vec = np.asarray(r.json()["data"][0]["embedding"], dtype="float32").reshape(-1)
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(vec).all() or norm == 0:
            raise ValueError("пустой или некорректный вектор")
        _EMB["fails"] = 0
        _stat("emb_ok")
        return vec / norm
    except Exception as exc:  # noqa: BLE001 - любой сбой эмбеддингов не должен мешать ответу
        _stat("emb_failures")
        _note_error(f"embeddings: {type(exc).__name__}: {exc}")
        _EMB["fails"] += 1
        if _EMB["fails"] >= 4 and not _EMB["disabled"]:
            _EMB["disabled"] = True
            _stat("emb_disabled")
            print("[emb] эмбеддинги отключены до конца прогона: 4 сбоя подряд, работаем на BM25",
                  file=sys.stderr, flush=True)
        return None


def disable_embeddings(reason: str) -> None:
    if not _EMB["disabled"]:
        _EMB["disabled"] = True
        _stat("emb_disabled")
        print(f"[emb] эмбеддинги отключены: {reason}", file=sys.stderr, flush=True)


def stats_line() -> str:
    with _LOCK:
        s = dict(STATS)
    avg = s["chat_seconds"] / s["chat_ok"] if s["chat_ok"] else 0.0
    line = (
        f"[stats] модель: вызовов={s['chat_calls']} успешно={s['chat_ok']} повторов={s['chat_retries']} "
        f"сбоев={s['chat_failures']} таймаутов={s['chat_timeouts']} 429={s['http_429']} 5xx={s['http_5xx']} "
        f"4xx={s['http_4xx']}; среднее время вызова={avg:.1f} с; обрезано лимитом токенов={s['truncated']} "
        f"пустых ответов={s['empty_content']} ответов с рассуждениями={s['reasoning_seen']}; "
        f"выключатель: срабатываний={s['breaker_trips']} быстрых отказов={s['breaker_fastfail']}; "
        f"лимит токенов поднят={'да' if _RAISED['on'] else 'нет'}; подстроек параметров={s['param_fallbacks']} (reasoning_effort={(_reasoning_effort() or 'не задан') if _NEGOTIATED['extras'] else 'отвергнут прокси'}, "
        f"temperature={'да' if _NEGOTIATED['temperature'] else 'нет'}, свой max_tokens={'да' if _NEGOTIATED['own_max_tokens'] and not config.RESPECT_PLATFORM_MAX_TOKENS else 'нет'}); "
        f"эмбеддинги: вызовов={s['emb_calls']} успешно={s['emb_ok']} сбоев={s['emb_failures']} "
        f"отключены={'да' if _EMB['disabled'] else 'нет'}"
    )
    if s["first_error"]:
        line += f"; первая ошибка: {s['first_error']}"
    return line
