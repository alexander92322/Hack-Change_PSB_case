"""Точка входа для автоматической проверки на платформе.

Читает DATA_DIR/questions.csv (id,question), отвечает через assistant.answer()
и пишет submission.csv (id,answer) в рабочий каталог.
Вся логика ответа - в agent.py; здесь только ввод/вывод, дедлайн и параллельность.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import config
import agent as agent_module
from assistant import answer, split_dialog, warmup


class _NoFastMode:
    def is_set(self) -> bool:
        return True

    def set(self) -> None:
        pass

FALLBACK = "Не удалось сформировать ответ по базе знаний. Уточните, пожалуйста, вопрос."
TIMEOUT_FALLBACK = "Не удалось обработать вопрос в отведённое время. Уточните, пожалуйста, вопрос."
MAX_ANSWER_CHARS = 10_000
RUN_LIMIT_SECONDS = 17 * 60  # после этого новые вопросы не начинаем
HARD_LIMIT_SECONDS = 18 * 60 + 30  # после этого не ждём зависшие запросы: пишем CSV и выходим
FAST_MODE_AT_SECONDS = 15 * 60  # если прогноз всего прогона больше - включаем быстрый режим


def find_file(*names: str) -> Path:
    for root in (config.DATA_DIR, Path(__file__).resolve().parent / config.DATA_DIR, Path.cwd()):
        for name in names:
            path = root / name
            if path.is_file():
                return path
    raise FileNotFoundError(f"Не найден один из файлов: {', '.join(names)}")


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("В questions.csv нужны колонки id и question")
        rows = [(str(r["id"]).strip(), str(r.get("question") or "").strip()) for r in reader]
    if not rows or any(not qid for qid, _ in rows):
        raise ValueError("В questions.csv отсутствуют вопросы или id")
    return rows


def clean(text: str, fallback: str = FALLBACK) -> str:
    text = " ".join((text or "").split())  # одна физическая строка
    return (text or fallback)[:MAX_ANSWER_CHARS]


def main() -> None:
    started = time.monotonic()
    deadline = started + RUN_LIMIT_SECONDS
    questions = load_questions(find_file("questions.csv"))
    agent = warmup()
    meta = getattr(agent, "meta", {}) or {}
    print(
        f"Вопросов: {len(questions)}; Документов: {len(meta.get('docs', []))}; "
        f"Чанков: {len(meta.get('chunks', []))}; MAX_WORKERS={config.MAX_WORKERS}; "
        f"дедлайн={RUN_LIMIT_SECONDS // 60} мин",
        flush=True,
    )

    statuses: dict[str, int] = {}
    durations: list[float] = []

    def fallback(raw: str) -> str:
        try:
            question, history = split_dialog(raw)
            return clean(agent.fallback_answer(question, history), TIMEOUT_FALLBACK)
        except Exception as exc:
            print(f"[fallback] {type(exc).__name__}: {exc}", file=sys.stderr)
            return TIMEOUT_FALLBACK

    def work(position: int, raw: str) -> tuple[int, str]:
        if time.monotonic() >= deadline:
            statuses["timeout_fallback"] = statuses.get("timeout_fallback", 0) + 1
            return position, fallback(raw)
        t0 = time.monotonic()
        try:
            question, history = split_dialog(raw)
            result = answer(question, history)
            status = str(result.get("status", "?"))
            text = clean(result["answer"])
        except Exception as exc:  # один упавший вопрос не должен ронять весь прогон
            print(f"Ошибка вопроса {position}: {type(exc).__name__}: {exc}", file=sys.stderr)
            status, text = "exception", fallback(raw)
        statuses[status] = statuses.get(status, 0) + 1
        durations.append(time.monotonic() - t0)
        return position, text

    answers = [FALLBACK] * len(questions)
    done_answered: set[int] = set()
    fast_mode = getattr(agent_module, "FAST_MODE", None) or _NoFastMode()
    pool = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
    pending = {pool.submit(work, i, q) for i, (_, q) in enumerate(questions)}
    done_count = 0
    while pending:
        left = started + HARD_LIMIT_SECONDS - time.monotonic()
        if left <= 0:
            break
        finished, pending = wait(pending, timeout=min(left, 30), return_when=FIRST_COMPLETED)
        for future in finished:
            i, text = future.result()
            answers[i] = text
            done_answered.add(i)
            done_count += 1
            elapsed = time.monotonic() - started
            # Прогноз по фактической скорости (уже учитывает параллельность). Если не успеваем
            # к 15 минутам - выключаем реранкер и сокращаем контекст для оставшихся вопросов.
            if (not fast_mode.is_set() and done_count >= max(3, config.MAX_WORKERS)
                    and elapsed + (len(questions) - done_count) * elapsed / done_count > FAST_MODE_AT_SECONDS):
                fast_mode.set()
                print(f"[time] прогноз не укладывается в лимит ({done_count}/{len(questions)} за "
                      f"{elapsed:.0f} с) — включаю быстрый режим", file=sys.stderr, flush=True)
            if done_count % 10 == 0 or done_count == len(questions):
                print(f"Обработано {done_count}/{len(questions)} за {time.monotonic() - started:.0f} с", flush=True)
    timed_out = len(pending)
    if timed_out:
        print(f"Жёсткий лимит времени: {timed_out} вопросов не дождались, отвечаем найденным фрагментом", file=sys.stderr)
        pool.shutdown(wait=False, cancel_futures=True)
        for i, (_, raw) in enumerate(questions):  # ответ без модели вместо пустой заглушки
            if i not in done_answered:
                answers[i] = fallback(raw)
    else:
        pool.shutdown(wait=True)

    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for i, (qid, _) in enumerate(questions):
            writer.writerow([qid, answers[i] or FALLBACK])
    if durations:
        ordered = sorted(durations)
        print(f"[stats] время на вопрос: медиана={ordered[len(ordered) // 2]:.1f} с, "
              f"макс={ordered[-1]:.1f} с; статусы: {statuses}; не дождались: {timed_out}", flush=True)
    try:
        print(agent_module.stats_summary(), flush=True)
    except Exception as exc:  # статистика не должна ломать сабмит
        print(f"[stats] недоступна: {exc}", file=sys.stderr)
    print(
        f"Создан submission.csv: {len(questions)} ответов за {time.monotonic() - started:.0f} с",
        flush=True,
    )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Если остались зависшие сетевые потоки, не ждём их: CSV уже записан.
    os._exit(0)
