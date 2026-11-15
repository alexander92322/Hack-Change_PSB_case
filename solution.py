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
from assistant import answer, split_dialog, warmup

FALLBACK = "Не удалось сформировать ответ по базе знаний. Уточните, пожалуйста, вопрос."
TIMEOUT_FALLBACK = "Не удалось обработать вопрос в отведённое время. Уточните, пожалуйста, вопрос."
MAX_ANSWER_CHARS = 10_000
RUN_LIMIT_SECONDS = 17 * 60  # платформа даёт 20 минут; оставляем запас на запись CSV


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

    def work(position: int, raw: str) -> tuple[int, str]:
        if time.monotonic() >= deadline:
            return position, TIMEOUT_FALLBACK
        try:
            question, history = split_dialog(raw)
            return position, clean(answer(question, history)["answer"])
        except Exception as exc:  # один упавший вопрос не должен ронять весь прогон
            print(f"Ошибка вопроса {position}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return position, FALLBACK

    answers = [FALLBACK] * len(questions)
    pool = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
    futures = {pool.submit(work, i, q): i for i, (_, q) in enumerate(questions)}
    pending = set(futures)
    processed = 0
    try:
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            for future in done:
                i, text = future.result()
                answers[i] = text
                processed += 1
                if processed % 10 == 0 or processed == len(questions):
                    print(f"Обработано {processed}/{len(questions)}", flush=True)
    finally:
        if pending:
            for future in pending:
                answers[futures[future]] = TIMEOUT_FALLBACK
                future.cancel()
            print(f"Дедлайн: не обработано вопросов - {len(pending)}", file=sys.stderr, flush=True)
        pool.shutdown(wait=False, cancel_futures=True)

    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for i, (qid, _) in enumerate(questions):
            writer.writerow([qid, answers[i]])
        f.flush()
        os.fsync(f.fileno())
    print(
        f"Создан submission.csv: {len(questions)} ответов за {time.monotonic() - started:.0f} с",
        flush=True,
    )
    if pending:
        # Зависшие запросы не должны помешать сдать уже готовый CSV.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
