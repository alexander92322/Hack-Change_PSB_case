"""Точка входа для автоматической проверки на платформе.

Читает DATA_DIR/questions.csv (id,question), отвечает через assistant.answer()
и пишет submission.csv (id,answer) в рабочий каталог.
Вся логика ответа - в assistant.py; здесь только ввод/вывод и параллельность.
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
import kb
from assistant import answer, split_dialog

FALLBACK = "Не удалось сформировать ответ по базе знаний."
MAX_ANSWER_CHARS = 10_000


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
        rows = [(str(r["id"]).strip(), str(r["question"]).strip()) for r in reader]
    if not rows or any(not qid for qid, _ in rows):
        raise ValueError("В questions.csv отсутствуют вопросы или id")
    return rows


def clean(text: str) -> str:
    text = " ".join((text or "").split())  # одна физическая строка
    return (text or FALLBACK)[:MAX_ANSWER_CHARS]


def main() -> None:
    questions = load_questions(find_file("questions.csv"))
    docs = kb.load_documents()
    chunks = kb.load_chunks()
    print(
        f"Вопросов: {len(questions)}; Документов: {len(docs)}; Чанков: {len(chunks)}; MAX_WORKERS={config.MAX_WORKERS}",
        flush=True,
    )

    def work(position: int, raw: str) -> tuple[int, str]:
        try:
            question, history = split_dialog(raw)
            return position, clean(answer(question, history)["answer"])
        except Exception as exc:  # один упавший вопрос не должен ронять весь прогон
            print(f"Ошибка вопроса {position}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return position, FALLBACK

    answers = [FALLBACK] * len(questions)
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(work, i, q) for i, (_, q) in enumerate(questions)]
        for done, future in enumerate(as_completed(futures), 1):
            i, text = future.result()
            answers[i] = text
            if done % 10 == 0 or done == len(questions):
                print(f"Обработано {done}/{len(questions)}", flush=True)

    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for i, (qid, _) in enumerate(questions):
            writer.writerow([qid, answers[i]])
    print(f"Создан submission.csv: {len(questions)} ответов", flush=True)


if __name__ == "__main__":
    main()