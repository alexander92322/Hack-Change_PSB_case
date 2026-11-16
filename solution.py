"""Точка входа для автоматической проверки на платформе.

Читает DATA_DIR/questions.csv (id,question), отвечает через агента и пишет submission.csv (id,answer)
в рабочий каталог. Вся логика ответа - в agent.py; здесь ввод/вывод, время и параллельность.

Гарантии этого файла (приоритет № 1 - прогон всегда завершается вовремя с корректным CSV):
- submission.csv появляется в первые секунды: в нём ответы без модели (дословные положения базы),
  затем файл перезаписывается по мере готовности настоящих ответов. Даже если процесс убьют,
  на диске лежит полный корректный файл;
- у каждого вопроса свой крайний срок; после мягкого лимита новые вызовы модели не начинаются,
  после жёсткого - файл дописывается и процесс завершается, не дожидаясь зависших запросов;
- сторожевой таймер завершит процесс, даже если главный поток застрянет;
- вопросы, на которые модель не ответила из-за сбоя прокси, повторяются в конце, если осталось время.
"""

from __future__ import annotations

import csv
import math
import os
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import config

STARTED = time.monotonic()
GENERIC = ("В базе знаний ООО «Меридиан» не нашлось положений по этому вопросу. По памяти отвечать не буду, "
           "так как нормы могли измениться; уточните вопрос или обратитесь к юристу.")
MAX_ANSWER_CHARS = 10_000
OUTPUT = "submission.csv"

_WRITE_LOCK = threading.Lock()
_STATE: dict = {"questions": [], "answers": []}


def elapsed() -> float:
    return time.monotonic() - STARTED


def find_file(*names: str) -> Path | None:
    for root in (config.DATA_DIR, Path(__file__).resolve().parent / config.DATA_DIR, Path.cwd() / "data", Path.cwd(),
                 Path(__file__).resolve().parent):
        for name in names:
            path = Path(root) / name
            if path.is_file():
                return path
    return None


def load_questions(path: Path) -> list[tuple[str, str]]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", "replace")
    rows = list(csv.reader(text.splitlines(keepends=True)))
    rows = [r for r in rows if r and any(cell.strip() for cell in r)]
    if not rows:
        raise ValueError("questions.csv пуст")
    header = [h.strip().lower() for h in rows[0]]
    if "id" in header and "question" in header:
        i_id, i_q, body = header.index("id"), header.index("question"), rows[1:]
    else:  # нет заголовка или другие названия колонок: считаем, что первые две колонки - id и вопрос
        print(f"[io] неожиданный заголовок questions.csv: {rows[0][:4]}; беру первые две колонки", file=sys.stderr)
        i_id, i_q = 0, 1 if len(rows[0]) > 1 else 0
        body = rows[1:] if header and header[0] in {"id", "#", "номер"} else rows
    out = []
    for n, r in enumerate(body):
        qid = (r[i_id] if i_id < len(r) else "").strip() or str(n)
        out.append((qid, (r[i_q] if i_q < len(r) else "").strip()))
    return out


def clean(text: str, fallback: str = GENERIC) -> str:
    text = " ".join(str(text or "").split())  # одна физическая строка
    return (text or fallback)[:MAX_ANSWER_CHARS]


def write_csv() -> None:
    """Атомарная запись текущего состояния ответов."""
    with _WRITE_LOCK:
        questions, answers = _STATE["questions"], list(_STATE["answers"])
        tmp = OUTPUT + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(["id", "answer"])
            for (qid, _), text in zip(questions, answers):
                writer.writerow([qid, clean(text)])
        os.replace(tmp, OUTPUT)


def finish(code: int = 0) -> None:
    try:
        write_csv()
    except Exception as exc:  # noqa: BLE001
        print(f"[io] ошибка записи CSV: {exc}", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # не ждём зависшие сетевые потоки: файл уже записан


def _watchdog() -> None:
    print(f"[time] сторожевой таймер: {elapsed():.0f} с, завершаю процесс с тем, что есть", file=sys.stderr)
    finish(0)


def _on_signal(signum, _frame) -> None:
    print(f"[time] получен сигнал {signum}, записываю CSV и выхожу", file=sys.stderr)
    finish(0)


def main() -> None:
    soft_end = STARTED + config.RUN_SOFT_LIMIT
    hard_end = STARTED + config.RUN_HARD_LIMIT
    timer = threading.Timer(config.RUN_HARD_LIMIT + 25, _watchdog)
    timer.daemon = True
    timer.start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    path = find_file("questions.csv")
    if path is None:
        raise FileNotFoundError("Не найден questions.csv")
    questions = load_questions(path)
    _STATE["questions"] = questions
    _STATE["answers"] = [GENERIC] * len(questions)
    write_csv()  # корректный файл существует с первой секунды

    import agent as agent_module  # импорт после записи файла: сбой загрузки базы не оставит нас без CSV
    agent = agent_module.get_agent()
    kb_zip = find_file("knowledge-base.zip", "kb.zip", "knowledge_base.zip")
    workers = config.MAX_WORKERS
    # Аварийный потолок параллельности: используется, только если прогон не укладывается во время
    # и прокси не отвечает ошибками 429 (иначе вопросы без ответа всё равно получили бы 0).
    max_workers = workers if config.EMERGENCY_WORKERS_FACTOR <= 1.0 else \
        max(workers, min(16, max(8, math.ceil(workers * config.EMERGENCY_WORKERS_FACTOR))))
    print(
        f"Вопросов: {len(questions)}; документов в базе: {len(agent.kb.docs)}; чанков: {len(agent.kb.chunks)}; "
        f"архив базы на платформе: {'найден ' + kb_zip.name if kb_zip else 'не найден (используется встроенная копия)'}; "
        f"MAX_WORKERS={workers} (аварийно до {max_workers}); MAX_TOKENS платформы="
        f"{config.MAX_TOKENS if config.MAX_TOKENS_FROM_ENV else 'не задан'}; TEMPERATURE={config.TEMPERATURE}; "
        f"модель={config.LLM_MODEL}; лимиты времени: мягкий {config.RUN_SOFT_LIMIT} с, жёсткий {config.RUN_HARD_LIMIT} с; "
        f"постоянный контекст: {len(agent.kb.static_text)} символов",
        flush=True,
    )

    # Ответы без модели для всех вопросов сразу: страховка на случай убийства процесса.
    for i, (_, raw) in enumerate(questions):
        _STATE["answers"][i] = clean(agent.fallback_answer(raw))
    write_csv()
    print(f"[time] страховочный CSV записан на {elapsed():.1f} с", flush=True)

    statuses: dict[str, int] = {}
    durations: list[float] = []
    calls: list[int] = []
    final_fallback: set[int] = set()
    samples: list[str] = []

    def work(i: int, deadline: float) -> tuple[int, dict, dict, float]:
        t0 = time.monotonic()
        try:
            result, info = agent.answer_ex(questions[i][1], None, deadline=deadline)
        except Exception as exc:  # noqa: BLE001 - один упавший вопрос не роняет прогон
            print(f"[q{i}] ошибка: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            result = {"answer": agent.fallback_answer(questions[i][1]), "status": "exception", "sources": []}
            info = {"fallback": True, "calls": 0, "error": str(exc)[:200]}
        return i, result, info, time.monotonic() - t0

    pool = ThreadPoolExecutor(max_workers=max_workers)
    state = {"limit": workers, "done": 0, "last_write": time.monotonic(), "last_raise": 0.0}

    def run_pass(indices: list[int], label: str) -> list[int]:
        """Обрабатывает вопросы; возвращает те, на которые модель не ответила (для повторного прохода)."""
        queue = list(indices)
        active: dict = {}
        failed: list[int] = []
        finished_here = 0
        pass_start = time.monotonic()
        while queue or active:
            now = time.monotonic()
            if now >= hard_end:
                break
            while queue and len(active) < state["limit"] and now < soft_end:
                i = queue.pop(0)
                remaining = len(queue) + len(active) + 1
                per_question = (soft_end - now) * state["limit"] / remaining
                budget = min(max(60.0, 3.0 * per_question), 210.0)
                deadline = min(hard_end - 8.0, now + budget)
                active[pool.submit(work, i, deadline)] = i
            if not active:
                break  # мягкий лимит: новые вопросы не начинаем
            done, _ = wait(list(active), timeout=max(0.1, min(5.0, hard_end - time.monotonic())),
                           return_when=FIRST_COMPLETED)
            for future in done:
                i = active.pop(future)
                _, result, info, seconds = future.result()
                finished_here += 1
                state["done"] += 1
                durations.append(seconds)
                calls.append(int(info.get("calls", 0)))
                if info.get("fallback"):
                    failed.append(i)
                    final_fallback.add(i)
                    # страховочный ответ уже записан; заменяем его только более свежим дословным
                    _STATE["answers"][i] = clean(result["answer"])
                else:
                    final_fallback.discard(i)
                    _STATE["answers"][i] = clean(result["answer"])
                    statuses[result.get("status", "?")] = statuses.get(result.get("status", "?"), 0) + 1
                    if len(samples) < 3:
                        samples.append(f"[пример] id={questions[i][0]} ({result.get('status')}): "
                                       f"{_STATE['answers'][i][:300]}")
                if state["done"] % 10 == 0:
                    print(f"[{label}] готово {finished_here}/{len(indices)} за {elapsed():.0f} с "
                          f"(параллельно {state['limit']}, сбоев модели {len(final_fallback)})", flush=True)
            # прогноз по фактической скорости этого прохода
            now = time.monotonic()
            if finished_here >= max(3, state["limit"]) and (queue or active):
                speed = finished_here / max(1e-6, now - pass_start)          # вопросов в секунду
                eta = now + (len(queue) + len(active)) / max(1e-6, speed)
                if eta > soft_end - 20 and not agent_module.FAST_MODE.is_set():
                    agent_module.FAST_MODE.set()
                    agent_module.llm.FAST.set()
                    print(f"[time] прогноз {eta - STARTED:.0f} с > лимита: включаю быстрый режим "
                          f"(без самопроверки, меньше контекста)", file=sys.stderr, flush=True)
                elif (eta > soft_end - 20 and state["limit"] < max_workers and now - state["last_raise"] > 15
                      and not agent_module.llm.recent_429() and not agent_module.llm.breaker_open()):
                    state["last_raise"] = now
                    state["limit"] = min(max_workers, state["limit"] + max(1, workers // 2))
                    print(f"[time] всё ещё не успеваем: параллельность поднята до {state['limit']}",
                          file=sys.stderr, flush=True)
            if now - state["last_write"] > 4:
                write_csv()
                state["last_write"] = now
        if active:
            print(f"[time] жёсткий лимит: {len(active)} вопросов не дождались ответа модели", file=sys.stderr)
            for i in active.values():
                failed.append(i)
                final_fallback.add(i)
        for i in queue:  # не успели даже начать
            failed.append(i)
            final_fallback.add(i)
        return failed

    failed = run_pass(list(range(len(questions))), "основной проход")
    write_csv()
    retried = 0
    if failed and time.monotonic() < soft_end - 30:
        retried = len(failed)
        print(f"[retry] повторяю {retried} вопросов, на которые модель не ответила", flush=True)
        failed = run_pass(sorted(set(failed)), "повторный проход")
    write_csv()
    pool.shutdown(wait=False, cancel_futures=True)

    total = elapsed()
    if durations:
        ordered = sorted(durations)
        p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
        print(f"[stats] время на вопрос: медиана={ordered[len(ordered) // 2]:.1f} с, p90={p90:.1f} с, "
              f"макс={ordered[-1]:.1f} с; вызовов модели на вопрос: среднее={sum(calls) / max(1, len(calls)):.2f}",
              flush=True)
    print(f"[stats] итог: вопросов={len(questions)}; ответов модели={len(questions) - len(final_fallback)} "
          f"по статусам {dict(sorted(statuses.items()))}; резервных ответов без модели={len(final_fallback)}; "
          f"ушло на повтор={retried}; параллельность в конце={state['limit']}", flush=True)
    if final_fallback:
        ids = [questions[i][0] for i in sorted(final_fallback)][:40]
        print(f"[stats] id вопросов с резервным ответом: {ids}", flush=True)
    for line in samples:
        print(line, flush=True)
    try:
        print(agent_module.stats_summary(), flush=True)
    except Exception as exc:  # noqa: BLE001 - статистика не должна ломать сабмит
        print(f"[stats] недоступна: {exc}", file=sys.stderr)
    print(f"Создан {OUTPUT}: {len(questions)} ответов за {total:.0f} с", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 - что бы ни случилось, оставляем корректный CSV
        if not isinstance(exc, SystemExit):
            import traceback
            traceback.print_exc()
            print(f"[fatal] {type(exc).__name__}: {exc}; записываю то, что есть", file=sys.stderr)
        if not _STATE["questions"]:
            # не удалось даже прочитать вопросы: пишем пустой корректный файл с заголовком
            try:
                with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
                    csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n").writerow(["id", "answer"])
            except OSError:
                pass
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
    finish(0)
