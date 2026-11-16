"""Логика ответа ассистента.

Конвейер одного вопроса:
1. Разбор истории диалога из текста вопроса.
2. Контекст: документы компании и короткие статьи законов - всегда целиком; большие статьи
   (КоАП 13.11, НК 145/164/224/284) - по правилам и поиску (kb_store.py).
3. Вызов модели. Ответ - обычный текст в тегах <analysis>/<answer>, без JSON: обрезанный или
   «кривой» вывод всё равно даёт пригодный текст.
4. Мягкая постобработка: ничего не вырезаем и не заменяем заглушками. Если код нашёл в ответе число
   или ссылку, которых нет в документах, модель один раз просят перепроверить именно это место.
5. Если модель недоступна - дословное положение базы со ссылкой (без служебных фраз).

Публичный контракт для веб-интерфейса сохранён: LegalAgent.answer() -> {answer, status, sources}.
"""
from __future__ import annotations

import re
import sys
import threading
import time

import config
import kb_store
import llm
import prompts

STATUSES = ("answered", "partial", "refused_no_data", "refused_offtopic", "refused_illegal")
# Оставлены для совместимости со старым кодом, который мог импортировать эти имена.
REFUSAL = ("В базе знаний ООО «Меридиан» не нашлось положений по этому вопросу. По памяти отвечать не буду, "
           "так как нормы могли измениться; уточните вопрос или обратитесь к юристу.")
PROCESSING_ERROR = REFUSAL
_TYPE_STATUS = {"A": "answered", "B": "partial", "C": "answered", "D": "refused_no_data",
                "E": "refused_offtopic", "F": "refused_illegal"}
_CYR_TO_LAT = {"А": "A", "В": "B", "С": "C", "Д": "D", "Е": "E", "Ф": "F"}
MAX_ANSWER_WORDS = 220
MAX_ANSWER_CHARS = 10000
MIN_ANSWER_CHARS = 12
ANALYSIS_MIN_TOKENS = 600     # при меньшем лимите разбор не просим: весь лимит нужен ответу
FAST_DYN_BUDGET = 20000

# Быстрый режим включает solution.py, когда прогон не укладывается во время:
# без второго вызова, без дозапроса статей, с меньшим объёмом больших статей.
FAST_MODE = threading.Event()

_ROLE_MARKER = re.compile(r"(?:^|\n|\s/\s|\s\|\s)\s*(Пользователь|Ассистент|user|assistant)\s*:\s*", flags=re.I)


# ------------------------------------------------------------------ история диалога
def _parse_serialized_history(text: str) -> list[dict[str, str]]:
    """Разбирает историю вида «Пользователь: … / Ассистент: …»."""
    matches = list(_ROLE_MARKER.finditer(text or ""))
    messages = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[match.end():end].strip().strip("/|").strip()
        role = "user" if match.group(1).lower() in {"пользователь", "user"} else "assistant"
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _normalise_history(history) -> list[dict[str, str]]:
    """Принимает список role/content или сериализованный текст диалога."""
    if not history:
        return []
    if isinstance(history, str):
        parsed = _parse_serialized_history(history)
        return parsed or [{"role": "user", "content": history.strip()}]
    if not isinstance(history, (list, tuple)):
        return []
    messages = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        role = {"пользователь": "user", "ассистент": "assistant"}.get(role, role)
        content = item.get("content", item.get("text", ""))
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    return messages


def parse_history(text: str) -> tuple[str, list[dict[str, str]]]:
    """Вернуть последнюю реплику пользователя и предыдущие реплики диалога."""
    text = (text or "").strip()
    messages = _parse_serialized_history(text)
    last_user = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), None)
    if last_user is None:
        return text, []
    first = _ROLE_MARKER.search(text)
    prefix = text[:first.start()].strip() if first and first.start() > 0 else ""
    history = messages[:last_user]
    if prefix:  # текст до первой метки роли - тоже часть диалога, не теряем его
        history = [{"role": "user", "content": prefix}] + history
    return messages[last_user]["content"], history


def last_user_message(text: str) -> str:
    return parse_history(text)[0]


# ------------------------------------------------------------------ статистика
_STATS_LOCK = threading.Lock()
RUN_STATS: dict[str, int] = {}


def _count(*keys: str) -> None:
    with _STATS_LOCK:
        for key in keys:
            RUN_STATS[key] = RUN_STATS.get(key, 0) + 1


def stats_summary() -> str:
    with _STATS_LOCK:
        s = dict(RUN_STATS)
    order = ["questions", "answered", "partial", "refused_no_data", "refused_offtopic", "refused_illegal",
             "fallback_extractive", "fallback_empty_question"]
    head = "; ".join(f"{k}={s.pop(k, 0)}" for k in order)
    rest = "; ".join(f"{k}={v}" for k, v in sorted(s.items()))
    return (f"[stats] ответы: {head}\n[stats] защиты и ветки: {rest or 'нет срабатываний'}; "
            f"быстрый режим: {'да' if FAST_MODE.is_set() else 'нет'}; SELF_CHECK={config.SELF_CHECK}; "
            f"CONTEXT_MODE={config.CONTEXT_MODE}\n{llm.stats_line()}")


# ------------------------------------------------------------------ разбор вывода модели
_KEY_LEAK = re.compile(r"\s*\[(?:A08|P17|DS1|PT05|REG2025|REG2023|K12|YUO7|TK\d+|MROT|FZ152|NK\d+|KOAP1311)\]")
_SENT_END = re.compile(r"[.!?…][»\")]*(?=\s|$)")


def clean_answer(text: str) -> str:
    """Приводит ответ к одной строке обычного текста. Ничего по смыслу не удаляет."""
    text = text or ""
    text = re.sub(r"</?(?:answer|analysis|need|history|question|documents?|catalog|previous_answer)[^>]*>", " ", text, flags=re.I)
    text = re.sub(r"```\w*", " ", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.S)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+(?=\S)", "", text) if text.count("\n") else text
    text = _KEY_LEAK.sub("", text)
    text = " ".join(text.split())
    text = re.sub(r"^(?:Ответ|Ответ пользователю|Итоговый ответ)\s*[:\-–—]\s*", "", text, flags=re.I)
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text.strip()


def _cut_to_sentence(text: str, max_words: int | None = None) -> str:
    """Обрезает по границе предложения (для оборванных или слишком длинных ответов)."""
    if max_words is not None:
        words = text.split()
        if len(words) <= max_words:
            return text
        text = " ".join(words[:max_words])
    ends = [m.end() for m in _SENT_END.finditer(text)]
    # не режем на сокращениях вида «п.», «ст.», «г.»
    ends = [e for e in ends if not re.search(r"(?:\b(?:п|пп|ст|ч|г|д|подп|абз|руб|т|тыс|млн|стр)\.|\d\.)$", text[:e])]
    return text[:ends[-1]].strip() if ends else text.strip()


def parse_output(raw: str) -> dict:
    """Достаёт из вывода модели ответ, тип ситуации и запрос статей. Терпит обрезку и отсутствие тегов."""
    raw = (raw or "").strip()
    out = {"answer": "", "type": "", "need": [], "analysis": ""}
    out["need"] = [k.strip().upper() for part in re.findall(r"<need>(.*?)(?:</need>|$)", raw, flags=re.S | re.I)
                   for k in re.split(r"[,\s;]+", part) if k.strip()]
    m = re.search(r"<analysis>(.*?)(?:</analysis>|<answer>|$)", raw, flags=re.S | re.I)
    if m:
        out["analysis"] = m.group(1).strip()
        t = re.search(r"тип\s*[:\-–—]?\s*\**\s*([A-FАВСДЕФ])\b", out["analysis"], flags=re.I)
        if t:
            letter = t.group(1).upper()
            out["type"] = _CYR_TO_LAT.get(letter, letter)
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", raw, flags=re.S | re.I)
    if m:
        body = m.group(1)
    elif re.search(r"<analysis>", raw, flags=re.I):
        closed = re.search(r"</analysis>(.*)$", raw, flags=re.S | re.I)
        body = closed.group(1) if closed else ""   # разбор не закрыт - вывод оборван до ответа
    else:
        body = re.sub(r"<need>.*?(?:</need>|$)", " ", raw, flags=re.S | re.I)
    out["answer"] = clean_answer(body)
    return out


def guess_status(answer: str, kind: str = "") -> str:
    if kind in _TYPE_STATUS:
        return _TYPE_STATUS[kind]
    n = kb_store.norm(answer)
    if n.startswith("я отвечаю только на юридические"):
        return "refused_offtopic"
    if n.startswith(("не могу помочь", "я не могу помочь", "не могу содействовать")):
        return "refused_illegal"
    if re.match(r"^в базе знаний (?:ооо «меридиан» )?(?:нет|не нашлось|отсутству)", n):
        return "refused_no_data"
    if re.search(r"в базе (?:знаний )?(?:нет|отсутству)|по памяти отвечать не буду", n):
        return "partial"
    return "answered"


# ------------------------------------------------------------------ проверки ответа кодом
_DOC_MARKERS = {
    "A08": r"а-08/2024|северн\w+ склад",
    "P17": r"п-17/2024|альфа-трейд»|альфа-трейд\b(?!инг)",
    "PT05": r"пт-05/2023|альфа-трейдинг",
    "REG2023": r"редакци\w+ от 10\.01\.2023|ред\. от 10\.01\.2023|редакци\w+ 2023",
    "REG2025": r"редакци\w+ от 0?1\.03\.2025|ред\. от 0?1\.03\.2025|редакци\w+ 2025|действующ\w+ редакци|действующ\w+ регламент",
}
_ALWAYS_OK_NUMBERS = {str(i) for i in range(0, 11)} | {str(y) for y in range(1990, 2036)} | {"100"}


def check_answer(kb: kb_store.KB, answer: str, context: str, user_text: str) -> list[str]:
    """Находит в ответе числа и ссылки, которых нет в документах. Возвращает замечания для модели."""
    findings: list[str] = []
    n = kb_store.norm(answer)
    known = kb_store.numbers_in(context) | kb_store.numbers_in(user_text)
    extra = sorted(x for x in kb_store.claim_numbers(answer) if x not in known and x not in _ALWAYS_OK_NUMBERS)
    if extra:
        findings.append("в ответе есть числа, которых нет ни в документах, ни в вопросе: " + ", ".join(extra[:8]))
    # пункты вида «п. 4.2» должны существовать в названном документе компании
    point_re = r"\bп(?:ункт\w*|п)?\.?\s*(\d+\.\d+)(?![\d-])"
    points = set(re.findall(point_re, n)) - set(re.findall(point_re, kb_store.norm(user_text)))
    if points:
        named = [key for key, pattern in _DOC_MARKERS.items() if re.search(pattern, n)]
        law_named = bool(re.search(r"\bнк\b|коап|\bтк\b|-фз|кодекс", n))
        if named and not law_named:
            allowed = set().union(*(kb.doc_points[k] for k in named))
            if "P17" in named:
                allowed |= {"3.2"}
            bad = sorted(points - allowed)
            if bad:
                titles = ", ".join(kb.by_key[k]["short"] for k in named)
                findings.append(f"в названном документе ({titles}) нет пунктов: " + ", ".join(f"п. {b}" for b in bad))
    # статьи законов: либо из базы, либо упомянутые в текстах базы или в вопросе
    cited = set(re.findall(r"\bст(?:ать[а-я]+|\.)\s*(\d+(?:\.\d+)*)", n))
    if cited:
        ctx = kb_store.norm(context + " " + user_text)
        mentioned = set(re.findall(r"\bстать[а-я]+\s+(\d+(?:\.\d+)*)", ctx)) | \
            set(re.findall(r"\bст\.\s*(\d+(?:\.\d+)*)", ctx))
        for group in re.findall(r"\bстать[а-я]+\s+((?:\d+(?:\.\d+)*(?:\s*(?:,|и|-)\s*)?)+)", ctx):
            mentioned |= set(re.findall(r"\d+(?:\.\d+)*", group))
        bad = sorted(cited - kb.articles - mentioned)
        if bad:
            findings.append("в ответе упомянуты статьи, которых нет в базе и в вопросе: " +
                            ", ".join(f"ст. {b}" for b in bad))
    return findings


_ALFA_ANY = re.compile(r"альф\w*", re.I)
_ALFA_SPECIFIC = re.compile(r"трейдинг|трейд\b|трейд[»\"]|п-17|пт-05|мебел|техник|компьютер|комплектующ|ноутбук|"
                            r"оргтехник|стол|кресл|шкаф", re.I)
_USER_INJECTION = re.compile(
    r"игнорируй|забудь (?:все|всё|предыдущ)|ты теперь|новые инструкци|системн\w+ (?:сообщени|промпт|инструкци)|"
    r"режим разработчика|developer mode|system\s*:|покажи (?:свой |системный )?промпт|ignore (?:all|previous)", re.I)


def _hints(current: str, history: list[dict]) -> list[str]:
    hints = []
    user_text = " ".join([current] + [m["content"] for m in history if m["role"] == "user"])
    norm_user = kb_store.norm(user_text).replace("альфа-трейдинг", "трейдинг ").replace("альфа-трейд", "трейд ")
    if _ALFA_ANY.search(kb_store.norm(user_text)) and not _ALFA_SPECIFIC.search(norm_user):
        hints.append(prompts.ALFA_HINT)
        _count("hint_alfa")
    if _USER_INJECTION.search(user_text):
        hints.append(prompts.INJECTION_HINT)
        _count("hint_injection")
    return hints


# ------------------------------------------------------------------ агент
class LegalAgent:
    """База загружается один раз; каждый вопрос обрабатывается независимо, потокобезопасно."""

    def __init__(self, index=None) -> None:  # index оставлен для совместимости со старым вызовом
        self.kb = kb_store.KB()
        self.meta = {"docs": self.kb.docs, "chunks": self.kb.chunks}
        self._system = {True: prompts.system_prompt(True), False: prompts.system_prompt(False)}
        self._catalog = self.kb.catalog()
        if self.kb.emb is None:
            llm.disable_embeddings("нет файла векторов документов")

    # --- контекст -------------------------------------------------------------------------
    def build_context(self, current: str, history: list[dict], use_api: bool = True,
                      extra_keys: list[str] | None = None) -> dict:
        users = [m["content"] for m in history if m["role"] == "user"][-3:]
        search_text = " ".join(users + [current])
        route_text = " ".join([m["content"] for m in history[-6:]] + [current])
        qvec = llm.embed_query(search_text) if (use_api and self.kb.dynamic_keys) else None
        found = self.kb.search(search_text, qvec)
        strong, weak, info = self.kb.route(route_text, found)
        for key in extra_keys or []:
            if key in self.kb.dynamic_keys and key not in strong:
                strong.insert(0, key)
                weak = [k for k in weak if k != key]
        budget = FAST_DYN_BUDGET if FAST_MODE.is_set() else config.DYN_BUDGET_CHARS
        dynamic, used = self.kb.dynamic_context(strong, weak, found["fused"], budget)
        return {"dynamic": dynamic, "used": used, "strong": strong, "weak": weak, "info": info,
                "search_text": search_text, "embedded": found["cos"] is not None}

    def _messages(self, current: str, history: list[dict], ctx: dict, hints: list[str], analysis: bool) -> list[dict]:
        user = prompts.user_message(self._catalog, self.kb.static_text, ctx["dynamic"], history[-8:], current,
                                    hints, analysis)
        return [{"role": "system", "content": self._system[analysis]}, {"role": "user", "content": user}]

    # --- генерация ------------------------------------------------------------------------
    def _generate(self, messages: list[dict], analysis: bool, deadline: float | None) -> tuple[dict, str]:
        """Вызов модели с одним повтором при пустом или оборванном выводе. Возвращает (разбор, сырой текст)."""
        res = llm.chat(messages, max_tokens=llm.effective_max_tokens(), deadline=deadline)
        parsed = parse_output(res.text)
        good = len(parsed["answer"]) >= MIN_ANSWER_CHARS
        if parsed["need"] or (good and not res.truncated):
            return parsed, res.text
        # Вывод пуст, не по формату или оборван лимитом токенов.
        trimmed = _cut_to_sentence(parsed["answer"]) if (res.truncated and parsed["answer"]) else parsed["answer"]
        bigger = llm.effective_max_tokens(retry=True)
        rescue = res.truncated and bigger > res.max_tokens
        if res.truncated and not rescue and len(trimmed) >= 40:
            _count("truncated_trimmed")   # больший лимит недоступен: берём законченные предложения
            parsed["answer"] = trimmed
            return parsed, res.text
        _count("regenerated_truncated" if res.truncated else "regenerated_empty_or_bad_format")
        retry_messages = messages
        if analysis:  # второй раз просим только ответ - так он точно поместится
            retry_messages = [dict(messages[0], content=self._system[False]), dict(messages[1])]
            retry_messages[1]["content"] = messages[1]["content"].rsplit("Ответь строго в формате", 1)[0] + \
                "Верни только <answer>…</answer>."
        try:
            res2 = llm.chat(retry_messages, max_tokens=bigger if rescue else llm.effective_max_tokens(),
                            deadline=deadline, attempts=2)
        except llm.LLMError:
            if len(trimmed) >= 40:
                _count("truncated_trimmed")
                parsed["answer"] = trimmed
                return parsed, res.text
            raise
        parsed2 = parse_output(res2.text)
        if res2.truncated and parsed2["answer"]:
            parsed2["answer"] = _cut_to_sentence(parsed2["answer"]) or parsed2["answer"]
        if len(parsed2["answer"]) >= MIN_ANSWER_CHARS:
            if rescue and not res2.truncated:
                llm.note_truncation_recovered()
            parsed2["type"] = parsed2["type"] or parsed["type"]
            return parsed2, res2.text
        if len(trimmed) >= MIN_ANSWER_CHARS:
            _count("truncated_trimmed")
            parsed["answer"] = trimmed
            return parsed, res.text
        raise llm.LLMError("модель дважды вернула пустой ответ")

    def answer_ex(self, question: str, history=None, deadline: float | None = None) -> tuple[dict, dict]:
        """Ответ и служебные сведения (для логов и повторного прохода)."""
        t0 = time.monotonic()
        info: dict = {"fallback": False, "calls": 0, "error": ""}
        question = (question or "").strip()
        current, embedded = parse_history(question)
        hist = (_normalise_history(history) + embedded)[-10:]
        _count("questions")
        if not current:
            _count("fallback_empty_question")
            text = ("Вопрос не распознан. Я отвечаю на юридические вопросы по базе знаний ООО «Меридиан» "
                    "(договоры, регламенты, статьи законов) — сформулируйте, пожалуйста, вопрос.")
            return {"answer": text, "status": "refused_offtopic", "sources": []}, info
        analysis = config.USE_ANALYSIS and llm.effective_max_tokens() >= ANALYSIS_MIN_TOKENS
        fast = FAST_MODE.is_set()
        ctx: dict | None = None
        try:
            hints = _hints(current, hist)
            ctx = self.build_context(current, hist)
            if ctx["used"]:
                _count("dynamic_docs_added")
            if ctx["embedded"]:
                _count("search_with_embeddings")
            info["docs"] = ctx["used"]
            messages = self._messages(current, hist, ctx, hints, analysis)
            parsed, raw = self._generate(messages, analysis, deadline)
            info["calls"] += 1
            # модель попросила показать статью, которой не было в контексте
            wanted = [k for k in parsed["need"] if k in self.kb.dynamic_keys and k not in ctx["used"]]
            if parsed["need"] and (not parsed["answer"] or wanted):
                if wanted and config.ALLOW_NEED and not fast:
                    _count("need_requests")
                    ctx2 = self.build_context(current, hist, extra_keys=wanted)
                    ctx["dynamic"], ctx["used"] = ctx2["dynamic"], ctx2["used"]
                    messages = self._messages(current, hist, ctx, hints, analysis)
                    parsed, raw = self._generate(messages, analysis, deadline)
                    info["calls"] += 1
                if not parsed["answer"]:  # снова без ответа: просим ответить по тому, что есть
                    _count("need_unresolved")
                    analysis = False
                    messages = self._messages(current, hist, ctx, hints, False)
                    parsed, raw = self._generate(messages, False, deadline)
                    info["calls"] += 1
            answer = parsed["answer"]
            context_text = self.kb.static_text + "\n" + ctx["dynamic"]
            user_text = " ".join([current] + [m["content"] for m in hist])
            status = guess_status(answer, parsed["type"])
            # самопроверка
            time_left = (deadline - time.monotonic()) if deadline is not None else 1e9
            if config.SELF_CHECK in {"targeted", "full"} and not fast and time_left > 25:
                findings = check_answer(self.kb, answer, context_text, user_text) \
                    if status in {"answered", "partial", "refused_illegal"} else []
                followup = None
                if findings:
                    _count("check_flagged")
                    followup = prompts.repair_message(findings, analysis)
                elif config.SELF_CHECK == "full":
                    followup = prompts.VERIFY_MESSAGE
                if followup:
                    try:
                        # Прошлый ответ передаём внутри сообщения пользователя, а не отдельной репликой
                        # ассистента: некоторые прокси рассуждающих моделей отвергают такие реплики.
                        check_messages = [messages[0], dict(messages[1])]
                        check_messages[1]["content"] = (
                            messages[1]["content"] + "\n\nТвой предыдущий ответ на эту реплику:\n<previous_answer>\n"
                            + answer + "\n</previous_answer>\n\n" + followup)
                        res = llm.chat(check_messages, max_tokens=llm.effective_max_tokens(), deadline=deadline,
                                       attempts=2)
                        info["calls"] += 1
                        fixed = parse_output(res.text)
                        candidate = _cut_to_sentence(fixed["answer"]) if res.truncated else fixed["answer"]
                        if len(candidate) >= max(MIN_ANSWER_CHARS, 30) and not fixed["need"]:
                            if candidate != answer:
                                _count("check_changed_answer")
                            answer = candidate
                            status = guess_status(answer, fixed["type"] or parsed["type"])
                    except llm.LLMError as exc:  # проверка не удалась - оставляем первый ответ
                        _count("check_call_failed")
                        info["error"] = str(exc)[:200]
            elif config.SELF_CHECK != "off":
                _count("check_skipped_no_time")
        except Exception as exc:  # noqa: BLE001 - любой сбой модели: отвечаем дословным положением базы
            info.update(fallback=True, error=f"{type(exc).__name__}: {exc}"[:300])
            _count("fallback_extractive")
            if not isinstance(exc, llm.LLMError):
                print(f"[agent] непредвиденная ошибка: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            users = [m["content"] for m in hist if m["role"] == "user"][-3:]
            search_text = ctx["search_text"] if ctx else " ".join(users + [current])
            return {"answer": self.kb.extractive(search_text), "status": "partial", "sources": []}, info
        if len(answer.split()) > MAX_ANSWER_WORDS:
            answer = _cut_to_sentence(answer, MAX_ANSWER_WORDS)
            _count("too_long_trimmed")
        answer = answer[:MAX_ANSWER_CHARS]
        if status not in STATUSES:
            status = "answered"
        _count(status)
        sources = []
        if status in {"answered", "partial"}:
            try:
                sources = self.kb.sources(answer)
            except Exception:  # noqa: BLE001 - источники нужны только веб-интерфейсу
                sources = []
        info["seconds"] = time.monotonic() - t0
        return {"answer": answer, "status": status, "sources": sources}, info

    def answer(self, question: str, history=None) -> dict:
        return self.answer_ex(question, history)[0]

    def fallback_answer(self, question: str, history=None) -> str:
        """Быстрый ответ без модели: дословное положение базы со ссылкой."""
        try:
            current, embedded = parse_history((question or "").strip())
            hist = _normalise_history(history) + embedded
            users = [m["content"] for m in hist if m["role"] == "user"][-3:]
            return self.kb.extractive(" ".join(users + [current]))
        except Exception as exc:  # noqa: BLE001
            print(f"[fallback] {type(exc).__name__}: {exc}", file=sys.stderr)
            return ("В базе знаний ООО «Меридиан» не нашлось положений по этому вопросу. По памяти отвечать "
                    "не буду, так как нормы могли измениться; уточните вопрос или обратитесь к юристу.")

    def ask(self, question: str) -> dict:
        return self.answer(question, history=None)


_DEFAULT_AGENT: LegalAgent | None = None
_AGENT_LOCK = threading.Lock()


def get_agent() -> LegalAgent:
    global _DEFAULT_AGENT
    with _AGENT_LOCK:
        if _DEFAULT_AGENT is None:
            _DEFAULT_AGENT = LegalAgent()
        return _DEFAULT_AGENT


def answer(question: str, history=None) -> dict:
    return get_agent().answer(question, history)
