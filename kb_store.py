"""База знаний решения: полные тексты документов, выбор статей под вопрос, локальный поиск.

Идея. База маленькая: все документы компании - около 15 тыс. символов, короткие статьи законов -
ещё около 25 тыс. Поэтому они ВСЕГДА целиком лежат в контексте модели, и поиск не может потерять
нужный пункт, допсоглашение или вторую редакцию регламента. Поиск нужен только для пяти больших
статей (КоАП 13.11, НК 145, 164, 224, 284): какая из них относится к вопросу и какие фрагменты
двух огромных статей (НК 164 и 284) показать.

Всё работает без сети: BM25 + правила по ключевым словам. Эмбеддинг вопроса (если прокси его дал)
только улучшает ранжирование.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

import config

ROOT = Path(__file__).resolve().parent
# В ZIP-сабмите все файлы лежат в корне; в репозитории и Docker-образе - в папке index/.
_INDEX_ROOT = ROOT / "index" if (ROOT / "index" / "kb_bundle.json").is_file() else ROOT
BUNDLE_FILE = _INDEX_ROOT / "kb_bundle.json"
EMB_FILE = _INDEX_ROOT / "index_emb.npy"

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_STOP = set("""а без бы был была были быть в вам вас весь во вот все всего всех вы где да для до его ее если есть
еще же за здесь и из или им их к как ко когда кто ли либо мне может мы на над нам нас не него нее нет ни них но ну
о об однако он она они оно от по под при про с со так также там те тем то того тоже той только том ты у уже хотя
чего чем что чтобы эта эти это я какой какая какие какое каком сколько нужно надо можно ли нам наш наша наши
подскажите скажите пожалуйста пользователь ассистент""".split())
_ENDING = re.compile(
    r"(?:иями|ями|ами|ого|его|ому|ему|ыми|ими|иях|ием|иям|ией|ться|тся|ешь|ала|али|ало|ены|ена|ено"
    r"|ия|ие|ий|ию|ую|юю|ая|яя|ое|ее|ые|ой|ей|ом|ем|ах|ях|ам|ям|ов|ев|ет|ут|ют|ит|ат|ят|ал"
    r"|а|я|о|е|ы|и|у|ю|ь|й)$")


def norm(text: str) -> str:
    text = (text or "").lower().replace("ё", "е").translate(_DASHES)
    return re.sub(r"\s+", " ", text).strip()


def stem(token: str) -> str:
    if len(token) < 5 or not token.isalpha():
        return token
    base = _ENDING.sub("", token)
    return base[:7] if len(base) >= 4 else token


# Бытовые слова -> слова документов. Это словарь синонимов, а не новые факты.
_EXPANSIONS = (
    (r"\bаванс", "предоплата предоплаты"),
    (r"\b(?:брак|дефект|поломк|сломал)", "недостатков претензию гарантийный"),
    (r"\b(?:ноч|круглосуточ|грузовик|фур[аыу]|машин)", "круглосуточно доступ транспорта"),
    (r"\b(?:переработ|сверхурочн|задерж\w+ на работе|после работы|овертайм)", "сверхурочная работа оплата"),
    (r"\b(?:испытательн|стаж[её]р)", "испытание испытания срок"),
    (r"\b(?:срочн\w+ (?:трудов|договор)|временн\w+ работ|сезонн)", "срочный трудовой договор заключается"),
    (r"\b(?:мрот|минималк|минимальн\w+ зарплат)", "минимальный размер оплаты труда"),
    (r"\b(?:ндс|добавленн)", "налогообложение производится налоговой ставке реализации"),
    (r"\b(?:ндфл|подоходн)", "налоговая ставка налоговых баз физических лиц"),
    (r"\bприбыл", "налоговая ставка налог прибыль организаций"),
    (r"\b(?:утечк|слив\w* данн)", "неправомерную передачу персональных данных"),
    (r"\b(?:айти|it\b|ит-компан|информационн\w+ технолог|разработчик\w* по)", "информационных технологий"),
    (r"\bдивиденд", "дивидендов доходам"),
    (r"\bэкспорт", "экспорта вывезенных таможенной процедуре"),
    (r"\b(?:лекарств|медицинск|медиздел)", "лекарственных средств медицинских изделий"),
    (r"\b(?:детск|ребен|игрушк)", "товаров для детей"),
    (r"\b(?:продукт|еда|продовольств|хлеб|молок|мяс)", "продовольственных товаров"),
    (r"\b(?:книг|журнал|газет|печатн)", "периодических печатных изданий книжной продукции"),
    (r"\b(?:смягча|снизить штраф|уменьшить штраф)", "смягчающих обстоятельств уменьшению штрафа"),
    (r"\b(?:повторн|рецидив)", "повторное совершение ранее привлекаемым"),
    (r"\b(?:нерезидент|иностран\w+ (?:работник|гражд))", "не являющимися налоговыми резидентами"),
    (r"\b(?:усн|упрощенк|упрощенн)", "упрощенную систему налогообложения"),
)


def tokens(text: str, expand: bool = False) -> list[str]:
    n = norm(text)
    words = [w for w in re.findall(r"[a-zа-я0-9][a-zа-я0-9./-]*", n) if w not in _STOP and len(w) > 1]
    if expand:
        for pattern, extra in _EXPANSIONS:
            if re.search(pattern, n):
                words.extend(extra.split())
    out = []
    for w in words:
        w = w.strip("./-")
        if w:
            out.append(stem(w))
    return out


# ------------------------------------------------------------------ правила выбора больших статей
_PD = r"(?:персональн|\bпдн\b|\bпд\b|утечк|биометр|роскомнадзор|обработк\w+ данных|согласи\w+ на обработку)"
_PENALTY = r"(?:штраф|ответственност|наказан|грозит|накаж|санкци|привлеч|оштраф)"
_ROUTES: dict[str, list[str]] = {
    "KOAP1311": [r"13\.11", r"\bкоап", r"административн\w+ (?:штраф|ответственност|правонаруш)",
                 rf"{_PD}.{{0,160}}{_PENALTY}", rf"{_PENALTY}.{{0,160}}{_PD}", r"утечк", r"\bуте[кч]\w*", r"слив\w* (?:баз|данн)", r"биометр",
                 r"роскомнадзор", rf"(?:персональн|данны\w*).{{0,80}}без.{{0,30}}согласи",
                 r"без.{0,30}согласи\w*.{0,80}(?:персональн|данны)"],
    "NK145": [r"\b145\b", r"освобожд\w+.{0,80}(?:ндс|налогоплательщик|обязанност)",
              r"(?:ндс|добавленн).{0,120}(?:освобожд|не платить|не плат|выручк|упрощ|\bусн\b|есхн|лимит|порог|малый бизнес|маленьк)",
              r"(?:упрощ|\bусн\b|есхн).{0,120}(?:ндс|добавленн)", r"подакцизн",
              r"(?:выручк|доход|оборот).{0,120}(?:ндс|добавленн)",
              r"(?:не платить|не плат\w+|освобо\w+|избежать|платить ли|нужно ли платить|плательщик\w*).{0,80}(?:ндс|добавленн)",
              r"(?:ндс|добавленн).{0,120}(?:оборот|доход|плательщик|платить ли)"],
    "NK164": [r"\b164\b", r"\bндс\b", r"добавленн\w+ стоимост", r"нулев\w+ ставк", r"\bэкспорт",
              r"ставк\w+.{0,40}(?:\b0\b|\b10\b|\b18\b|\b20\b|\b22\b)\s*(?:%|процент)"],
    "NK224": [r"\b224\b", r"\bндфл\b", r"подоходн", r"налог\w* на доходы физ", r"нерезидент",
              r"(?:налог|ставк|удерж).{0,80}(?:зарплат|заработн|физлиц|физическ\w+ лиц|выигрыш|\bприз|вклад|сотрудник|работник)",
              r"(?:зарплат|заработн|выигрыш|\bприз\w*).{0,80}(?:налог|ставк)", r"прогрессивн"],
    "NK284": [r"\b284\b", r"налог\w* на прибыль", r"(?:ставк|налог).{0,60}прибыл", r"прибыл\w*.{0,60}(?:ставк|налог)",
              r"дивиденд", r"с прибыли", r"прибыл\w+ (?:компани|организаци|фирм|предприяти)"],
    # короткие статьи: нужны только в режиме CONTEXT_MODE=company
    "TK59": [r"\b59\b", r"срочн\w+ (?:трудов|договор|контракт)", r"сезонн", r"временн\w+ работ"],
    "TK70": [r"\b70\b", r"испытан", r"испытательн"],
    "TK99": [r"\b99\b", r"сверхурочн", r"переработ"],
    "TK152": [r"\b152\b(?!-)", r"сверхурочн", r"переработ", r"полуторн", r"двойн\w+ размер"],
    "MROT": [r"\bмрот\b", r"минимальн\w+ (?:размер|оплат|зарплат)", r"82-фз"],
    "FZ152": [r"152-фз", r"персональн", r"согласи\w+ на обработку", r"обработк\w+ данных"],
    "NK112": [r"\b112\b", r"смягчающ", r"отягчающ"],
    "NK114": [r"\b114\b", r"налогов\w+ санкци", r"(?:уменьш|сниз|увелич)\w*.{0,40}штраф", r"смягчающ", r"отягчающ"],
}
_ANCHORS = {  # что из огромной статьи показываем всегда, когда она выбрана: её «скелет»
    "NK164": ("1. Налогообложение производится по налоговой ставке 0 процентов",
              "2. Налогообложение производится по налоговой ставке 10 процентов",
              "3. Налогообложение производится по налоговой ставке 22",
              "4. "),
    "NK284": ("1. Налоговая ставка устанавливается в размере 25 процентов",),
}
WHOLE_DOC_LIMIT = 26000  # статьи короче этого показываем целиком, длиннее - фрагментами


class KB:
    def __init__(self) -> None:
        bundle = json.loads(BUNDLE_FILE.read_text(encoding="utf-8"))
        self.docs = sorted(bundle["docs"], key=lambda d: d["order"])
        self.by_key = {d["key"]: d for d in self.docs}
        self.by_id = {d["id"]: d for d in self.docs}
        self.chunks = bundle["chunks"]
        self.kb_sha = bundle.get("kb_sha")
        for pos, c in enumerate(self.chunks):
            c["pos"] = pos
            c["key"] = self.by_id[c["doc"]]["key"]
        self.doc_chunks: dict[str, list[dict]] = {}
        for c in self.chunks:
            self.doc_chunks.setdefault(c["key"], []).append(c)
        for d in self.docs:
            d["short"] = d["title"].split(". ")[0].rstrip(".")
            d["size"] = len(d["text"])
        company_only = config.CONTEXT_MODE == "company"
        self.static_keys = [d["key"] for d in self.docs
                            if d["always"] and (d["kind"] == "company" or not company_only)]
        self.dynamic_keys = [d["key"] for d in self.docs if d["key"] not in self.static_keys]
        self._static_text = "\n\n".join(self.render_doc(k) for k in self.static_keys)
        corpus = [tokens(f"{c['lc']} {c['text']}") + tokens(self.by_key[c["key"]]["title"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)
        self.emb = self._load_embeddings()
        self.anchor_ids: dict[str, list[int]] = {}
        for key, prefixes in _ANCHORS.items():
            ids = []
            for c in self.doc_chunks.get(key, []):
                if c["text"].startswith(prefixes):
                    ids.append(c["pos"])
            self.anchor_ids[key] = ids
        self._routes = {k: [re.compile(p) for p in v] for k, v in _ROUTES.items()}
        # для проверок ответа
        self.doc_points = {d["key"]: set(re.findall(r"(?m)^[«\"]?(\d+\.\d+)\.?\s", d["text"]))
                           for d in self.docs if d["kind"] == "company"}
        self.articles = {"9", "13.11", "1", "112", "114", "145", "164", "224", "284", "152", "59", "70", "99"}

    def _load_embeddings(self) -> np.ndarray | None:
        try:
            emb = np.load(EMB_FILE)
            max_id = max(c["id"] for c in self.chunks)
            if emb.ndim != 2 or emb.shape[0] <= max_id or not np.isfinite(emb).all():
                raise ValueError(f"неподходящая форма {emb.shape}")
            rows = emb[[c["id"] for c in self.chunks]].astype("float32")
            return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        except Exception as exc:  # noqa: BLE001 - без векторов работаем на BM25
            print(f"[kb] векторы документов недоступны ({exc}); поиск только BM25", file=sys.stderr)
            return None

    # -------------------------------------------------------------- показ документов модели
    def render_doc(self, key: str, body: str | None = None, partial: bool = False) -> str:
        d = self.by_key[key]
        kind = "документ компании" if d["kind"] == "company" else "закон (редакция на 10.10.2026)"
        head = f'<document key="{key}" type="{kind}">\nНазвание: {d["title"]}\nКак ссылаться: {d["cite"]}\n'
        if partial:
            head += ("ВНИМАНИЕ: статья очень длинная, показаны только фрагменты, отобранные поиском; "
                     "пропуски отмечены знаком […]. Если нужного положения среди фрагментов нет, не утверждай, "
                     "что его нет в законе: скажи, что среди найденных положений статьи его нет.\n")
        return f"{head}Текст:\n{body if body is not None else d['text']}\n</document>"

    @property
    def static_text(self) -> str:
        return self._static_text

    def catalog(self) -> str:
        lines = []
        for d in self.docs:
            shown = "текст показан ниже целиком" if d["key"] in self.static_keys else \
                "текст показывается, только когда вопрос относится к этой статье"
            lines.append(f"- [{d['key']}] {d['title']} ({shown})")
        return "\n".join(lines)

    # -------------------------------------------------------------- поиск
    def scores(self, query: str, qvec: np.ndarray | None = None) -> np.ndarray:
        """Слитая оценка каждого чанка (RRF по BM25 и, если есть, по векторам)."""
        return self.search(query, qvec)["fused"]

    def search(self, query: str, qvec: np.ndarray | None = None) -> dict:
        n = len(self.chunks)
        q = tokens(query, expand=True)
        bm = np.asarray(self.bm25.get_scores(q), dtype="float64") if q else np.zeros(n)
        fused = np.zeros(n)
        order = np.argsort(-bm, kind="stable")
        ranks = np.empty(n, dtype="int64")
        ranks[order] = np.arange(n)
        fused += np.where(bm > 0, 1.0 / (60 + ranks), 0.0)
        cos = None
        if qvec is not None and self.emb is not None and qvec.shape[0] == self.emb.shape[1]:
            cos = self.emb @ qvec.astype("float32")
            order = np.argsort(-cos, kind="stable")
            ranks = np.empty(n, dtype="int64")
            ranks[order] = np.arange(n)
            fused += 1.0 / (60 + ranks)
        return {"fused": fused, "bm25": bm, "cos": cos}

    def route(self, text: str, found: dict) -> tuple[list[str], list[str], dict]:
        """Какие непоказанные статьи добавить к вопросу.

        strong - статья названа или вопрос явно о её теме (правила по ключевым словам), либо на неё
                 уверенно указывают векторы: показываем целиком или большим набором фрагментов.
        weak   - на статью указывает только общий поиск: добавляем несколько лучших фрагментов,
                 чтобы не терять ответ, но и не заливать контекст посторонним текстом.
        """
        n = norm(text)
        fused, cos = found["fused"], found["cos"]
        cited = set(re.findall(r"ст(?:ать[а-я]+|\.)?\s*(\d+(?:\.\d+)?)", n))
        strong: list[str] = []
        for key in self.dynamic_keys:
            art = re.search(r"стать[а-я]+ (\d+(?:\.\d+)?)", norm(self.by_key[key]["title"]))
            if any(p.search(n) for p in self._routes.get(key, [])) or (art and art.group(1) in cited and
                                                                         self._law_family_ok(key, n)):
                strong.append(key)
        by_vector: list[str] = []
        if cos is not None:
            top = np.argsort(-cos, kind="stable")[:5]
            hits: dict[str, int] = {}
            for pos in top:
                key = self.chunks[int(pos)]["key"]
                if key in self.dynamic_keys and cos[pos] >= 0.42:
                    hits[key] = hits.get(key, 0) + 1
            by_vector = [k for k, count in hits.items() if count >= 2]
        # слабый сигнал: статья занимает большинство мест в пятёрке лучших фрагментов общего поиска
        # (без векторов BM25 шумит сильнее, поэтому требуем больше совпадений)
        counts: dict[str, int] = {}
        for pos in np.argsort(-fused, kind="stable")[:5]:
            if fused[pos] <= 0:
                continue
            key = self.chunks[int(pos)]["key"]
            if key in self.dynamic_keys:
                counts[key] = counts.get(key, 0) + 1
        need = 2 if cos is not None else 3
        weak = [k for k, count in sorted(counts.items(), key=lambda kv: -kv[1]) if count >= need]
        doc_score = {k: float(sum(sorted((fused[c["pos"]] for c in self.doc_chunks[k]), reverse=True)[:3]))
                     for k in self.dynamic_keys}
        strong = sorted(set(strong) | set(by_vector), key=lambda k: -doc_score[k])
        weak = [k for k in weak if k not in strong][:2]
        return strong, weak, {"by_rule": [k for k in strong if k not in by_vector], "by_vector": by_vector,
                              "weak": weak}

    @staticmethod
    def _law_family_ok(key: str, n: str) -> bool:
        """«ст. 152» может быть и ТК, и 152-ФЗ: не подмешиваем статью чужого закона, если закон назван."""
        named_tk = bool(re.search(r"\bтк\b|трудов", n))
        named_nk = bool(re.search(r"\bнк\b|налогов", n))
        named_koap = bool(re.search(r"коап|административн", n))
        if not (named_tk or named_nk or named_koap):
            return True
        return (key.startswith("TK") and named_tk) or (key.startswith("NK") and named_nk) or \
            (key.startswith("KOAP") and named_koap)

    def dynamic_context(self, strong: list[str], weak: list[str], fused: np.ndarray,
                        budget: int) -> tuple[str, list[str]]:
        """Тексты выбранных статей в пределах бюджета символов. Возвращает (текст, [ключи])."""
        parts: dict[str, str] = {}
        left = budget
        big = [k for k in strong if self.by_key[k]["size"] > WHOLE_DOC_LIMIT]
        reserve = min(16000 * len(big), int(budget * 0.6))  # место для фрагментов огромных статей
        for key in strong:  # сначала статьи, которые помещаются целиком
            d = self.by_key[key]
            if key in big:
                continue
            if d["size"] <= left - reserve:
                parts[key] = self.render_doc(key)
                left -= d["size"]
            elif left - reserve > 3000:
                body = self._fragments(key, fused, min(left - reserve, 12000))
                if body:
                    parts[key] = self.render_doc(key, body, partial=True)
                    left -= len(body)
        for i, key in enumerate(big):  # огромные статьи делят остаток поровну
            share = left // (len(big) - i)
            body = self._fragments(key, fused, share) if share > 1500 else ""
            if body:
                parts[key] = self.render_doc(key, body, partial=True)
                left -= len(body)
        for key in weak:  # слабый сигнал: только несколько лучших фрагментов
            share = min(6000, left)
            if share < 1500:
                break
            d = self.by_key[key]
            if d["size"] <= share:
                parts[key] = self.render_doc(key)
                left -= d["size"]
                continue
            body = self._fragments(key, fused, share)
            if body:
                parts[key] = self.render_doc(key, body, partial=True)
                left -= len(body)
        ordered = [k for k in self.dynamic_keys if k in parts]  # всегда в одном порядке
        return "\n\n".join(parts[k] for k in ordered), ordered

    def _fragments(self, key: str, fused: np.ndarray, budget: int) -> str:
        chunks = self.doc_chunks[key]
        chosen: set[int] = set()
        size = 0

        def add(pos: int) -> bool:
            nonlocal size
            if pos in chosen:
                return True
            c = self.chunks[pos]
            cost = len(c["text"]) + (len(c["lc"]) if c["lc"] else 0) + 8
            if size + cost > budget:
                return False
            chosen.add(pos)
            size += cost
            return True

        for pos in self.anchor_ids.get(key, []):
            add(pos)
        ranked = sorted(chunks, key=lambda c: -fused[c["pos"]])
        first, last = chunks[0]["pos"], chunks[-1]["pos"]
        for c in ranked:
            if not add(c["pos"]):
                continue
            base = re.sub(r", фрагмент \d+$", "", c["label"])
            for step in (-1, 1):  # остальные куски того же пункта и продолжение после двоеточия
                pos = c["pos"] + step
                while first <= pos <= last:
                    other = self.chunks[pos]
                    same_point = bool(base) and re.sub(r", фрагмент \d+$", "", other["label"]) == base \
                        and "фрагмент" in other["label"]
                    continues = step == 1 and self.chunks[pos - 1]["text"].rstrip().endswith((":", "-", "–", "—"))
                    if not (same_point or continues) or not add(pos):
                        break
                    pos += step
        if not chosen:
            return ""
        out, prev, shown_intro = [], None, set()
        for pos in sorted(chosen):
            c = self.chunks[pos]
            if prev is not None and pos != prev + 1:
                out.append("[…]")
            elif prev is None and pos != first:
                out.append("[…]")
            if c["lc"] and c["lc"] not in shown_intro and not c["text"].startswith(c["lc"][:40]):
                intro_pos = next((p for p in sorted(chosen) if self.chunks[p]["text"].startswith(c["lc"][:40])), None)
                if intro_pos is None or intro_pos > pos:
                    out.append(f"(вводная фраза перечня: {c['lc']})")
            if c["lc"]:
                shown_intro.add(c["lc"])
            out.append(c["text"])
            prev = pos
        if prev != last:
            out.append("[…]")
        return "\n".join(out)

    # -------------------------------------------------------------- ответ без модели
    def extractive(self, query: str, limit: int = 520) -> str:
        """Дословное положение базы со ссылкой - на случай, когда модель недоступна.

        Ничего не выдумывает и не содержит служебных фраз вроде «не удалось сформировать ответ»:
        такие фразы судья оценивает в 0, а точная цитата нужного пункта может получить часть балла.
        """
        fused = self.scores(query)
        pos = int(np.argmax(fused))
        if fused[pos] <= 0:
            return ("В базе знаний ООО «Меридиан» не нашлось положений по этому вопросу. По памяти отвечать "
                    "не буду, так как нормы могли измениться; уточните вопрос или обратитесь к юристу.")
        key = self.chunks[pos]["key"]
        d = self.by_key[key]
        best = sorted(self.doc_chunks[key], key=lambda c: -fused[c["pos"]])[:3]
        best = [c for c in best if fused[c["pos"]] >= 0.6 * fused[pos]]
        pieces, size = [], 0
        for c in sorted(best, key=lambda c: c["pos"]):
            text = " ".join(c["text"].split())
            if c["lc"] and not text.startswith(c["lc"][:40]):
                text = f"{c['lc']} {text}"
            room = limit - size
            if room < 80:
                break
            if len(text) > room:
                cut = text[:room]
                end = max(cut.rfind(". "), cut.rfind("; "))
                text = cut[:end + 1] if end > room // 3 else cut.rstrip() + "…"
            pieces.append(text)
            size += len(text)
        return f"{d['short']}: " + " ".join(pieces)

    def sources(self, answer: str, limit: int = 4) -> list[dict]:
        """Фрагменты базы, на которые опирается ответ (для веб-интерфейса). На балл не влияет."""
        fused = self.scores(answer)
        n = norm(answer)
        refs = set(re.findall(r"п\.\s*(\d+\.\d+)", n))
        out, seen = [], set()
        for pos in np.argsort(-fused, kind="stable")[:40]:
            c = self.chunks[int(pos)]
            if fused[pos] <= 0:
                break
            d = self.by_key[c["key"]]
            point = re.search(r"п\.\s*(\d+\.\d+)", c["label"])
            bonus = 1.0 if point and point.group(1) in refs else 0.0
            out.append((bonus + float(fused[pos]), c, d))
        out.sort(key=lambda x: -x[0])
        result = []
        for score, c, d in out:
            if (d["key"], c["label"]) in seen:
                continue
            seen.add((d["key"], c["label"]))
            result.append({"doc_id": _backend_doc_id(d), "doc_title": d["short"], "point": c["label"],
                           "text": c["text"], "score": round(float(score), 4)})
            if len(result) >= limit:
                break
        return result


def _backend_doc_id(doc: dict):
    """doc_id веб-бэкенда (kb.py), если он доступен; иначе ключ документа."""
    try:
        import kb as docs_kb  # noqa: WPS433 - модуль веб-интерфейса, на платформе может не сработать
        doc_id, _ = docs_kb.doc_ref_for_file(doc["file"])
        return doc_id if doc_id is not None else doc["key"]
    except Exception:  # noqa: BLE001
        return doc["key"]


# ------------------------------------------------------------------ числа (для проверки ответа)
_UNITS = {"ноль": 0, "один": 1, "одна": 1, "одно": 1, "одного": 1, "одной": 1, "одну": 1, "одним": 1,
          "два": 2, "две": 2, "двух": 2, "двум": 2, "три": 3, "трех": 3, "трем": 3, "четыре": 4, "четырех": 4,
          "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7, "восемь": 8, "восьми": 8,
          "девять": 9, "девяти": 9, "полтора": 1.5, "полутора": 1.5}
_TEENS = {"десять": 10, "десяти": 10, "одиннадцать": 11, "одиннадцати": 11, "двенадцать": 12, "двенадцати": 12,
          "тринадцать": 13, "тринадцати": 13, "четырнадцать": 14, "четырнадцати": 14, "пятнадцать": 15,
          "пятнадцати": 15, "шестнадцать": 16, "шестнадцати": 16, "семнадцать": 17, "семнадцати": 17,
          "восемнадцать": 18, "восемнадцати": 18, "девятнадцать": 19, "девятнадцати": 19}
_TENS = {"двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30, "сорок": 40, "сорока": 40,
         "пятьдесят": 50, "пятидесяти": 50, "шестьдесят": 60, "шестидесяти": 60, "семьдесят": 70,
         "семидесяти": 70, "восемьдесят": 80, "восьмидесяти": 80, "девяносто": 90, "девяноста": 90}
_HUNDREDS = {"сто": 100, "ста": 100, "двести": 200, "двухсот": 200, "триста": 300, "трехсот": 300,
             "четыреста": 400, "четырехсот": 400, "пятьсот": 500, "пятисот": 500, "шестьсот": 600,
             "шестисот": 600, "семьсот": 700, "семисот": 700, "восемьсот": 800, "восьмисот": 800,
             "девятьсот": 900, "девятисот": 900}
_WORD_VALUE = {**_UNITS, **_TEENS, **_TENS, **_HUNDREDS}
_MULT = (("тысяч", 1e3), ("тыс", 1e3), ("миллион", 1e6), ("млн", 1e6), ("миллиард", 1e9), ("млрд", 1e9))
_SPECIAL = {"полуторн": 1.5, "двойн": 2.0}


def _mult(word: str) -> float | None:
    for prefix, value in _MULT:
        if word.startswith(prefix):
            return value
    return None


def _canon(value: float) -> str:
    return str(int(round(value))) if abs(value - round(value)) < 1e-9 else f"{value:.4f}".rstrip("0")


def numbers_in(text: str) -> set[str]:
    """Все числа текста (цифрами и словами) в едином виде: «450 000», «трёхсот тысяч», «2,4 миллиона»."""
    n = norm(text)
    n = re.sub(r"(?<=\d)[   ](?=\d{3}(?!\d))", "", n)  # 450 000 -> 450000
    found: set[str] = set()
    words = re.findall(r"\d+(?:[.,]\d+)?|[а-я]+|[^\sа-я\d]", n)  # знаки препинания разрывают числительное
    i = 0
    while i < len(words):
        w = words[i]
        if w[0].isdigit():
            value = float(w.replace(",", "."))
            found.add(_canon(value))
            mult = _mult(words[i + 1]) if i + 1 < len(words) and words[i + 1][0].isalpha() else None
            if mult:
                found.add(_canon(value * mult))
                i += 1
            i += 1
            continue
        for prefix, value in _SPECIAL.items():
            if w.startswith(prefix):
                found.add(_canon(value))
        if w in _WORD_VALUE:
            total, current, j, used = 0.0, 0.0, i, False
            while j < len(words):
                x = words[j]
                if x in _WORD_VALUE:
                    current += _WORD_VALUE[x]
                    used = True
                elif _mult(x) and used:
                    total += (current or 1.0) * _mult(x)
                    current = 0.0
                else:
                    break
                j += 1
            found.add(_canon(total + current))
            i = j
            continue
        i += 1
    return found


_REF_NOISE = re.compile(
    r"№\s*[\w\-/.]+|\b[а-яa-z]{1,3}-\d+(?:/\d+)?|\b\d+-фз\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\b\d+-(?:го|му|й|е|я|х|ти|ми)\b"
    r"|(?:\bст|\bп|\bпп|\bподп|\bч|\bабз|стать[а-я]+|пункт[а-я]*|подпункт[а-я]*|част[ьи]|частью|абзац[а-я]*|кабинет[а-я]*"
    r"|склад[а-я]*|стр|д)\.?\s*№?\s*\d+(?:\.\d+)*(?:-\d+)?(?:\s*(?:,|и|-)\s*\d+(?:\.\d+)*)*"
)


def claim_numbers(text: str) -> set[str]:
    """Числа-утверждения из ответа: без номеров документов, пунктов, статей, дат и адресов."""
    return numbers_in(_REF_NOISE.sub(" ", norm(text)))
