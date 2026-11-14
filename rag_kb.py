"""База знаний: структурные чанки с датами, точными локаторами и версионированным индексом."""
import hashlib
import json
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import yaml

import config

MAX_CHARS = 1400          # максимум символов в чанке, кроме длинных неделимых предложений
HEAD_CHARS = 350          # «начало документа» для справки модели (редакция, даты, стороны)
ALIAS_SCAN_CHARS = 4000   # где искать номера договоров и названия организаций
INDEX_SCHEMA_VERSION = 6
_INDEX_DIR = config.ROOT / "index"
# В репозитории индекс хранится в index/; build_submission.py кладёт его файлы в корень ZIP.
_INDEX_ROOT = _INDEX_DIR if _INDEX_DIR.is_dir() else config.ROOT
META_FILE = _INDEX_ROOT / "index_meta.json"
EMB_FILE = _INDEX_ROOT / "index_emb.npy"
REGISTRY_FILE = config.ROOT / "document_registry.yaml"
_LOCAL_MODEL = None

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def norm(s: str) -> str:
    """Для сопоставления: нижний регистр, ё→е, единые дефисы и пробелы."""
    s = s.lower().replace("ё", "е").translate(_DASHES).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- чтение

def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_zip(path) -> list[tuple[str, str]]:
    """[(имя файла, текст)] для всех .txt внутри архива (вложенные папки допускаются)."""
    out = []
    with zipfile.ZipFile(path) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            name = info.filename
            if info.is_dir() or "__MACOSX" in name or not name.lower().endswith((".txt", ".md")):
                continue
            raw = z.read(info)
            for enc in ("utf-8-sig", "cp1251"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = raw.decode("utf-8", "replace")
            # NFC: архивы, перепакованные на macOS, хранят имена в NFD («й» = «и» + знак) - иначе меняется kb_sha
            out.append((unicodedata.normalize("NFC", Path(name).name), clean(text)))
    return out


def files_sha(files: list[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for name, text in files:
        h.update(name.encode())
        h.update(text.encode())
    return h.hexdigest()


def _registry_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", norm(value))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch) and ch.isalnum())


def _load_document_registry() -> list[dict]:
    if not REGISTRY_FILE.exists():
        return []
    payload = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8")) or {}
    entries = payload.get("documents", [])
    if not isinstance(entries, list):
        raise ValueError("В document_registry.yaml поле documents должно быть списком")
    return [entry for entry in entries if isinstance(entry, dict)]


def _registry_version() -> int:
    if not REGISTRY_FILE.exists():
        return 0
    payload = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8")) or {}
    return int(payload.get("version", 0)) if isinstance(payload, dict) else 0


def _registry_sha() -> str:
    if not REGISTRY_FILE.exists():
        return ""
    return hashlib.sha256(REGISTRY_FILE.read_bytes()).hexdigest()


def _registry_record(filename: str, records: list[dict]) -> dict:
    filename_key = _registry_key(filename)
    matches = [r for r in records if r.get("file_match") and _registry_key(str(r["file_match"])) in filename_key]
    if len(matches) > 1:
        raise ValueError(f"Для файла {filename!r} найдено несколько карточек в document_registry.yaml")
    return matches[0] if matches else {}


# ---------------------------------------------------------------- нарезка

ART = re.compile(r"^\s*статья\s+(\d+(?:\.\d+)*)\s*[.:]?\s*(.*)$", re.I)
SEC = re.compile(r"^\s*(?:раздел|глава|приложение)\s+([IVXLCDM\d]+(?:\.\d+)*)\s*[.:]?\s*(.*)$", re.I)
SUB = re.compile(r"^\s*[«\"“‘(]*\s*(\d+(?:\.\d+)+)\.?\s+(\S.*)$")   # 4.2 текст  /  «4.2. текст»
TOP = re.compile(r"^\s*[«\"“‘(]*\s*(\d+)\.\s+(\S.*)$")               # 4. текст
LETTER = re.compile(r"^\s*[«\"“‘(]*\s*([а-яё])\)\s+(\S.*)$", re.I)
PAREN = re.compile(r"^\s*[«\"“‘(]*\s*(\d+)\)\s+(\S.*)$")
DATE = re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{2,4}")      # строка, начинающаяся с даты, — не пункт


def _is_heading(rest: str) -> bool:
    rest = rest.strip()
    return len(rest) <= 100 and rest[:1].isupper() and not rest.endswith((".", ";", ",", ":"))


def _is_list_intro(line: str) -> bool:
    """Короткая вводная с двоеточием задаёт контекст следующего перечня."""
    value = line.strip()
    return bool(value.endswith(":") and len(value) <= 700)


def _is_unmarked_list_item(line: str) -> bool:
    """В законах перечни часто состоят из отдельных абзацев без номера."""
    value = line.strip()
    if not value:
        return False
    if value.startswith(("—", "–", "-", "•")):
        return True
    first = next((c for c in value if c.isalpha()), "")
    return bool(first and first.islower())


def _split_sentences(s: str, limit: int) -> list[str]:
    out, cur = [], ""
    for p in re.split(r"(?<=[.;!?])\s+", s):
        while len(p) > limit:
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:limit])
            p = p[limit:]
        if cur and len(cur) + 1 + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        out.append(cur)
    return out


def _pieces(body: str, limit: int) -> list[str]:
    units: list[str] = []
    for ln in body.split("\n"):
        units.extend([ln] if len(ln) <= limit else _split_sentences(ln, limit))
    pieces, cur = [], ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > limit:
            pieces.append(cur.strip())
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def split_unit(label: str, body: str) -> list[tuple[str, str]]:
    """Длинный пункт режем по подпунктам/абзацам/предложениям, сохраняя локатор."""
    if len(body) <= MAX_CHARS:
        return [(label, body)]
    res = []
    for k, piece in enumerate(_pieces(body, MAX_CHARS), 1):
        suffix = f", фрагмент {k}"
        res.append((f"{label}{suffix}" if label else f"фрагмент {k}", piece))
    return res


def chunk_document(doc_id: int, text: str) -> list[dict]:
    chunks: list[dict] = []
    st = {"art": None, "art_title": "", "section": ""}
    unit = {"label": "", "lines": [], "art": None, "art_title": "", "section": "",
            "point": "", "subpoint": "", "list_context": "", "list_group": "", "list_item": ""}
    active_list_context = ""
    active_list_group = ""
    active_list_number = 0
    list_group_number = 0

    def flush():
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(unit["lines"]).strip())
        if len(body) >= 15:
            for label, piece in split_unit(unit["label"], body):
                chunks.append({"doc": doc_id, "label": label, "article": unit["art"],
                               "art_title": unit["art_title"], "section": unit["section"],
                               "point": unit["point"], "subpoint": unit["subpoint"],
                               "list_context": unit["list_context"],
                               "list_group": unit["list_group"], "list_item": unit["list_item"],
                               "text": piece})
        unit["lines"] = []

    def start(label: str, point: str = "", subpoint: str = "", *,
              list_context: str = "", list_group: str = "", list_item: str = ""):
        flush()
        unit.update(label=label, art=st["art"], art_title=st["art_title"], section=st["section"],
                    point=point, subpoint=subpoint, list_context=list_context,
                    list_group=list_group, list_item=list_item)

    def next_list_item() -> tuple[str, str]:
        nonlocal active_list_number
        active_list_number += 1
        return active_list_group, f"{active_list_group}:{active_list_number}"

    def point_label(num: str) -> str:
        return (f"ст. {st['art']}, " if st["art"] else "") + f"п. {num}"

    has_article_heading = any(ART.match(x) for x in text.split("\n"))
    article_seen = False
    for line in text.split("\n"):
        if has_article_heading and not article_seen:
            if not ART.match(line):
                continue
            article_seen = True
        if not line.strip():
            unit["lines"].append("")
            continue
        if st["art"] is None and re.match(r"^\s*(?:источник:|url:|дата выгрузки:)", line, re.I):
            continue
        if DATE.match(line):
            unit["lines"].append(line.strip())
            continue
        # Вводная перечня. Нумерованный пункт с двоеточием («1. Ставка … При этом:»,
        # «2. … по ставке 10 процентов при реализации:») — сам норма, поэтому его
        # сохраняем отдельным чанком и дополнительно повторяем как контекст перечня.
        # Раньше такая строка терялась, и ключевая ставка/правило исчезали из базы.
        if _is_list_intro(line):
            flush()
            list_group_number += 1
            active_list_context = line.strip()
            active_list_group = f"{doc_id}:{st['art'] or st['section']}:{list_group_number}"
            active_list_number = 0
            if m := SUB.match(line):
                nums = m.group(1).split(".")
                start(point_label(m.group(1)), point=nums[0], subpoint=".".join(nums[1:]),
                      list_context=active_list_context, list_group=active_list_group)
                unit["lines"].append(line.strip())
            elif (m := TOP.match(line)) and not (_is_heading(m.group(2)) and not st["art"]):
                start(point_label(m.group(1)), point=m.group(1),
                      list_context=active_list_context, list_group=active_list_group)
                unit["lines"].append(line.strip())
            continue
        if m := ART.match(line):
            active_list_context = active_list_group = ""
            st.update(art=m.group(1), art_title=m.group(2).strip(), section="")
            start(f"ст. {m.group(1)}")
            # В некоторых выгрузках текст нормы идёт в одной строке с заголовком
            # («Статья 1. Установить ... 27 093 рубля»). Без этого теряется сам факт.
            if m.group(2).strip() and not _is_heading(m.group(2)):
                unit["lines"].append(line.strip())
        elif m := SEC.match(line):
            active_list_context = active_list_group = ""
            st["section"] = line.strip()[:120]
            start("")
        elif m := SUB.match(line):
            nums = m.group(1).split(".")
            group, item = next_list_item() if active_list_context else ("", "")
            start(point_label(m.group(1)), point=nums[0], subpoint=".".join(nums[1:]),
                  list_context=active_list_context, list_group=group, list_item=item)
            unit["lines"].append(line.strip())
        elif (m := TOP.match(line)) and _is_heading(m.group(2)) and not st["art"]:
            active_list_context = active_list_group = ""
            st["section"] = f"{m.group(1)}. {m.group(2).strip()}"
            start(f"разд. {m.group(1)}")
        elif m := TOP.match(line):
            group, item = next_list_item() if active_list_context else ("", "")
            start(point_label(m.group(1)), point=m.group(1), list_context=active_list_context,
                  list_group=group, list_item=item)
            unit["lines"].append(line.strip())
        elif m := PAREN.match(line):
            group, item = next_list_item() if active_list_context else ("", "")
            start(f"{point_label(unit['point'])}, подп. {m.group(1)}" if unit["point"] else f"подп. {m.group(1)}",
                  point=unit["point"], subpoint=m.group(1), list_context=active_list_context,
                  list_group=group, list_item=item)
            unit["lines"].append(line.strip())
        elif m := LETTER.match(line):
            group, item = next_list_item() if active_list_context else ("", "")
            start(f"{point_label(unit['point'])}, подп. {m.group(1)}" if unit["point"] else f"подп. {m.group(1)}",
                  point=unit["point"], subpoint=m.group(1).lower(), list_context=active_list_context,
                  list_group=group, list_item=item)
            unit["lines"].append(line.strip())
        elif active_list_context and _is_unmarked_list_item(line):
            group, item = next_list_item()
            label = f"ст. {st['art']}, абз. {active_list_number}" if st["art"] else f"абз. {active_list_number}"
            start(label, point=unit["point"], subpoint=str(active_list_number),
                  list_context=active_list_context, list_group=group, list_item=item)
            unit["lines"].append(line.strip())
        elif active_list_context and re.match(
                r"^\s*(?:(?:пункт|абзац)\s+\d+\s+(?:изменен|утратил|дополнен)|"
                r"о конституционно-правовом смысле|см\. постановление)", line, re.I):
            # Amendment notes between numbered items must not sever their list context.
            unit["lines"].append(line.strip())
        else:
            if active_list_context:
                # Обычный абзац после перечня: сохраняем привязку к статье/разделу,
                # иначе чанк теряет локатор (например, продолжение ст. 70 ТК РФ).
                label = f"ст. {st['art']}" if st["art"] else (st["section"] or "")
                start(label, point=unit["point"])
            active_list_context = active_list_group = ""
            active_list_number = 0
            unit["lines"].append(line.rstrip())
    flush()
    return chunks


# ---------------------------------------------------------------- карточки документов

ID_RE = re.compile(r"(?<![\w/-])([A-ZА-ЯЁ]{1,4}-\d{1,6}(?:/\d{2,4})?)(?![\w/-])")
NUM_RE = re.compile(r"№\s*([\w][\w\-/.]*)")
ORG_RE = re.compile(r"(?:ООО|АО|ПАО|ЗАО|ОАО|ИП)\s*[«\"“]([^»\"”]{2,60})[»\"”]")
QUOTE_RE = re.compile(r"«([^»]{3,50})»")


def extract_aliases(text: str) -> set[str]:
    head = text[:ALIAS_SCAN_CHARS]
    found: set[str] = set()
    found.update(m.group(1) for m in ID_RE.finditer(head))
    for m in NUM_RE.finditer(head):
        v = m.group(1).rstrip(".,;/-")
        if len(v) >= 3 or "-" in v or "/" in v:
            found.add(v)
    found.update(m.group(1).strip() for m in ORG_RE.finditer(head))
    for m in QUOTE_RE.finditer(head):
        v = m.group(1).strip()
        if v[:1].isupper() and len(v.split()) <= 4:
            found.add(v)
    return {norm(a) for a in found if len(a) >= 3}


RU_MONTHS = {"января": "01", "февраля": "02", "марта": "03", "апреля": "04", "мая": "05",
             "июня": "06", "июля": "07", "августа": "08", "сентября": "09", "октября": "10",
             "ноября": "11", "декабря": "12"}


def _iso_date(value: str) -> str | None:
    value = value.replace("\xa0", " ")
    m = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", value)
    if m:
        return f"{int(m[3]):04d}-{int(m[2]):02d}-{int(m[1]):02d}"
    m = re.search(r"\b(\d{1,2})\s+([а-яё]+)\s+(\d{4})\b", value, re.I)
    if m and m[2].lower() in RU_MONTHS:
        return f"{m[3]}-{RU_MONTHS[m[2].lower()]}-{int(m[1]):02d}"
    return None


def _document_fields(name: str, text: str) -> dict:
    n, t = norm(name), norm(text[:12000])
    doc_type = "закон" if any(x in n for x in ("нк_", "тк_", "коап_", "152фз", "мрот_")) else "документ компании"
    if "регламент" in n: doc_type = "регламент"
    elif "договор" in n: doc_type = "договор"
    elif "доп соглаш" in n: doc_type = "дополнительное соглашение"
    elif "письмо" in n: doc_type = "письмо"
    elif "служебная записка" in n: doc_type = "служебная записка"
    date_expr = r"(\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\s+[а-яё]+\s+\d{4})"
    edition_match = re.search(r"редакция от\s*" + date_expr, t)
    d = {"document_type": doc_type, "document_date": None,
         "edition_date": _iso_date(edition_match.group(1)) if edition_match else None,
         "as_of_date": None, "effective_from": None, "effective_to": None, "amendments": []}
    for key, pattern in (("as_of_date", r"дата выгрузки\s*[:\-]?\s*([^\n]+)"),
                         ("effective_from", r"(?:действует|вступает в силу) с\s*" + date_expr),
                         ("effective_to", r"(?:утратила|утрачивает) силу с\s*" + date_expr)):
        m = re.search(pattern, t)
        if m: d[key] = _iso_date(m.group(1))
    for line in text.splitlines():
        if re.search(r"(?:дополнена|изменена|вступает в силу|утрачивает силу)", line, re.I):
            dt = _iso_date(line)
            if dt: d["amendments"].append({"date": dt, "text": line.strip()[:300]})
    # Дата в имени письма/договора/соглашения
    d["document_date"] = _iso_date(name) or _iso_date(text[:800])
    return d


def _clean_title(name: str, text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # В законах первая строка часто «Статья N изменена с ...»; настоящий заголовок —
    # «Статья N. Название». Ищем именно его, иначе карточка и поиск теряют тему статьи.
    law_heading = next(
        (line for line in lines
         if re.match(r"^Статья\s+\d+(?:\.\d+)*\.\s+\S", line, re.I)
         and not re.match(r"^Статья\s+\d+(?:\.\d+)*\s+(?:изменена|дополнена|утратила)", line, re.I)),
        None,
    )
    if law_heading:
        return law_heading
    article_heading = next((line for line in lines if re.match(r"^Статья\s+\d", line, re.I)), None)
    if article_heading:
        return article_heading
    for line in lines:
        s = line
        if re.match(r"^(источник:|url:|дата выгрузки:|редакция от|редакция утратила силу)", s, re.I):
            continue
        if len(s) <= 150 and not re.match(r"^ч\w*\s+\w+\s+сил\w*", s, re.I):
            return s
    return Path(name).stem.replace("__", " — ")


def build_meta(files: list[tuple[str, str]]) -> dict:
    docs, chunks, alias_sets = [], [], []
    registry = _load_document_registry()
    for i, (name, text) in enumerate(files):
        title = _clean_title(name, text)
        record = _registry_record(name, registry)
        d = {"id": i, "file": name, "title": title,
             "head": re.sub(r"\s+", " ", " ".join(x.strip() for x in text.splitlines()
                     if x.strip() and not re.match(r"^(Источник:|URL:|Дата выгрузки:)", x.strip(), re.I))[:HEAD_CHARS]).strip()}
        d.update(_document_fields(name, text))
        if record:
            d["registry_id"] = record.get("id", "")
            d["document_type"] = record.get("document_type", d.get("document_type", ""))
            if record.get("title"):
                d["title"] = str(record["title"])
            d["number"] = record.get("number", "")
            d["parties"] = record.get("parties", [])
            # У законов дата принятия не равна редакции; не выдаём её за дату документа.
            if record.get("document_date"):
                d["document_date"] = record["document_date"]
            elif d.get("document_type") == "закон":
                d["document_date"] = None
            d["edition_date"] = record.get("edition") or d.get("edition_date")
            for key in ("status", "effective_from", "effective_to", "related_contract", "related_documents",
                        "amends", "changes", "applies_to", "references", "law_name", "article",
                        "article_title", "as_of_date"):
                if key in record:
                    d[key] = record[key]
            d["registry_aliases"] = [norm(str(a)) for a in record.get("aliases", []) if str(a).strip()]
            d["family"] = record.get("family", d.get("family", ""))
        else:
            d["registry_aliases"] = []
            d["family"] = "регламент согласования договоров ооо меридиан" if "регламент согласования договоров" in norm(name) else ""
        docs.append(d)
        alias_sets.append(extract_aliases(text))
        chunks.extend(chunk_document(i, text))

    # слишком «общие» алиасы (название самой компании и т. п.) отбрасываем
    df: dict[str, int] = {}
    for s in alias_sets:
        for a in s:
            df[a] = df.get(a, 0) + 1
    limit = max(4, int(0.25 * len(files)))
    for d, s in zip(docs, alias_sets):
        d["aliases"] = sorted({a for a in s if df[a] <= limit} | set(d.get("registry_aliases", [])))

    for n, c in enumerate(chunks):
        c["id"] = n
    for d in docs:
        d["n_chunks"] = sum(1 for c in chunks if c["doc"] == d["id"])
    return {"version": INDEX_SCHEMA_VERSION, "kb_sha": files_sha(files),
            "registry_version": _registry_version(), "registry_sha": _registry_sha(),
            "docs": docs, "chunks": chunks}


def chunk_context(docs: list[dict], c: dict) -> str:
    """Текст чанка вместе с контекстом — его индексируем, эмбеддим и отдаём реранкеру."""
    doc = docs[c["doc"]]
    parts = [doc["title"]]
    if doc.get("head"):
        parts.append(doc["head"])
    for key, caption in (("edition_date", "Редакция"), ("as_of_date", "Срез базы"),
                         ("effective_from", "Действует с"), ("effective_to", "Утратила силу с")):
        if doc.get(key): parts.append(f"{caption}: {doc[key]}")
    if doc.get("number"): parts.append(f"Номер документа: {doc['number']}")
    if doc.get("parties"): parts.append("Стороны: " + "; ".join(doc["parties"]))
    if doc.get("status"): parts.append(f"Статус по справочнику: {doc['status']}")
    if doc.get("related_contract"): parts.append(f"Связанный договор: {doc['related_contract']}")
    if doc.get("related_documents"): parts.append("Связанные документы: " + "; ".join(doc["related_documents"]))
    if doc.get("amends"): parts.append("Изменяет пункты: " + ", ".join(doc["amends"]))
    if doc.get("applies_to"): parts.append(f"Применение: {doc['applies_to']}")
    if doc.get("references"): parts.append("Ссылки: " + "; ".join(doc["references"]))
    if doc.get("registry_aliases"): parts.append("Синонимы: " + ", ".join(doc["registry_aliases"]))
    if c["article"]:
        parts.append(f"Статья {c['article']}. {c['art_title']}".strip())
    if c["section"]:
        parts.append(c["section"])
    if c["label"]:
        parts.append(c["label"])
    if c.get("point"): parts.append(f"Пункт: {c['point']}")
    if c.get("subpoint"): parts.append(f"Подпункт: {c['subpoint']}")
    if c.get("list_context"):
        parts.append(f"Контекст перечня: {c['list_context']}")
    parts.append(c["text"])
    return "\n".join(parts)


def _unique(values) -> list[str]:
    out: list[str] = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in out:
            out.append(value)
    return out


_POINT_IN_LABEL = re.compile(r"(?:^|,\s*)п(?:одп)?\.\s*(\d+(?:\.\d+)*)")


def _doc_points(meta: dict, doc_id: int) -> list[str]:
    """Номера пунктов/частей документа в порядке появления (из точных локаторов чанков)."""
    points: list[str] = []
    for chunk in meta.get("chunks", []):
        if chunk.get("doc") != doc_id:
            continue
        label = norm(chunk.get("label", ""))
        match = _POINT_IN_LABEL.search(label)
        if match:
            points.append(match.group(1))
            continue
        # У статей без разбивки на пункты локатор — «ст. N»; саму статью в список частей не добавляем.
        if chunk.get("point") and not chunk.get("label", "").startswith("ст."):
            points.append(str(chunk["point"]))
    return _unique(points)


def _doc_sections(meta: dict, doc_id: int) -> list[str]:
    return _unique(c.get("section") for c in meta.get("chunks", []) if c.get("doc") == doc_id)


def document_catalog(meta: dict) -> str:
    """Компактный справочник всех документов: по нему модель распознаёт отсутствующие акты и редакции."""
    lines = []
    for doc in meta.get("docs", []):
        facts = [f"[{doc['id']}] {doc.get('title') or doc['file']}"]
        if doc.get("number"):
            facts.append(f"номер: {doc['number']}")
        if doc.get("document_date"):
            facts.append(f"дата: {doc['document_date']}")
        if doc.get("edition_date"):
            facts.append(f"редакция: {doc['edition_date']}")
        if doc.get("effective_from"):
            facts.append(f"действует с: {doc['effective_from']}")
        if doc.get("effective_to"):
            facts.append(f"утратил силу с: {doc['effective_to']}")
        if doc.get("status"):
            facts.append(f"статус: {doc['status']}")
        if doc.get("related_contract"):
            facts.append(f"связан с договором: {doc['related_contract']}")
        if doc.get("amends"):
            facts.append("изменяет: " + ", ".join(doc["amends"]))
        if doc.get("applies_to"):
            facts.append(f"применяется: {doc['applies_to']}")
        lines.append("; ".join(facts))
    return "\n".join(lines)


def document_toc(meta: dict, doc_ids: set[int] | None = None) -> str:
    """Оглавление найденных документов: какие статьи, части и пункты в них реально есть."""
    lines = []
    for doc in meta.get("docs", []):
        if doc_ids is not None and int(doc["id"]) not in doc_ids:
            continue
        points = _doc_points(meta, int(doc["id"]))
        sections = _doc_sections(meta, int(doc["id"]))
        if doc.get("article") or doc.get("law_name"):
            parts = ", ".join(points[:80]) if points else "номера частей не распознаны"
            title = f" — {doc['article_title']}" if doc.get("article_title") else ""
            line = (f"- {doc.get('title')}: статья {doc.get('article', '')}{title}; "
                    f"доступны части/пункты: {parts}.")
        else:
            section_text = "; ".join(sections[:12]) if sections else "разделы не распознаны"
            point_text = ", ".join(points[:80]) if points else "номера пунктов не распознаны"
            line = f"- {doc.get('title')}: разделы: {section_text}. Пункты: {point_text}."
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------- индекс

def embed_all(texts: list[str]) -> np.ndarray:
    if config.EMB_BACKEND == "local":  # только оффлайн, на вашей машине (sentence-transformers в ZIP не нужен)
        model = _local_model()
        return np.asarray(model.encode(texts, batch_size=16, normalize_embeddings=True,
                                       show_progress_bar=True), dtype="float32")
    import llm
    return llm.embed(texts)


def _local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer
        if not config.LOCAL_EMB_PATH:
            raise ValueError("Для локальных эмбеддингов задайте EMB_MODEL или LOCAL_EMB_PATH в .env")
        _LOCAL_MODEL = SentenceTransformer(config.LOCAL_EMB_PATH)
    return _LOCAL_MODEL


def embed_queries(texts: list[str]) -> np.ndarray:
    """Query embeddings using the same backend as the prebuilt index."""
    if config.EMB_BACKEND == "local":
        return np.asarray(_local_model().encode(texts, batch_size=16, normalize_embeddings=True,
                                                show_progress_bar=False), dtype="float32")
    import llm
    return llm.embed(texts)


def save_index(meta: dict, emb: np.ndarray) -> None:
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    np.save(EMB_FILE, emb.astype("float32"))


def load_index() -> tuple[dict, np.ndarray] | None:
    if not (META_FILE.exists() and EMB_FILE.exists()):
        return None
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    emb = np.load(EMB_FILE)
    return (meta, emb) if meta.get("version") == INDEX_SCHEMA_VERSION and emb.shape[0] == len(meta["chunks"]) else None


def ensure_index(zip_path=None) -> tuple[dict, np.ndarray]:
    """Готовый индекс из корня ZIP; пересборка — только если он действительно не подходит.

    На платформе база и справочник те же, поэтому совпадения состава файлов и версии
    схемы достаточно: не тратим минуты прогона на повторные эмбеддинги из-за различий
    в переводах строк или хэше YAML.
    """
    loaded = load_index()
    if zip_path and Path(zip_path).exists():
        files = read_zip(zip_path)
        if loaded:
            indexed_names = {str(doc.get("file")) for doc in loaded[0].get("docs", [])}
            current_names = {name for name, _ in files}
            if indexed_names == current_names and loaded[0].get("kb_sha") == files_sha(files):
                return loaded
            if indexed_names == current_names:
                print("[kb] хэш базы отличается, но состав документов совпадает — "
                      "использую готовый индекс", file=sys.stderr)
                return loaded
        print("[kb] индекс отсутствует или устарел — пересобираю", file=sys.stderr)
        meta = build_meta(files)
        emb = embed_all([chunk_context(meta["docs"], c) for c in meta["chunks"]])
        try:
            save_index(meta, emb)
        except OSError:
            pass
        return meta, emb
    if loaded:
        return loaded
    raise FileNotFoundError("Нет ни готового индекса, ни knowledge-base.zip")
