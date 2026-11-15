# Юридический ассистент ПСБ (Hack&Change 2026)

ИИ-ассистент для юристов банка: RAG-поиск по нормативной базе, ответы со ссылками на пункты документов
и обоснованные отказы без выдуманных норм.

## Структура

| Путь | Что там |
|---|---|
| `solution.py` | Точка входа для автоматической проверки на платформе |
| `assistant.py` | Ядро: `answer(question, history)` - общее для сабмита и веба |
| `agent.py` | Гибридный retrieval, выбор редакции, промпт, проверка цитат и статусов |
| `rag_kb.py`, `kb.py` | Разбор базы, чанкинг, метаданные, каталог/оглавления, готовый индекс |
| `llm.py` | Клиент платформенных `/chat/completions`, `/embeddings`, `/rerank` с ретраями |
| `config.py` | Все настройки из переменных окружения |
| `document_registry.yaml` | Справочник 21 документа: реквизиты, редакции, связи, алиасы |
| `backend/` | Веб-API (FastAPI) |
| `frontend/` | Веб-чат |
| `index/` | Готовый индекс BAAI/bge-m3 (`index_emb.npy`, `index_meta.json`) |
| `scripts/build_submission.py` | Сборка плоского ZIP для платформы |
| `docs/` | Регламент, контракт API, решения команды |
| `data/` | Локальные данные (в git не попадают) |

## Быстрый старт (локально)

```bash
cp .env.example .env          # заполнить LLM_BASE_URL и LLM_API_KEY
pip install -r requirements.txt
# положить knowledge-base.zip и questions.csv в data/
python solution.py            # -> submission.csv
uvicorn backend.main:app --reload --port 8000
```

Локально без платформенного ключа можно использовать Ollama:

```bash
LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama \
LLM_MODEL=qwen2.5:latest EMB_MODEL=bge-m3:latest EMB_BACKEND=api \
DATA_DIR=data python solution.py
```

## Запуск через Docker

```bash
cp .env.example .env          # указать адрес и ключ API
docker compose up --build
```

API: http://localhost:8000 (проверка: `GET /health`).

## Сборка сабмита

```bash
python scripts/build_submission.py   # -> dist/submission.zip
```

Все файлы лежат в корне архива, без вложенных папок: корневые `.py`, `requirements.txt`
(из `requirements-submission.txt`), `document_registry.yaml`, `index_emb.npy`, `index_meta.json`.
`data/`, эталонные ответы и `typical_answers.csv` в архив не попадают.

## Переменные окружения

См. `.env.example`. Обязательные: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
Дополнительно: `EMB_MODEL`, `RERANKER_MODEL`, `TEMPERATURE`, `MAX_TOKENS`, `MAX_WORKERS`, `DATA_DIR`.
Реальные ключи в репозиторий не коммитим.

## Архитектура

`assistant.answer()` использует единый `LegalAgent`:

1. **Индекс.** 21 документ, 585 структурных чанков, эмбеддинги bge-m3 (1024). Схема индекса
   проверяется при загрузке; готовый индекс из ZIP используется без пересборки.
2. **Retrieval.** BM25 + dense-поиск, reciprocal rank fusion, точные бусты по номерам документов,
   пунктам и алиасам, приоритет документов из вопроса и связанных документов
   (допсоглашение ↔ договор, письмо → регламент). Опционально - платформенный реранкер.
3. **Редакции и даты.** Дата события извлекается из вопроса; для Регламента согласования
   выбирается редакция, действовавшая на эту дату. Допсоглашение применяется только к изменённым пунктам.
4. **Генерация.** Один вызов LLM на вопрос. В промпт входят каталог всех документов,
   оглавление релевантных, найденные фрагменты, история диалога и правила
   `answered / partial / refused_no_data / refused_offtopic / refused_illegal`.
5. **Проверка.** Ссылки на статьи и пункты сверяются с фрагментами и оглавлением документов;
   предложения с выдуманными ссылками удаляются. Ответ ограничен 150 словами и 10 000 символов.
6. **Надёжность.** `solution.py` держит общий дедлайн 17 минут, пишет CSV с `fsync`,
   при зависших запросах сдаёт уже готовые ответы.

## Порядок проверки

```bash
# 1. Синтаксис и загрузка индекса
python -m py_compile *.py
python -c "import assistant; a=assistant.warmup(); print(len(a.meta['chunks']))"

# 2. Прогон на демо-вопросах (локальный API)
LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... \
EMB_MODEL=bge-m3:latest EMB_BACKEND=api DATA_DIR=data python solution.py

# 3. Проверка формата submission.csv
python - <<'PY'
import csv
rows = list(csv.DictReader(open("submission.csv", encoding="utf-8")))
assert rows and all(r["id"] and r["answer"].strip() for r in rows)
print(len(rows), "ответов")
PY
```

На демо-наборе из 98 вопросов проверены: непустые ответы на все id, отсутствие превышения
150 слов, retrieval ключевых пунктов и отсутствие ложных срабатываний проверки цитат
на эталонных ответах.

## Известные ограничения

- Локальная `qwen2.5` заметно слабее платформенного DeepSeek V4.1 Flash: её ответы годятся
  для smoke-тестов, но не для оценки качества.
- Реранкер используется только если задан `RERANKER_MODEL`; при его отсутствии работает
  гибридный retrieval.
- База знаний фиксирована (21 документ); новые нормы и редакции не добавляются.
- Проверка цитат эвристическая: она страхует от выдуманных пунктов, но не заменяет
  юридическую верификацию ответа.
- На сложных вопросах с нехваткой данных ассистент даёт частичный ответ и явно перечисляет,
  чего не хватает, вместо расчёта или предположения.
