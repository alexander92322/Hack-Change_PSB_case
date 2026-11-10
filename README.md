# Юридический ассистент ПСБ (Hack&Change 2026)

ИИ-ассистент для юристов банка: RAG-поиск по нормативной базе, ответы со ссылками на пункты документов
и обоснованные отказы без выдуманных норм.

> Черновик. Разделы с пометкой TODO заполняются по ходу работы.

## Структура

| Путь | Что там |
|---|---|
| `solution.py` | Точка входа для автоматической проверки на платформе |
| `assistant.py` | Ядро: `answer(question, history)` - общее для сабмита и веба |
| `config.py` | Все настройки из переменных окружения |
| `backend/` | Веб-API (FastAPI) |
| `frontend/` | Веб-чат |
| `index/` | Готовый индекс базы знаний |
| `scripts/build_submission.py` | Сборка ZIP для платформы |
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

## Запуск через Docker

```bash
cp .env.example .env          # указать адрес и ключ API
docker compose up --build
```
API: http://localhost:8000 (проверка: `GET /health`). Интерфейс: TODO.

## Сборка сабмита

```bash
python scripts/build_submission.py   # -> dist/submission.zip
```

## Переменные окружения

См. `.env.example`. Обязательные: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
Реальные ключи в репозиторий не коммитим.

## Архитектура

TODO: схема (поиск -> один вызов LLM -> самопроверка ссылок).

## Порядок проверки основного сценария

TODO.

## Известные ограничения

TODO.
