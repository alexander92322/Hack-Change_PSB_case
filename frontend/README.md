# frontend/

Веб-чат (React + TypeScript). Инициализация - задача 0.FE.1.

Работает с API бэкенда (`http://localhost:8000`):
- `POST /ask` - `{question, history}` -> `{answer, status, sources}`;
- `GET /documents`, `GET /documents/{doc_id}` - просмотр документа и подсветка фрагмента.

Контракт - в `docs/api.md`.
