"""Веб-API для чата. Использует то же ядро assistant.answer(), что и solution.py.

Запуск локально (из корня проекта):
    uvicorn backend.main:app --reload --port 8000
Моки вместо ядра (для фронта): задать переменную окружения MOCK=1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # корневые модули ядра
from assistant import answer  # noqa: E402
from backend import documents  # noqa: E402
from backend.mocks import mock_answer  # noqa: E402

app = FastAPI(title="Юридический ассистент ПСБ")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[Message] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock": os.getenv("MOCK", "0") == "1"}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "Вопрос не может быть пустым")
    history = [m.model_dump() for m in req.history]
    if os.getenv("MOCK", "0") == "1":
        return mock_answer(q, history)
    return answer(q, history)


@app.get("/documents")
def list_documents() -> list[dict]:
    docs = documents.load_documents()
    if not docs:
        raise HTTPException(503, "База знаний не найдена: положите knowledge-base.zip в папку data/")
    return [
        {k: d[k] for k in ("doc_id", "doc_title", "doc_type", "is_active")}
        for d in docs
    ]


@app.get("/documents/{doc_id}")
def get_document(doc_id: str, point: str | None = None) -> dict:
    doc = documents.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Документ не найден")

    doc_chunks = [
        {
            "chunk_id": c["chunk_id"],
            "section": c["section"],
            "point": c["point"],
            "start_char": c["start_char"],
            "end_char": c["end_char"],
        }
        for c in documents.load_chunks()
        if c["doc_id"] == doc_id
    ]
    result = {**doc, "chunks": doc_chunks}
    if point:
        result["highlight"] = documents.find_chunk_position(doc_id, point=point)
    return result


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str) -> dict:
    chunk = documents.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(404, "Фрагмент (чанк) не найден")
    return chunk