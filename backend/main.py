"""Веб-API для чата. Использует то же ядро assistant.answer(), что и solution.py.

Запуск локально (из корня проекта):
    uvicorn backend.main:app --reload --port 8000
Моки вместо ядра (для фронта): задать переменную окружения MOCK=1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
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
    history = [m.model_dump() for m in req.history]
    if os.getenv("MOCK", "0") == "1":
        return mock_answer(req.question, history)
    return answer(req.question, history)


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
def get_document(doc_id: str) -> dict:
    doc = documents.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Документ не найден")
    return doc