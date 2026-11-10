"""Веб-API для чата. Использует то же ядро assistant.answer(), что и solution.py.

Запуск локально:  uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # корневые модули ядра
from assistant import answer  # noqa: E402

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
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return answer(req.question, [m.model_dump() for m in req.history])


# --- Моки до готовности индекса (задача 1.BE.2) ---
_DEMO_DOC = {
    "doc_id": "demo",
    "doc_title": "Демо-документ",
    "text": "1.1. Здесь будет фрагмент документа, подтверждающий ответ.\n1.2. Остальной текст документа.",
}


@app.get("/documents")
def list_documents() -> list[dict]:
    return [{"doc_id": _DEMO_DOC["doc_id"], "doc_title": _DEMO_DOC["doc_title"]}]


@app.get("/documents/{doc_id}")
def get_document(doc_id: str) -> dict:
    if doc_id != _DEMO_DOC["doc_id"]:
        raise HTTPException(404, "Документ не найден")
    return _DEMO_DOC
