#!/usr/bin/env python3
"""
Mini-RAG API Server
===================
Run:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /answer  – answer a free-text question
    GET  /health  – liveness check
    GET  /chunks  – list all indexed chunks (debugging)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from pipeline import (
    TFIDFIndex,
    chunk_by_sentences,
    check_grounding,
    generate_answers,
    load_documents,
    run_retrieval,
)

# ── Shared pipeline state ──────────────────────────────────────────────────────
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Build the index once at startup."""
    docs   = load_documents("kb")
    chunks = chunk_by_sentences(docs)
    index  = TFIDFIndex(chunks)
    _state["docs"]   = docs
    _state["chunks"] = chunks
    _state["index"]  = index
    print(f"[API] Index ready: {len(chunks)} chunks from {len(docs)} documents")
    yield
    _state.clear()


app = FastAPI(
    title="Mini-RAG API",
    description="Citation-strict Q&A over a local knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ──────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class AnswerResponse(BaseModel):
    answer_label: str
    answer:       str
    citations:    list[str]
    top_chunks:   list[dict]   # included for transparency


class HealthResponse(BaseModel):
    status:        str
    chunks_loaded: int
    docs_loaded:   int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/answer", response_model=AnswerResponse, summary="Answer a question")
async def answer_question(req: QuestionRequest) -> AnswerResponse:
    """
    Retrieve relevant chunks from the knowledge base and generate a
    citation-strict answer.

    - **answer_label**: `grounded_answer` | `insufficient_context` | `conflicting_context`
    - **answer**: prose answer with inline `[Doc Title §chunk_id]` citations
    - **citations**: list of citation strings referenced in the answer
    - **top_chunks**: top-5 retrieved chunks (for transparency / debugging)
    """
    index: TFIDFIndex = _state["index"]
    chunks: list[dict] = _state["chunks"]

    # Build a synthetic single-query list
    query = [{"query_id": "API_QUERY", "question": req.question}]

    retrieval = run_retrieval(query, index, top_k=5)
    answers   = generate_answers(query, retrieval)
    ans       = answers[0]

    return AnswerResponse(
        answer_label=ans["answer_label"],
        answer=ans["answer"],
        citations=ans["citations"],
        top_chunks=retrieval[0]["top_k"],
    )


@app.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        chunks_loaded=len(_state.get("chunks", [])),
        docs_loaded=len(_state.get("docs", [])),
    )


@app.get("/chunks", summary="List all indexed chunks (debug)")
async def list_chunks(limit: int = 50) -> JSONResponse:
    chunks = _state.get("chunks", [])
    return JSONResponse(
        content={
            "total": len(chunks),
            "returned": min(limit, len(chunks)),
            "chunks": chunks[:limit],
        }
    )
