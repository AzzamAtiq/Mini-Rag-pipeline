#!/usr/bin/env python3
"""
Mini-RAG Pipeline
=================
Stages (enforced in code):
  INIT -> DOCUMENTS_LOADED -> DOCUMENTS_CHUNKED -> INDEX_BUILT
       -> RETRIEVAL_COMPLETE -> ANSWERS_GENERATED -> EVALUATION_COMPLETE
       -> VALIDATION_COMPLETE -> RESULTS_FINALISED

Retrieval:   Custom TF-IDF (no external ML library required)
Generation:  Anthropic claude-sonnet-4-20250514  (falls back to rule-based if key absent)
Evaluation:  Fully deterministic code
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Controlled Vocabularies ────────────────────────────────────────────────────
ANSWER_LABELS      = {"grounded_answer", "insufficient_context", "conflicting_context"}
RETRIEVAL_STATUSES = {"hit", "partial_hit", "miss"}
# [Doc Title §chunk_N]
CITATION_RE        = re.compile(r'\[([^\]]+?) §(chunk_\d+)\]')

# ── Pipeline Stage Machine ─────────────────────────────────────────────────────
_STAGE_ORDER = [
    "INIT",
    "DOCUMENTS_LOADED",
    "DOCUMENTS_CHUNKED",
    "INDEX_BUILT",
    "RETRIEVAL_COMPLETE",
    "ANSWERS_GENERATED",
    "EVALUATION_COMPLETE",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]
_STAGE_IDX     = {s: i for i, s in enumerate(_STAGE_ORDER)}
_current_stage = "INIT"


def _advance(to: str) -> None:
    global _current_stage
    want = _STAGE_IDX[to]
    have = _STAGE_IDX[_current_stage]
    if want != have + 1:
        raise RuntimeError(f"Invalid stage jump: {_current_stage} -> {to}")
    _current_stage = to
    print(f"[STAGE] ── {_current_stage}")


# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path("artifacts")
LLM_LOG_PATH  = Path("llm_calls.jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – TF-IDF ENGINE  (zero external dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    tf: dict[str, int] = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    total = len(tokens) or 1
    return {t: c / total for t, c in tf.items()}


def _build_idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    N  = len(corpus_tokens)
    df: dict[str, int] = defaultdict(int)
    for tokens in corpus_tokens:
        for t in set(tokens):
            df[t] += 1
    # Smooth IDF: log((N+1)/(df+1)) + 1
    return {t: math.log((N + 1) / (d + 1)) + 1.0 for t, d in df.items()}


def _tfidf_vec(tf: dict[str, float], idf: dict[str, float],
               vocab: list[str]) -> list[float]:
    return [tf.get(t, 0.0) * idf.get(t, 0.0) for t in vocab]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class TFIDFIndex:
    """In-memory TF-IDF index over a list of chunk dicts."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        corpus_tokens = [_tokenize(c["text"]) for c in chunks]
        self.idf   = _build_idf(corpus_tokens)
        self.vocab = sorted(self.idf.keys())
        self.vecs  = [
            _tfidf_vec(_term_freq(toks), self.idf, self.vocab)
            for toks in corpus_tokens
        ]

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        q_toks = _tokenize(question)
        q_tf   = _term_freq(q_toks)
        q_vec  = _tfidf_vec(q_tf, self.idf, self.vocab)
        scored = sorted(
            ((float(_cosine(q_vec, cv)), i) for i, cv in enumerate(self.vecs)),
            key=lambda x: -x[0],
        )
        results = []
        for rank, (score, idx) in enumerate(scored[:top_k], 1):
            c = self.chunks[idx]
            results.append({
                "rank":       rank,
                "chunk_id":   c["chunk_id"],
                "doc_title":  c["doc_title"],
                "score":      round(score, 6),
                "chunk_text": c["text"],
            })
        return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – DOCUMENT INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def load_documents(kb_dir: str = "kb") -> list[dict]:
    """
    Discover all *.txt files in kb_dir.
    Parse Title: and Section: header fields; everything after is the body.
    Does NOT depend on exact filenames or document order.
    """
    docs: list[dict] = []
    paths = sorted(glob.glob(os.path.join(kb_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in '{kb_dir}/'")

    for path in paths:
        raw = Path(path).read_text(encoding="utf-8").strip()

        title_m   = re.search(r"^Title:\s*(.+)$",   raw, re.MULTILINE)
        section_m = re.search(r"^Section:\s*(.+)$", raw, re.MULTILINE)

        # Body starts after the last header line
        last_header_end = 0
        for m in re.finditer(r"^(Title|Section):.+$", raw, re.MULTILINE):
            last_header_end = m.end()
        body = raw[last_header_end:].strip()

        docs.append({
            "filename": os.path.basename(path),
            "title":    title_m.group(1).strip()   if title_m   else Path(path).stem,
            "section":  section_m.group(1).strip() if section_m else "Unknown",
            "body":     body,
        })

    print(f"  Loaded {len(docs)} documents from '{kb_dir}/'")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – CHUNKING  (two strategies)
# ══════════════════════════════════════════════════════════════════════════════

def _assign_ids(chunks: list[dict], prefix: str = "chunk_") -> list[dict]:
    """Assign sequential chunk_N IDs to a list of partial chunk dicts."""
    for i, c in enumerate(chunks, 1):
        c["chunk_id"] = f"{prefix}{i}"
    return chunks


def chunk_by_sentences(docs: list[dict], strategy_id: str = "sentence") -> list[dict]:
    """
    One chunk per sentence.
    Preserves complete factual statements; ideal for fact-lookup queries.
    """
    raw_chunks: list[dict] = []
    for doc in docs:
        body = doc["body"]
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        cursor = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            start = body.find(sent, cursor)
            if start == -1:
                start = cursor
            end = start + len(sent)
            raw_chunks.append({
                "chunk_id":   "",          # assigned below
                "doc_title":  doc["title"],
                "section":    doc["section"],
                "text":       sent,
                "start_char": start,
                "end_char":   end,
                "strategy":   strategy_id,
            })
            cursor = end

    return _assign_ids(raw_chunks)


def chunk_fixed_size(
    docs: list[dict],
    size: int = 180,
    overlap: int = 30,
    strategy_id: str = "fixed_180",
) -> list[dict]:
    """
    Sliding window of `size` characters with `overlap` step-back.
    Snaps to word boundaries to avoid mid-word splits.
    """
    raw_chunks: list[dict] = []
    for doc in docs:
        body = doc["body"]
        pos  = 0
        while pos < len(body):
            end = min(pos + size, len(body))
            # Snap backwards to word boundary
            if end < len(body):
                snap = body.rfind(" ", pos, end)
                if snap > pos:
                    end = snap
            text = body[pos:end].strip()
            if text:
                raw_chunks.append({
                    "chunk_id":   "",
                    "doc_title":  doc["title"],
                    "section":    doc["section"],
                    "text":       text,
                    "start_char": pos,
                    "end_char":   end,
                    "strategy":   strategy_id,
                })
            step = max(1, size - overlap)
            pos += step

    return _assign_ids(raw_chunks)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def run_retrieval(
    queries: list[dict],
    index: TFIDFIndex,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve top_k chunks for every query."""
    results: list[dict] = []
    for q in queries:
        results.append({
            "query_id": q["query_id"],
            "question": q["question"],
            "top_k":    index.query(q["question"], top_k),
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – ANSWER GENERATION  (LLM-backed, with rule-based fallback)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are a citation-strict support assistant.
Answer ONLY using the context chunks provided. Each chunk is labelled [Doc Title §chunk_id].

Rules:
1. If the context is sufficient, answer concisely and append the citation after every factual claim, e.g. [Doc Title §chunk_1].
2. If the context does not contain enough information, respond: "I cannot answer based on the available context."
3. Never invent facts not present in the context.
4. Use EXACTLY this citation format: [Doc Title §chunk_id]

Return ONLY a valid JSON object — no markdown fences, no preamble:
{
  "answer_label": "<grounded_answer|insufficient_context|conflicting_context>",
  "answer": "<your answer with inline citations>",
  "citations": ["[Doc Title §chunk_id]", ...],
  "used_chunk_ids": ["chunk_id", ...]
}
"""


def _build_context_block(top_k: list[dict]) -> str:
    return "\n".join(
        f"[{item['doc_title']} §{item['chunk_id']}] {item['chunk_text']}"
        for item in top_k
    )


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256((system + user).encode()).hexdigest()[:16]


def _log_llm(record: dict) -> None:
    with open(LLM_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fallback_answer(qid: str, top_k: list[dict]) -> dict:
    """
    Rule-based answer when the LLM is unavailable.
    Uses the highest-scoring retrieved chunk verbatim.
    """
    if not top_k:
        return {
            "query_id":       qid,
            "answer_label":   "insufficient_context",
            "answer":         "Insufficient context to answer.",
            "citations":      [],
            "used_chunk_ids": [],
        }
    best     = top_k[0]
    citation = f"[{best['doc_title']} §{best['chunk_id']}]"
    return {
        "query_id":       qid,
        "answer_label":   "grounded_answer",
        "answer":         f"{best['chunk_text']} {citation}",
        "citations":      [citation],
        "used_chunk_ids": [best["chunk_id"]],
    }


def generate_answers(
    queries: list[dict],
    retrieval: list[dict],
) -> list[dict]:
    """
    Generate citation-strict answers.
    Uses Anthropic API if ANTHROPIC_API_KEY is set, otherwise falls back to
    the top-chunk rule-based approach.
    """
    # Try to initialise the Anthropic client
    client    = None
    use_llm   = False
    llm_model = "claude-sonnet-4-5"
    try:
        import anthropic as _anthropic  # noqa: PLC0415
        client  = _anthropic.Anthropic()
        use_llm = True
        print("  LLM: Anthropic client ready")
    except Exception as exc:
        print(f"  LLM: unavailable ({exc}); using rule-based fallback")

    ret_map = {r["query_id"]: r for r in retrieval}
    answers: list[dict] = []

    for q in queries:
        qid     = q["query_id"]
        ret     = ret_map[qid]
        context = _build_context_block(ret["top_k"])
        user_msg = f"Context:\n{context}\n\nQuestion: {q['question']}"

        if use_llm:
            try:
                ts = datetime.now(timezone.utc).isoformat()
                response = client.messages.create(
                    model=llm_model,
                    max_tokens=600,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = response.content[0].text.strip()
                # Strip accidental markdown fences
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$",           "", raw)
                parsed = json.loads(raw)

                # Log the call
                _log_llm({
                    "stage":            "ANSWERS_GENERATED",
                    "query_id":         qid,
                    "timestamp":        ts,
                    "provider":         "anthropic",
                    "model":            llm_model,
                    "prompt_hash":      _prompt_hash(_SYSTEM_PROMPT, user_msg),
                    "input_artifacts":  ["artifacts/retrieval.json"],
                    "output_artifact":  "artifacts/answers.json",
                })

                answer: dict = {
                    "query_id":       qid,
                    "answer_label":   parsed.get("answer_label", "insufficient_context"),
                    "answer":         parsed.get("answer", ""),
                    "citations":      parsed.get("citations", []),
                    "used_chunk_ids": parsed.get("used_chunk_ids", []),
                }
            except Exception as exc:
                print(f"  LLM call failed for {qid}: {exc}; using fallback")
                answer = _fallback_answer(qid, ret["top_k"])
        else:
            answer = _fallback_answer(qid, ret["top_k"])

        # Enforce controlled vocabulary
        if answer["answer_label"] not in ANSWER_LABELS:
            answer["answer_label"] = "insufficient_context"

        answers.append(answer)

    return answers


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – RETRIEVAL EVALUATION  (fully deterministic)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_retrieval(
    queries: list[dict],
    retrieval: list[dict],
) -> tuple[list[dict], dict]:
    """
    Compare top-3 retrieved doc titles against expected_doc_titles from queries.json.
    Returns (per-query records, aggregate summary).
    """
    ret_map = {r["query_id"]: r for r in retrieval}
    records: list[dict] = []
    hits = partial = misses = 0

    for q in queries:
        qid      = q["query_id"]
        expected = q.get("expected_doc_titles", [])
        top_all  = ret_map.get(qid, {}).get("top_k", [])
        top3     = top_all[:3]

        titles_top3 = [c["doc_title"] for c in top3]
        titles_all  = [c["doc_title"] for c in top_all]

        matched_top3 = any(t in titles_top3 for t in expected)

        if matched_top3:
            # Find the rank at which the expected title appears
            rank = next(
                (c["rank"] for c in top3 if c["doc_title"] in expected),
                None,
            )
            status      = "hit"
            explanation = f"Expected title found at rank {rank}"
            hits       += 1
        elif any(t in titles_all for t in expected):
            status      = "partial_hit"
            explanation = f"Expected title found outside top 3 (within top {len(top_all)})"
            partial    += 1
        else:
            status      = "miss"
            explanation = "Expected title not found in any retrieved result"
            misses     += 1

        records.append({
            "query_id":                  qid,
            "expected_doc_titles":       expected,
            "retrieved_doc_titles_top3": titles_top3,
            "retrieval_status":          status,
            "matched_expected_title":    matched_top3,
            "explanation":               explanation,
        })

    total = len(queries) or 1
    summary = {
        "top3_hit_rate": round(hits / total, 4),
        "total_queries": len(queries),
        "hits":          hits,
        "partial_hits":  partial,
        "misses":        misses,
    }
    return records, summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – GROUNDING CHECK  (deterministic heuristic)
# ══════════════════════════════════════════════════════════════════════════════

_MIN_OVERLAP_TOKENS = 3   # minimum shared content words to consider "supported"


def check_grounding(
    answers: list[dict],
    retrieval: list[dict],
    chunks: list[dict],
) -> list[dict]:
    """
    For each answer:
      1. Verify every citation exists in that query's retrieved set.
      2. Check keyword overlap between answer text and cited chunk text.
    """
    # Map query_id -> {chunk_id -> retrieval item}
    ret_map: dict[str, dict[str, dict]] = {
        r["query_id"]: {item["chunk_id"]: item for item in r["top_k"]}
        for r in retrieval
    }
    # Map chunk_id -> full chunk
    chunk_map: dict[str, dict] = {c["chunk_id"]: c for c in chunks}

    results: list[dict] = []

    for ans in answers:
        qid       = ans["query_id"]
        retrieved = ret_map.get(qid, {})
        issues: list[str] = []
        citation_checks: list[dict] = []

        for cit in ans.get("citations", []):
            m = CITATION_RE.match(cit.strip())
            if not m:
                issues.append(f"Malformed citation format: {cit!r}")
                citation_checks.append({
                    "citation": cit, "valid": False, "reason": "malformed_format",
                })
                continue

            chunk_id = m.group(2)

            if chunk_id not in retrieved:
                issues.append(f"{chunk_id} cited but not in retrieved set")
                citation_checks.append({
                    "citation": cit,
                    "valid":    False,
                    "reason":   "not_retrieved",
                    "chunk_id": chunk_id,
                })
                continue

            # Keyword-overlap check
            chunk_text   = chunk_map.get(chunk_id, {}).get("text", "")
            answer_words = set(_tokenize(ans.get("answer", "")))
            chunk_words  = set(_tokenize(chunk_text))
            overlap      = answer_words & chunk_words
            supported    = len(overlap) >= _MIN_OVERLAP_TOKENS

            if not supported:
                issues.append(
                    f"{chunk_id} has insufficient keyword overlap "
                    f"({len(overlap)} shared tokens, need {_MIN_OVERLAP_TOKENS})"
                )

            citation_checks.append({
                "citation":             cit,
                "chunk_id":             chunk_id,
                "valid":                True,
                "chunk_in_retrieval":   True,
                "keyword_overlap":      sorted(overlap),
                "overlap_count":        len(overlap),
                "supported":            supported,
                "reason":               "ok" if supported else "low_keyword_overlap",
            })

        results.append({
            "query_id":        qid,
            "answer_label":    ans["answer_label"],
            "citation_checks": citation_checks,
            "issues":          issues,
            "grounding_ok":    len(issues) == 0,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – CHUNKING COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def run_chunking_comparison(
    docs: list[dict],
    queries: list[dict],
) -> dict:
    """
    Run both chunking strategies, evaluate retrieval quality for each,
    and summarise the tradeoff.
    """
    strategies: dict[str, list[dict]] = {
        "sentence":  chunk_by_sentences(docs,  strategy_id="sentence"),
        "fixed_180": chunk_fixed_size(docs, size=180, overlap=30,
                                      strategy_id="fixed_180"),
    }

    comparison: dict[str, dict] = {}
    for name, cks in strategies.items():
        idx      = TFIDFIndex(cks)
        ret      = run_retrieval(queries, idx, top_k=3)
        _, summ  = evaluate_retrieval(queries, ret)
        avg_len  = round(sum(len(c["text"]) for c in cks) / len(cks), 1) if cks else 0
        comparison[name] = {
            "num_chunks":        len(cks),
            "avg_chunk_chars":   avg_len,
            "top3_hit_rate":     summ["top3_hit_rate"],
            "hits":              summ["hits"],
            "partial_hits":      summ["partial_hits"],
            "misses":            summ["misses"],
            "total_queries":     summ["total_queries"],
        }

    winner = max(comparison, key=lambda k: comparison[k]["top3_hit_rate"])

    return {
        "strategies":       comparison,
        "best_strategy":    winner,
        "tradeoff_analysis": (
            "Sentence-level chunking preserves complete atomic facts, so each "
            "retrieved chunk maps cleanly to a single answer point — ideal for "
            "narrow fact-lookup queries. Fixed-size chunking may split a sentence "
            "across two chunks, degrading TF-IDF signal and potentially returning "
            "a partial fact. However, fixed-size chunks can capture more context "
            "per chunk for longer, multi-sentence questions, improving recall at "
            "the expense of precision. For this knowledge base (short, assertion-"
            "dense articles) sentence chunking typically wins on top-3 hit rate."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    kb_dir: str   = "kb",
    queries_path: str = "queries.json",
    top_k: int    = 5,
) -> None:
    """Execute the full pipeline end-to-end."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    if LLM_LOG_PATH.exists():
        LLM_LOG_PATH.unlink()

    print(f"[STAGE] ── INIT")

    # ── 1. Load documents ────────────────────────────────────────────────────
    docs = load_documents(kb_dir)
    _advance("DOCUMENTS_LOADED")

    # ── 2. Chunk (sentence strategy = primary) ───────────────────────────────
    chunks = chunk_by_sentences(docs)
    print(f"  Produced {len(chunks)} sentence-level chunks")
    (ARTIFACTS_DIR / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _advance("DOCUMENTS_CHUNKED")

    # ── 3. Build TF-IDF index ────────────────────────────────────────────────
    index = TFIDFIndex(chunks)
    print(f"  Index built: vocab size = {len(index.vocab)}")
    _advance("INDEX_BUILT")

    # ── 4. Load queries ──────────────────────────────────────────────────────
    queries = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    print(f"  Loaded {len(queries)} queries from '{queries_path}'")

    # ── 5. Retrieval ─────────────────────────────────────────────────────────
    retrieval = run_retrieval(queries, index, top_k=top_k)
    (ARTIFACTS_DIR / "retrieval.json").write_text(
        json.dumps(retrieval, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Retrieved top-{top_k} chunks for {len(retrieval)} queries")
    _advance("RETRIEVAL_COMPLETE")

    # ── 6. Answer generation (only after RETRIEVAL_COMPLETE) ─────────────────
    answers = generate_answers(queries, retrieval)
    (ARTIFACTS_DIR / "answers.json").write_text(
        json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _advance("ANSWERS_GENERATED")

    # ── 7. Retrieval evaluation ───────────────────────────────────────────────
    eval_records, eval_summary = evaluate_retrieval(queries, retrieval)
    eval_output = {"queries": eval_records, "summary": eval_summary}
    (ARTIFACTS_DIR / "eval.json").write_text(
        json.dumps(eval_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Hit rate (top-3): {eval_summary['top3_hit_rate'] * 100:.0f}%  "
          f"({eval_summary['hits']}/{eval_summary['total_queries']} hits)")
    _advance("EVALUATION_COMPLETE")

    # ── 8. Grounding check ────────────────────────────────────────────────────
    grounding = check_grounding(answers, retrieval, chunks)
    (ARTIFACTS_DIR / "grounding_check.json").write_text(
        json.dumps(grounding, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ok_count = sum(1 for g in grounding if g["grounding_ok"])
    print(f"  Grounding OK: {ok_count}/{len(grounding)} answers")

    # ── 9. Chunking comparison ────────────────────────────────────────────────
    comparison = run_chunking_comparison(docs, queries)
    (ARTIFACTS_DIR / "chunking_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for strat, stats in comparison["strategies"].items():
        print(f"  [{strat}] {stats['num_chunks']} chunks, "
              f"hit_rate={stats['top3_hit_rate']}")

    _advance("VALIDATION_COMPLETE")
    _advance("RESULTS_FINALISED")

    print("\n✓ Pipeline complete – artifacts written to artifacts/")


if __name__ == "__main__":
    run_pipeline()
