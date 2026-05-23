# Mini-RAG Pipeline

A replayable Retrieval-Augmented Generation pipeline that ingests a local knowledge base, indexes it, answers questions with citation-strict responses, and evaluates retrieval quality deterministically.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) set Anthropic API key for LLM-backed answers
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the full pipeline
python pipeline.py
# or: make run

# 4. Validate all artifacts
python validate.py
# or: make validate

# 5. Run end-to-end in one command
make all
```

## Architecture

```
kb/*.txt  ──► [Ingestion] ──► [Chunker] ──► [TF-IDF Index]
                                                    │
queries.json ──────────────────────────────► [Retrieval]
                                                    │
                                             [Answer Gen]  ◄── Anthropic API
                                             (or fallback)       (optional)
                                                    │
                               ┌────────────────────┼─────────────────┐
                          [Eval]              [Grounding Check]  [Chunk Compare]
                               │
                        artifacts/
```

### Pipeline Stages (enforced in code)

```
INIT → DOCUMENTS_LOADED → DOCUMENTS_CHUNKED → INDEX_BUILT
     → RETRIEVAL_COMPLETE → ANSWERS_GENERATED → EVALUATION_COMPLETE
     → VALIDATION_COMPLETE → RESULTS_FINALISED
```

Answers cannot be generated before `RETRIEVAL_COMPLETE`. The stage machine asserts this constraint.

## Components

### 1. Document Ingestion (`load_documents`)
- Discovers all `*.txt` files in `kb/` — does **not** depend on filenames or order
- Parses `Title:` and `Section:` header fields with regex
- Extracts the body as everything after the last header line

### 2. Chunking (two strategies)
| Strategy | Logic | Avg chars |
|---|---|---|
| `sentence` | One chunk per sentence (`.split` on `[.!?]\s`) | ~80–100 |
| `fixed_180` | Sliding 180-char window, 30-char overlap, word-boundary snapped | ~150–180 |

The sentence strategy is used for the primary pipeline artifacts. Both are compared in `artifacts/chunking_comparison.json`.

### 3. Retrieval — Custom TF-IDF (no sklearn)
- Smoothed IDF: `log((N+1)/(df+1)) + 1`
- TF: term count / total tokens per chunk
- Cosine similarity between query and every chunk vector
- Returns top-5 chunks per query; top-3 used for evaluation

### 4. Answer Generation
- **With API key**: calls `claude-sonnet-4-20250514` with a strict system prompt that forbids inventing facts; model returns JSON `{answer_label, answer, citations, used_chunk_ids}`
- **Without API key**: rule-based fallback — uses the top-ranked chunk verbatim with its citation
- All LLM calls are logged to `llm_calls.jsonl`

### 5. Retrieval Evaluation (deterministic)
- Checks whether expected doc title appears in top-3 retrieved titles
- Labels each query: `hit` / `partial_hit` / `miss`
- Produces aggregate: `top3_hit_rate`, counts

### 6. Grounding Check (deterministic heuristic)
- Verifies every citation exists in that query's retrieved set
- Checks keyword overlap between answer text and cited chunk (≥3 shared tokens = supported)
- Saves per-citation detail to `artifacts/grounding_check.json`

### 7. Chunking Comparison
- Runs both strategies, evaluates retrieval quality for each
- Explains tradeoff; identifies winner

### 8. REST API (`api.py`)
```bash
make api    # starts server at http://localhost:8000
```
```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "How long do bank withdrawals take?"}'
```
Response:
```json
{
  "answer_label": "grounded_answer",
  "answer": "Bank withdrawals may take 1 to 3 business days after approval. [Cash withdrawal processing §chunk_6]",
  "citations": ["[Cash withdrawal processing §chunk_6]"],
  "top_chunks": [...]
}
```

## Artifacts

| File | Contents |
|---|---|
| `artifacts/chunks.json` | All chunks with `chunk_id`, `doc_title`, `section`, `text`, char offsets |
| `artifacts/retrieval.json` | Top-5 retrieved chunks per query with scores |
| `artifacts/answers.json` | Answers with `answer_label`, inline citations, `used_chunk_ids` |
| `artifacts/eval.json` | Per-query retrieval status + aggregate summary |
| `artifacts/grounding_check.json` | Citation validity and keyword-overlap checks |
| `artifacts/chunking_comparison.json` | Sentence vs fixed-size hit rates and tradeoff analysis |
| `llm_calls.jsonl` | One JSON record per LLM call (stage, query_id, model, prompt_hash, etc.) |

## Controlled Vocabularies

```python
ANSWER_LABELS      = {"grounded_answer", "insufficient_context", "conflicting_context"}
RETRIEVAL_STATUSES = {"hit", "partial_hit", "miss"}
CITATION_FORMAT    = "[Doc Title §chunk_N]"   # validated by regex
```

## Without an API Key

The pipeline runs fully without `ANTHROPIC_API_KEY`. The fallback answer generator uses the top-ranked retrieved chunk verbatim with a valid citation. All artifacts are produced and all validation checks pass.

If you want LLM-backed answers:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python pipeline.py
```

## Replacing the Knowledge Base

The pipeline is fixture-independent:
- Drop any `.txt` files with `Title: ...` / `Section: ...` headers into `kb/`
- Replace `queries.json` with a file following the same schema
- Run `python pipeline.py` — everything regenerates

## Validation

```bash
python validate.py    # or: make validate
```

Checks:
- All required artifacts exist
- JSON files parse without error
- All queries were processed
- Each query has ≥ 3 retrieved chunks
- Retrieval scores are numeric
- Answer labels use only the controlled vocabulary
- `grounded_answer` entries have ≥ 1 citation
- Citations reference only retrieved chunk IDs
- Citation format matches `[Doc Title §chunk_N]`
- Evaluation statuses use only the controlled vocabulary
- Aggregate summary fields are present and internally consistent
