# 🔍 Mini-RAG Pipeline

> A replayable Retrieval-Augmented Generation pipeline — TF-IDF retrieval, citation-strict answers, and fully deterministic evaluation. Zero ML library dependencies.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange?logo=anthropic)
![License](https://img.shields.io/badge/License-MIT-green)
![Dependencies](https://img.shields.io/badge/ML%20Dependencies-Zero-brightgreen)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Setup Guide](#setup-guide)
- [Running the Pipeline](#running-the-pipeline)
- [Architecture](#architecture)
- [Output Artifacts](#output-artifacts)
- [Validation](#validation)
- [REST API](#rest-api)
- [Troubleshooting](#troubleshooting)

---

## Overview

Mini-RAG Pipeline ingests a local knowledge base, indexes it with a hand-rolled TF-IDF engine, retrieves relevant passages for any question, and generates citation-strict answers via the Anthropic API.

**Key design principles:**
- ✅ Retrieval, ranking, and evaluation are implemented **in code** — not delegated to the LLM
- ✅ Runs **fully without an API key** using a rule-based fallback
- ✅ Every run produces auditable JSON artifacts
- ✅ Stage machine enforces correct execution order

### Pipeline Stages

```
INIT → DOCUMENTS_LOADED → DOCUMENTS_CHUNKED → INDEX_BUILT
     → RETRIEVAL_COMPLETE → ANSWERS_GENERATED → EVALUATION_COMPLETE
     → VALIDATION_COMPLETE → RESULTS_FINALISED
```

---

## Project Structure

```
mini-rag/
├── pipeline.py              # Main pipeline — run this
├── validate.py              # Artifact validator (47 checks)
├── api.py                   # Optional REST API server
├── requirements.txt         # Python dependencies
├── Makefile                 # Shortcuts
│
├── kb/                      # Knowledge base — put your .txt files here
│   ├── product_a.txt
│   ├── product_b.txt
│   └── returns.txt
│
├── queries.json             # Questions + expected docs for evaluation
│
├── artifacts/               # Auto-generated on every run
│   ├── chunks.json
│   ├── retrieval.json
│   ├── answers.json
│   ├── eval.json
│   ├── grounding_check.json
│   └── chunking_comparison.json
│
└── llm_calls.jsonl          # LLM audit log (written only when API key is set)
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/mini-rag.git
cd mini-rag
pip install -r requirements.txt

# 2. Set your API key (optional — pipeline works without it)
set ANTHROPIC_API_KEY=sk-ant-...        # Windows
export ANTHROPIC_API_KEY=sk-ant-...     # Mac / Linux

# 3. Run
python pipeline.py

# 4. Validate
python validate.py
```

Or in one command:

```bash
make all        # run + validate
make fresh      # clean → run → validate
```

---

## Setup Guide

### 1. Knowledge Base (`kb/`)

Create a `kb/` folder and add `.txt` files. Each file **must** start with `Title:` and `Section:` headers:

```
Title: WidgetPro X1 Mouse
Section: Product Specifications

WidgetPro X1 is a wireless ergonomic mouse with a 4000 DPI optical sensor.
Battery life is 90 days on a single AA.
Compatible with Windows, macOS, and Linux.
Price: $49.99. Warranty: 2 years. SKU: WPX1-BLK.
```

> **Important:** The `Title:` value is used for evaluation matching. It must exactly match `expected_doc_titles` in `queries.json`.

---

### 2. Queries File (`queries.json`)

```json
[
  {
    "query_id": "q1",
    "question": "How long does the WidgetPro mouse battery last?",
    "expected_doc_titles": ["WidgetPro X1 Mouse"]
  },
  {
    "query_id": "q2",
    "question": "What switch type does the keyboard use?",
    "expected_doc_titles": ["KeyboardPro TKL"]
  },
  {
    "query_id": "q3",
    "question": "What is the return window for products?",
    "expected_doc_titles": ["Return Policy"]
  }
]
```

| Field | Type | Description |
|---|---|---|
| `query_id` | string | Unique ID — links retrieval, answer, and eval records |
| `question` | string | Natural language question |
| `expected_doc_titles` | string[] | Must exactly match `Title:` values in `kb/*.txt` |

---

### 3. API Key (Optional)

The pipeline runs fully **without** an API key using a rule-based fallback. To get LLM-generated answers:

**Option A — Environment variable (recommended):**

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
python pipeline.py

# Mac / Linux
export ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
python pipeline.py
```

**Option B — Hard-code in `pipeline.py`:**

Find line ~230 and change:

```python
# Before
client = _anthropic.Anthropic()

# After
client = _anthropic.Anthropic(api_key="sk-ant-YOUR_KEY_HERE")
```

> ⚠️ Option A is safer — your key never appears in source code.

---

## Running the Pipeline

### Commands

| Command | Description |
|---|---|
| `python pipeline.py` | Run the full pipeline |
| `python validate.py` | Validate all artifacts (47 checks) |
| `make run` | Shortcut for `python pipeline.py` |
| `make validate` | Shortcut for `python validate.py` |
| `make all` | Run pipeline then validate |
| `make fresh` | Delete artifacts → re-run → validate |
| `make api` | Start REST API at `http://localhost:8000` |
| `make clean` | Delete `artifacts/` and `llm_calls.jsonl` |

### Expected Terminal Output

**With API key:**

```
[STAGE] ── INIT
  Loaded 4 documents from 'kb/'
[STAGE] ── DOCUMENTS_LOADED
  Produced 19 sentence-level chunks
[STAGE] ── DOCUMENTS_CHUNKED
  Index built: vocab size = 135
[STAGE] ── INDEX_BUILT
  Loaded 4 queries from 'queries.json'
  LLM: Anthropic client ready
  Retrieved top-5 chunks for 4 queries
[STAGE] ── RETRIEVAL_COMPLETE
[STAGE] ── ANSWERS_GENERATED
  Hit rate (top-3): 100%  (4/4 hits)
[STAGE] ── EVALUATION_COMPLETE
  Grounding OK: 4/4 answers
  [sentence]  19 chunks, hit_rate=1.0
  [fixed_180] 11 chunks, hit_rate=0.75
[STAGE] ── VALIDATION_COMPLETE
[STAGE] ── RESULTS_FINALISED

✓ Pipeline complete – artifacts written to artifacts/
```

**Without API key** — identical output except:

```
  LLM: unavailable (...); using rule-based fallback
```

All 9 stages still fire. All artifacts are still created. `validate.py` still passes.

---

## Architecture

### TF-IDF Retrieval Engine

Implemented from scratch (~80 lines, zero external ML libraries):

| Step | Formula |
|---|---|
| Term Frequency | `count(term) / total_tokens` |
| Inverse Doc Frequency | `log((N+1) / (df+1)) + 1` (smoothed) |
| Similarity | Cosine similarity between query and chunk vectors |
| Returns | Top-5 chunks per query; top-3 used for evaluation |

### Chunking Strategies

Two strategies are benchmarked on every run:

| Strategy | Logic | Avg chars | Best for |
|---|---|---|---|
| `sentence` *(primary)* | One chunk per sentence | ~80–100 | Fact-lookup queries |
| `fixed_180` | Sliding 180-char window, 30-char overlap | ~150–180 | Longer context questions |

### Answer Generation

The system prompt forces the model to:
- Answer **only** from the provided context chunks
- Append `[Doc Title §chunk_N]` after every factual claim
- Return structured JSON — never markdown prose
- Use `insufficient_context` if the answer is not in the context

```json
{
  "answer_label": "grounded_answer",
  "answer": "Battery life is 90 days. [WidgetPro X1 Mouse §chunk_2]",
  "citations": ["[WidgetPro X1 Mouse §chunk_2]"],
  "used_chunk_ids": ["chunk_2"]
}
```

### Deterministic Evaluation

No LLM is involved in scoring:

| Check | Method |
|---|---|
| Top-3 hit rate | Retrieved doc titles vs `expected_doc_titles` |
| Citation format | Regex: `[Doc Title §chunk_N]` |
| Citation grounding | `chunk_id` must be in that query's retrieved set |
| Keyword overlap | ≥3 shared tokens between answer and cited chunk text |
| Chunking comparison | Both strategies benchmarked; winner reported |

### Controlled Vocabularies

```python
ANSWER_LABELS      = {"grounded_answer", "insufficient_context", "conflicting_context"}
RETRIEVAL_STATUSES = {"hit", "partial_hit", "miss"}
CITATION_FORMAT    = "[Doc Title §chunk_N]"   # validated by regex
```

---

## Output Artifacts

All files written to `artifacts/` on every run:

### `artifacts/chunks.json`
```json
[
  {
    "chunk_id":   "chunk_1",
    "doc_title":  "WidgetPro X1 Mouse",
    "section":    "Product Specifications",
    "text":       "WidgetPro X1 is a wireless ergonomic mouse with a 4000 DPI optical sensor.",
    "start_char": 0,
    "end_char":   74,
    "strategy":   "sentence"
  }
]
```

### `artifacts/retrieval.json`
```json
[
  {
    "query_id": "q1",
    "question": "How long does the WidgetPro mouse battery last?",
    "top_k": [
      {
        "rank": 1,
        "chunk_id":   "chunk_2",
        "doc_title":  "WidgetPro X1 Mouse",
        "score":      0.412381,
        "chunk_text": "Battery life is 90 days on a single AA."
      }
    ]
  }
]
```

### `artifacts/answers.json`
```json
[
  {
    "query_id":       "q1",
    "answer_label":   "grounded_answer",
    "answer":         "The battery lasts 90 days. [WidgetPro X1 Mouse §chunk_2]",
    "citations":      ["[WidgetPro X1 Mouse §chunk_2]"],
    "used_chunk_ids": ["chunk_2"]
  }
]
```

### `artifacts/eval.json`
```json
{
  "queries": [
    {
      "query_id":                  "q1",
      "expected_doc_titles":       ["WidgetPro X1 Mouse"],
      "retrieved_doc_titles_top3": ["WidgetPro X1 Mouse", "Return Policy", "KeyboardPro TKL"],
      "retrieval_status":          "hit",
      "matched_expected_title":    true,
      "explanation":               "Expected title found at rank 1"
    }
  ],
  "summary": {
    "top3_hit_rate": 1.0,
    "total_queries": 4,
    "hits":          4,
    "partial_hits":  0,
    "misses":        0
  }
}
```

### `artifacts/grounding_check.json`
```json
[
  {
    "query_id":     "q1",
    "answer_label": "grounded_answer",
    "citation_checks": [
      {
        "citation":           "[WidgetPro X1 Mouse §chunk_2]",
        "valid":              true,
        "chunk_in_retrieval": true,
        "keyword_overlap":    ["battery", "days", "mouse"],
        "overlap_count":      3,
        "supported":          true,
        "reason":             "ok"
      }
    ],
    "issues":       [],
    "grounding_ok": true
  }
]
```

### `artifacts/chunking_comparison.json`
```json
{
  "strategies": {
    "sentence":  { "num_chunks": 19, "avg_chunk_chars": 83.4,  "top3_hit_rate": 1.0  },
    "fixed_180": { "num_chunks": 11, "avg_chunk_chars": 162.1, "top3_hit_rate": 0.75 }
  },
  "best_strategy":     "sentence",
  "tradeoff_analysis": "Sentence-level chunking preserves complete atomic facts..."
}
```

### `llm_calls.jsonl`
One JSON record per LLM call (only written when API key is set):
```json
{"stage":"ANSWERS_GENERATED","query_id":"q1","timestamp":"2026-05-23T10:01:22+00:00","provider":"anthropic","model":"claude-sonnet-4-5","prompt_hash":"a3f9c12b4d8e7f01","input_artifacts":["artifacts/retrieval.json"],"output_artifact":"artifacts/answers.json"}
```

---

## Validation

```bash
python validate.py
```

```
============================================================
Mini-RAG Pipeline Validation
============================================================

[1] Required artifact files
  ✓  artifacts/chunks.json exists
  ✓  artifacts/retrieval.json exists
  ✓  artifacts/answers.json exists
  ✓  artifacts/eval.json exists

[2] JSON validity
  ✓  All artifact files are valid JSON

[3] chunks.json  (19 chunks)
  ✓  At least 1 chunk produced
  ✓  All chunks have required fields
  ✓  All chunk_ids are unique

[4] retrieval.json  (4 records)
  ✓  All 4 queries have retrieval results
  ✓  Each query has >= 3 retrieved chunks
  ✓  All retrieval scores are numeric

[5] answers.json  (4 records)
  ✓  All answer_labels use controlled vocabulary
  ✓  grounded_answer entries have >= 1 citation
  ✓  All citations match format [Doc Title §chunk_N]
  ✓  All cited chunk_ids were actually retrieved

[6] eval.json
  ✓  Required keys present
  ✓  hits + partial_hits + misses == total_queries

[7] Optional artifacts
  ✓  grounding_check.json present and valid JSON
  ✓  chunking_comparison.json present and valid JSON

============================================================
  Passed: 47   Failed: 0
  ✓  VALIDATION PASSED
============================================================
```

---

## REST API

Start the server:

```bash
make api
# or: uvicorn api:app --reload --port 8000
```

### `POST /answer`

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "How long does the mouse battery last?"}'
```

```json
{
  "answer_label": "grounded_answer",
  "answer": "The battery lasts 90 days. [WidgetPro X1 Mouse §chunk_2]",
  "citations": ["[WidgetPro X1 Mouse §chunk_2]"],
  "top_chunks": [
    { "rank": 1, "chunk_id": "chunk_2", "score": 0.412381, "chunk_text": "..." }
  ]
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "chunks_loaded": 19, "docs_loaded": 4}
```

### `GET /chunks?limit=10`

```bash
curl http://localhost:8000/chunks?limit=5
# Returns first 5 indexed chunks for debugging
```

---

## Replacing the Knowledge Base

The pipeline is fixture-independent. To use different documents:

1. Drop any `.txt` files with `Title:` / `Section:` headers into `kb/`
2. Update `queries.json` with questions relevant to the new documents
3. Ensure `expected_doc_titles` values exactly match the `Title:` values
4. Run `make fresh` — clean, re-index, re-evaluate

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: 'queries.json'` | File missing from project root | Create `queries.json` — see Setup Guide |
| `No .txt files found in 'kb/'` | `kb/` folder missing or empty | Create `kb/` and add `.txt` files with headers |
| `LLM: unavailable` | API key not set or `anthropic` not installed | Set env var or `pip install anthropic` |
| `citation format invalid` | Citation does not match `[Doc Title §chunk_N]` | Check `Title:` in `kb/*.txt` matches citation prefix exactly |
| `cited chunk_id was NOT retrieved` | LLM cited a chunk outside the retrieved set | Check `llm_calls.jsonl` and re-run |
| `Invalid stage jump` | Pipeline stage fired out of order | Do not reorder `_advance()` calls in `pipeline.py` |

---

## Dependencies

```
anthropic>=0.25.0     # LLM generation (optional)
fastapi>=0.110.0      # REST API (optional)
uvicorn>=0.29.0       # ASGI server (optional)
```

The TF-IDF retrieval engine uses **only the Python standard library** — no numpy, no scikit-learn, no vector database.

---

## License

MIT — see [LICENSE](LICENSE) for details.
