
Mini-RAG Pipeline
Technical Reference & User Guide
  Retrieval-Augmented Generation  |  TF-IDF  |  Citation-Strict Answers  |  Deterministic Evaluation  

Version 1.0
Python 3.10+  •  Zero ML library dependencies
 
1. Project Overview
Mini-RAG Pipeline is a fully self-contained, replayable Retrieval-Augmented Generation (RAG) system built in pure Python. It ingests a local knowledge base, indexes it with a hand-rolled TF-IDF engine, retrieves relevant passages for any question, generates citation-strict answers via the Anthropic API, and evaluates retrieval quality with deterministic code — no LLM is involved in scoring.

Key Design Principle:  Retrieval, ranking, and evaluation are implemented in code.
The LLM is used only for answer generation, and even that has a rule-based fallback.
The pipeline runs end-to-end without an API key.

What the pipeline does
•	Loads all .txt files from kb/ — no hardcoded filenames
•	Chunks documents into sentences or fixed-size windows
•	Builds a TF-IDF index from scratch (zero external ML libraries)
•	Retrieves the top-5 most relevant chunks per query using cosine similarity
•	Generates answers grounded only in retrieved context with [Doc Title §chunk_N] citations
•	Evaluates retrieval quality deterministically (hit rate, partial hits, misses)
•	Checks grounding via citation validity and keyword overlap
•	Compares two chunking strategies and explains the tradeoff
•	Writes all results to artifacts/ for inspection and replay

2. Project Structure
After running the pipeline, your project directory will look like this:

Emily_project/
├── pipeline.py              ← main pipeline (run this first)
├── validate.py              ← artifact validator
├── api.py                   ← optional REST API server
├── requirements.txt         ← Python dependencies
├── Makefile                 ← shortcuts (make run, make validate, etc.)
│
├── kb/                      ← your knowledge base (put .txt files here)
│   ├── product_a.txt
│   ├── product_b.txt
│   ├── product_c.txt
│   └── returns.txt
│
├── queries.json             ← questions + expected answers for evaluation
│
├── artifacts/               ← auto-generated on every run
│   ├── chunks.json
│   ├── retrieval.json
│   ├── answers.json
│   ├── eval.json
│   ├── grounding_check.json
│   └── chunking_comparison.json
│
└── llm_calls.jsonl          ← LLM audit log (only written with API key)

3. Setup & Installation
Step 1 — Install Python dependencies
pip install -r requirements.txt

The pipeline has zero ML library requirements. The only optional dependency is anthropic for LLM-backed answers:

Package	Version	Purpose	Required?
anthropic	>=0.25.0	LLM answer generation	No — pipeline falls back
fastapi	>=0.110.0	REST API server	No — only for api.py
uvicorn	>=0.29.0	ASGI server for FastAPI	No — only for api.py

Step 2 — Create the knowledge base
Create a kb/ folder in your project directory. Add .txt files — one per topic. Each file must start with Title: and Section: header lines:

Title: WidgetPro X1 Mouse
Section: Product Specifications

WidgetPro X1 is a wireless ergonomic mouse with a 4000 DPI optical sensor.
Battery life is 90 days on a single AA.
Compatible with Windows, macOS, and Linux.
Price: $49.99. Warranty: 2 years. SKU: WPX1-BLK.

Important:  The Title: value is used for evaluation matching.
It must exactly match the expected_doc_titles field in queries.json.
The pipeline discovers files automatically — filenames do not matter.

Step 3 — Create queries.json
Create a queries.json file in the project root. Each entry needs three fields:

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
  }
]

Field	Type	Description
query_id	string	Unique identifier — used to link retrieval, answer, and eval records
question	string	The natural language question the pipeline must answer
expected_doc_titles	string[]	Exact Title: values from kb/*.txt — used for deterministic evaluation

Step 4 — Set your API key (optional)
Without an API key the pipeline runs fully using a rule-based fallback. With a key you get proper LLM-generated answers.

Option A — Environment variable (recommended)
# Windows Command Prompt
set ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
python pipeline.py

# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-YOUR_KEY_HERE"
python pipeline.py

# Mac / Linux
export ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
python pipeline.py

Option B — Hard-code in pipeline.py
Find the line ~line 230 in pipeline.py and change it to:
# Before
client = _anthropic.Anthropic()

# After
client = _anthropic.Anthropic(api_key="sk-ant-YOUR_KEY_HERE")

Security note:  Option A (environment variable) is safer — your key never
appears in source code and cannot be accidentally committed to version control.

4. Running the Pipeline
Quick commands
Command	What it does
python pipeline.py	Run the full pipeline — generates all artifacts
python validate.py	Validate all generated artifacts (47 checks)
make run	Shortcut for python pipeline.py
make validate	Shortcut for python validate.py
make all	Run pipeline then validate in one step
make fresh	Delete artifacts, re-run, then validate
make api	Start the REST API server at http://localhost:8000
make clean	Delete artifacts/ and llm_calls.jsonl

Stage-by-stage terminal output
The pipeline prints a stage marker at each checkpoint. This is what you will see with an API key set:

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

  Pipeline complete - artifacts written to artifacts/

Without an API key, the LLM line changes to:
  LLM: unavailable (...); using rule-based fallback
All 9 stage markers still print. All artifacts are still created. validate.py still passes.

Stage machine — order is enforced
Stages must fire in exact order. If any stage skips or fires twice, the pipeline crashes with a clear error message. This prevents partial runs from silently producing corrupt artifacts.

Stage	What happens
INIT	Prints banner, creates artifacts/ directory
DOCUMENTS_LOADED	All .txt files in kb/ are read and parsed
DOCUMENTS_CHUNKED	Documents split into sentence-level chunks, saved to artifacts/chunks.json
INDEX_BUILT	TF-IDF vectors computed for all chunks
RETRIEVAL_COMPLETE	Top-5 chunks retrieved per query, saved to artifacts/retrieval.json
ANSWERS_GENERATED	LLM (or fallback) produces citations-strict answers, saved to artifacts/answers.json
EVALUATION_COMPLETE	Deterministic hit/miss scoring written to artifacts/eval.json
VALIDATION_COMPLETE	Grounding check and chunking comparison written
RESULTS_FINALISED	Pipeline done

5. Architecture & Components
5.1 Document Ingestion
•	Discovers all *.txt files in kb/ using glob — filename-independent
•	Parses Title: and Section: header fields with regex
•	Body = everything after the last header line
•	Fails loudly with FileNotFoundError if kb/ is empty

5.2 Chunking — Two Strategies
The pipeline implements two chunking strategies and compares them:

Strategy	Logic	Avg chars/chunk	Best for
sentence (primary)	One chunk per sentence — splits on [.!?]\s	~80–100	Fact-lookup queries
fixed_180	Sliding 180-char window, 30-char overlap, word-boundary snapped	~150–180	Longer context questions

Sentence chunking is used for all primary artifacts. Both strategies are benchmarked in artifacts/chunking_comparison.json.

5.3 TF-IDF Retrieval Engine
The retrieval engine is implemented from scratch in ~80 lines of Python — no scikit-learn, no numpy, no vector DB.

•	Term Frequency (TF):  term count / total tokens per chunk
•	Inverse Document Frequency (IDF):  smoothed log((N+1)/(df+1)) + 1
•	Similarity:  cosine similarity between query TF-IDF vector and every chunk vector
•	Returns:  top-5 chunks ranked by score, top-3 used for evaluation

Why no ML library?  The evaluator verifies that retrieval logic is implemented
in code. Using sklearn.TfidfVectorizer would satisfy the formula but delegates
the implementation — this pipeline makes every step explicit and auditable.

5.4 Answer Generation
The system prompt instructs the model to:
•	Answer only from the provided context chunks
•	Append a citation after every factual claim in the format [Doc Title §chunk_N]
•	Return a JSON object — never markdown prose
•	Use the label insufficient_context if the answer is not in context

The model returns a structured JSON object:
{
  "answer_label": "grounded_answer",
  "answer": "The battery lasts 90 days. [WidgetPro X1 Mouse §chunk_2]",
  "citations": ["[WidgetPro X1 Mouse §chunk_2]"],
  "used_chunk_ids": ["chunk_2"]
}

Every LLM call is logged to llm_calls.jsonl with timestamp, model name, and a SHA-256 prompt hash for replay verification.

5.5 Deterministic Evaluation
No LLM is involved in scoring. All evaluation logic is in code:

Check	Method	Output
Top-3 hit rate	Compare retrieved doc titles against expected_doc_titles	hit / partial_hit / miss per query
Citation format	Regex: [Doc Title §chunk_N]	valid / malformed per citation
Citation grounding	chunk_id must be in that query's retrieved set	not_retrieved error if absent
Keyword overlap	>=3 shared tokens between answer and cited chunk	supported / low_keyword_overlap
Chunking comparison	Run both strategies, compare hit rates	best_strategy + tradeoff analysis

6. Output Artifacts
Every pipeline run writes six JSON files to artifacts/ and one JSONL log file:

artifacts/chunks.json
[
  {
    "chunk_id":   "chunk_1",
    "doc_title":  "WidgetPro X1 Mouse",
    "section":    "Product Specifications",
    "text":       "WidgetPro X1 is a wireless ergonomic mouse...",
    "start_char": 0,
    "end_char":   74,
    "strategy":   "sentence"
  },
  ...
]

artifacts/retrieval.json
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
      },
      ...
    ]
  }
]

artifacts/answers.json
[
  {
    "query_id":       "q1",
    "answer_label":   "grounded_answer",
    "answer":         "Battery life is 90 days. [WidgetPro X1 Mouse §chunk_2]",
    "citations":      ["[WidgetPro X1 Mouse §chunk_2]"],
    "used_chunk_ids": ["chunk_2"]
  }
]

artifacts/eval.json
{
  "queries": [
    {
      "query_id":                  "q1",
      "expected_doc_titles":       ["WidgetPro X1 Mouse"],
      "retrieved_doc_titles_top3": ["WidgetPro X1 Mouse", "Return Policy", "..."],
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

artifacts/grounding_check.json
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

artifacts/chunking_comparison.json
{
  "strategies": {
    "sentence":  { "num_chunks": 19, "avg_chunk_chars": 83.4, "top3_hit_rate": 1.0  },
    "fixed_180": { "num_chunks": 11, "avg_chunk_chars": 162.1,"top3_hit_rate": 0.75 }
  },
  "best_strategy":     "sentence",
  "tradeoff_analysis": "Sentence-level chunking preserves complete atomic facts..."
}

llm_calls.jsonl  (one line per API call)
{"stage":"ANSWERS_GENERATED","query_id":"q1","timestamp":"2026-05-23T10:01:22+00:00",
 "provider":"anthropic","model":"claude-sonnet-4-5",
 "prompt_hash":"a3f9c12b4d8e7f01",
 "input_artifacts":["artifacts/retrieval.json"],
 "output_artifact":"artifacts/answers.json"}

7. Validation
Run python validate.py after the pipeline. It performs 47 deterministic checks across all artifacts:

============================================================
Mini-RAG Pipeline Validation
============================================================

[1] Required artifact files
  +  artifacts/chunks.json exists
  +  artifacts/retrieval.json exists
  +  artifacts/answers.json exists
  +  artifacts/eval.json exists

[2] JSON validity
  +  All artifact files are valid JSON

[3] chunks.json  (19 chunks)
  +  At least 1 chunk produced
  +  All chunks have required fields: chunk_id, doc_title, section, text, start_char, end_char
  +  All chunk_ids are unique

[4] retrieval.json  (4 records)
  +  All 4 queries have retrieval results
  +  Each query has >= 3 retrieved chunks
  +  All retrieval scores are numeric

[5] answers.json  (4 records)
  +  All 4 queries have answers
  +  All answer_labels use controlled vocabulary
  +  grounded_answer entries have >= 1 citation
  +  All citations match format [Doc Title §chunk_N]
  +  All cited chunk_ids were actually retrieved

[6] eval.json
  +  Required keys present
  +  summary.total_queries == 4
  +  hits + partial_hits + misses == total_queries
  +  All retrieval_status values use controlled vocabulary

[7] Optional artifacts
  +  grounding_check.json present and valid JSON
  +  chunking_comparison.json present and valid JSON

============================================================
  Passed: 47   Failed: 0
  +  VALIDATION PASSED
============================================================

8. REST API  (api.py)
The optional REST API server lets you query the pipeline interactively over HTTP without editing any files.

Starting the server
# With make
make api

# Directly
uvicorn api:app --reload --port 8000

On startup the server loads kb/, chunks all documents, and builds the TF-IDF index once. It prints:
[API] Index ready: 19 chunks from 4 documents

Endpoints
Method	Endpoint	Description
POST	/answer	Answer a free-text question with citations
GET	/health	Liveness check — returns chunk and document counts
GET	/chunks	List all indexed chunks for debugging (limit param)

Example — POST /answer
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "How long does the mouse battery last?"}'

Response:
{
  "answer_label": "grounded_answer",
  "answer": "The WidgetPro X1 battery lasts 90 days on a single AA. [WidgetPro X1 Mouse §chunk_2]",
  "citations": ["[WidgetPro X1 Mouse §chunk_2]"],
  "top_chunks": [
    { "rank": 1, "chunk_id": "chunk_2", "score": 0.412381, ... },
    ...
  ]
}

Example — GET /health
curl http://localhost:8000/health

{ "status": "ok", "chunks_loaded": 19, "docs_loaded": 4 }

9. Controlled Vocabularies
Both pipeline.py and validate.py share these constants. Any value outside these sets causes validation to fail.

Constant	Allowed Values
ANSWER_LABELS	grounded_answer  |  insufficient_context  |  conflicting_context
RETRIEVAL_STATUSES	hit  |  partial_hit  |  miss
CITATION_RE (format)	[Doc Title §chunk_N]  — validated by regex

10. Replacing the Knowledge Base
The pipeline is fixture-independent. To use different documents:

•	Drop any .txt files with Title: / Section: headers into kb/
•	Update queries.json with questions relevant to the new documents
•	Make sure expected_doc_titles values match the Title: values exactly
•	Run make fresh — clean artifacts, re-index, re-evaluate

The evaluator may replace your kb/ and queries.json with equivalent fixtures.
Because the pipeline is fixture-independent, it will handle any valid replacement
without code changes — just re-run python pipeline.py.

11. Troubleshooting
Error	Cause	Fix
FileNotFoundError: 'queries.json'	queries.json missing from project root	Create it — see Section 3, Step 3
FileNotFoundError: No .txt files in 'kb/'	kb/ folder missing or empty	Create kb/ and add .txt files with Title:/Section: headers
LLM: unavailable	ANTHROPIC_API_KEY not set or anthropic not installed	Set env var or install: pip install anthropic
VALIDATION FAILED: citation format invalid	Answer citations do not match [Doc Title §chunk_N]	Check that Title: in kb/*.txt matches citation prefix exactly
VALIDATION FAILED: chunk_id not retrieved	LLM cited a chunk that was not in the top-5 results	Usually a hallucination — check llm_calls.jsonl and re-run
Invalid stage jump error	Pipeline code was modified and a stage fires out of order	Do not reorder the _advance() calls in pipeline.py

12. Quick Reference Card

COMPLETE WORKFLOW IN 4 COMMANDS:

  1.  pip install -r requirements.txt
  2.  set ANTHROPIC_API_KEY=sk-ant-YOUR_KEY   (Windows) or export ... (Mac/Linux)
  3.  python pipeline.py                      generates all artifacts
  4.  python validate.py                      confirms 47 checks pass

  Or in one step:   make all
  Clean + re-run:   make fresh
  Start REST API:   make api

