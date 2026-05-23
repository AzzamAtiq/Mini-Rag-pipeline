#!/usr/bin/env python3
"""
Validation script for the mini-RAG pipeline.
Usage:
    python validate.py
    make validate

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ── Controlled vocabularies (must match pipeline.py) ──────────────────────────
ANSWER_LABELS      = {"grounded_answer", "insufficient_context", "conflicting_context"}
RETRIEVAL_STATUSES = {"hit", "partial_hit", "miss"}
CITATION_RE        = re.compile(r'\[([^\]]+?) §(chunk_\d+)\]')

REQUIRED_ARTIFACTS = [
    "artifacts/chunks.json",
    "artifacts/retrieval.json",
    "artifacts/answers.json",
    "artifacts/eval.json",
]

OPTIONAL_ARTIFACTS = [
    "artifacts/grounding_check.json",
    "artifacts/chunking_comparison.json",
]

# ── Helpers ────────────────────────────────────────────────────────────────────
_passed = 0
_failed = 0


def ok(msg: str) -> bool:
    global _passed
    _passed += 1
    print(f"  ✓  {msg}")
    return True


def fail(msg: str) -> bool:
    global _failed
    _failed += 1
    print(f"  ✗  {msg}")
    return False


def check(condition: bool, msg_pass: str, msg_fail: str | None = None) -> bool:
    return ok(msg_pass) if condition else fail(msg_fail or msg_pass)


# ── Validation sections ────────────────────────────────────────────────────────

def validate_artifacts_exist() -> bool:
    print("\n[1] Required artifact files")
    all_ok = True
    for path in REQUIRED_ARTIFACTS:
        c = check(Path(path).exists(), f"{path} exists", f"{path} MISSING")
        all_ok = all_ok and c
    return all_ok


def validate_json_parseable() -> bool:
    print("\n[2] JSON validity")
    all_ok = True
    for path in REQUIRED_ARTIFACTS:
        p = Path(path)
        if not p.exists():
            continue
        try:
            json.loads(p.read_text(encoding="utf-8"))
            ok(f"{path} is valid JSON")
        except json.JSONDecodeError as exc:
            fail(f"{path} JSON parse error: {exc}")
            all_ok = False
    return all_ok


def validate_chunks(chunks: list[dict]) -> bool:
    print(f"\n[3] chunks.json  ({len(chunks)} chunks)")
    all_ok = True
    required_fields = ["chunk_id", "doc_title", "section", "text", "start_char", "end_char"]
    c = check(len(chunks) > 0, "At least 1 chunk produced", "No chunks found")
    all_ok = all_ok and c

    ids_seen: set[str] = set()
    for chunk in chunks:
        cid = chunk.get("chunk_id", "<missing>")
        for field in required_fields:
            c = check(
                field in chunk,
                f"chunk {cid} has field '{field}'",
                f"chunk {cid} MISSING field '{field}'",
            )
            all_ok = all_ok and c
        # No duplicate IDs
        c = check(
            cid not in ids_seen,
            f"chunk_id '{cid}' is unique",
            f"chunk_id '{cid}' is DUPLICATE",
        )
        all_ok = all_ok and c
        ids_seen.add(cid)
    return all_ok


def validate_retrieval(retrieval: list[dict], query_ids: set[str]) -> bool:
    n = len(query_ids)
    print(f"\n[4] retrieval.json  ({len(retrieval)} records)")
    all_ok = True

    c = check(
        len(retrieval) == n,
        f"All {n} queries have retrieval results",
        f"Expected {n} retrieval records, got {len(retrieval)}",
    )
    all_ok = all_ok and c

    retrieved_chunks: dict[str, set[str]] = {}  # query_id -> {chunk_id}
    for r in retrieval:
        qid = r.get("query_id", "<missing>")
        c = check(
            qid in query_ids,
            f"query_id '{qid}' matches queries.json",
            f"query_id '{qid}' not found in queries.json",
        )
        all_ok = all_ok and c

        top_k = r.get("top_k", [])
        c = check(
            len(top_k) >= 3,
            f"query {qid} has >= 3 retrieved chunks ({len(top_k)})",
            f"query {qid} has only {len(top_k)} chunk(s) — need >= 3",
        )
        all_ok = all_ok and c

        for item in top_k:
            score = item.get("score")
            c = check(
                isinstance(score, (int, float)),
                f"query {qid} chunk {item.get('chunk_id')} has numeric score",
                f"query {qid} chunk {item.get('chunk_id')} score is not numeric: {score!r}",
            )
            all_ok = all_ok and c

        retrieved_chunks[qid] = {item["chunk_id"] for item in top_k}

    return all_ok


def validate_answers(
    answers: list[dict],
    retrieval: list[dict],
    query_ids: set[str],
) -> bool:
    n = len(query_ids)
    print(f"\n[5] answers.json  ({len(answers)} records)")
    all_ok = True

    c = check(
        len(answers) == n,
        f"All {n} queries have answers",
        f"Expected {n} answers, got {len(answers)}",
    )
    all_ok = all_ok and c

    # Build retrieval map: query_id -> set of retrieved chunk_ids
    ret_map: dict[str, set[str]] = {
        r["query_id"]: {item["chunk_id"] for item in r.get("top_k", [])}
        for r in retrieval
    }

    for ans in answers:
        qid   = ans.get("query_id", "<missing>")
        label = ans.get("answer_label", "")

        # Label in controlled vocabulary
        c = check(
            label in ANSWER_LABELS,
            f"query {qid} answer_label '{label}' is valid",
            f"query {qid} answer_label '{label}' not in {ANSWER_LABELS}",
        )
        all_ok = all_ok and c

        if label == "grounded_answer":
            citations = ans.get("citations", [])
            c = check(
                len(citations) >= 1,
                f"query {qid} grounded_answer has >= 1 citation",
                f"query {qid} grounded_answer has NO citations",
            )
            all_ok = all_ok and c

            retrieved = ret_map.get(qid, set())
            for cit in citations:
                m = CITATION_RE.match(cit.strip())
                c = check(
                    m is not None,
                    f"query {qid} citation '{cit}' has valid format",
                    f"query {qid} citation '{cit}' has INVALID format",
                )
                all_ok = all_ok and c
                if m:
                    chunk_id = m.group(2)
                    c = check(
                        chunk_id in retrieved,
                        f"query {qid} cited chunk_id '{chunk_id}' was retrieved",
                        f"query {qid} cited chunk_id '{chunk_id}' was NOT retrieved",
                    )
                    all_ok = all_ok and c

    return all_ok


def validate_eval(eval_data: dict, query_ids: set[str]) -> bool:
    print(f"\n[6] eval.json")
    all_ok = True
    n = len(query_ids)

    c = check("queries"  in eval_data, "eval.json has 'queries' key",  "eval.json MISSING 'queries'")
    all_ok = all_ok and c
    c = check("summary" in eval_data, "eval.json has 'summary' key", "eval.json MISSING 'summary'")
    all_ok = all_ok and c

    if "summary" in eval_data:
        s = eval_data["summary"]
        for field in ["top3_hit_rate", "total_queries", "hits", "partial_hits", "misses"]:
            c = check(field in s, f"summary has '{field}'", f"summary MISSING '{field}'")
            all_ok = all_ok and c
        c = check(
            s.get("total_queries") == n,
            f"summary.total_queries == {n}",
            f"summary.total_queries = {s.get('total_queries')} ≠ {n}",
        )
        all_ok = all_ok and c

        total = s.get("total_queries", 0)
        accounted = s.get("hits", 0) + s.get("partial_hits", 0) + s.get("misses", 0)
        c = check(
            accounted == total,
            f"hits + partial_hits + misses == total_queries ({accounted} == {total})",
            f"hits + partial_hits + misses ({accounted}) ≠ total_queries ({total})",
        )
        all_ok = all_ok and c

    if "queries" in eval_data:
        for r in eval_data["queries"]:
            status = r.get("retrieval_status", "")
            c = check(
                status in RETRIEVAL_STATUSES,
                f"query {r.get('query_id')} status '{status}' is valid",
                f"query {r.get('query_id')} status '{status}' not in {RETRIEVAL_STATUSES}",
            )
            all_ok = all_ok and c

    return all_ok


def validate_optional() -> None:
    print("\n[7] Optional artifacts")
    for path in OPTIONAL_ARTIFACTS:
        p = Path(path)
        if p.exists():
            try:
                json.loads(p.read_text(encoding="utf-8"))
                ok(f"{path} present and valid JSON")
            except json.JSONDecodeError as exc:
                fail(f"{path} present but invalid JSON: {exc}")
        else:
            print(f"  –  {path} not present (optional, skipping)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Mini-RAG Pipeline Validation")
    print("=" * 60)

    # Phase 0 – artifacts exist
    if not validate_artifacts_exist():
        print("\n  Cannot continue: required artifacts are missing.")
        print("  Run:  python pipeline.py")
        return 1

    # Phase 1 – JSON parseable
    validate_json_parseable()

    # Load data
    queries   = json.loads(Path("queries.json").read_text(encoding="utf-8"))
    query_ids = {q["query_id"] for q in queries}
    chunks    = json.loads(Path("artifacts/chunks.json").read_text(encoding="utf-8"))
    retrieval = json.loads(Path("artifacts/retrieval.json").read_text(encoding="utf-8"))
    answers   = json.loads(Path("artifacts/answers.json").read_text(encoding="utf-8"))
    eval_data = json.loads(Path("artifacts/eval.json").read_text(encoding="utf-8"))

    validate_chunks(chunks)
    validate_retrieval(retrieval, query_ids)
    validate_answers(answers, retrieval, query_ids)
    validate_eval(eval_data, query_ids)
    validate_optional()

    print("\n" + "=" * 60)
    print(f"  Passed: {_passed}   Failed: {_failed}")
    if _failed == 0:
        print("  ✓  VALIDATION PASSED")
        print("=" * 60)
        return 0
    else:
        print("  ✗  VALIDATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
