"""Retrieval evaluation: recall@k and MRR over labelled questions in eval/queries.csv.

A question may have several valid windows (e.g. the slide and the moment it's explained): one CSV
row per window, sharing the same id. A retrieved moment is a hit if it overlaps any window of its
question, with `tol` seconds of slack at either end.
"""
import csv
from pathlib import Path

QUERIES_CSV = Path(__file__).resolve().parent.parent / "eval" / "queries.csv"
# Questions on these talks are the dev set, analysed freely while designing. Questions on the
# other talks are the held-out test set, scored once with the final configuration.
DEV_TALKS = {"PD8WGF", "NHNPMY"}


def secs(t):
    """'2:33' or '153' -> 153.0"""
    m, _, s = str(t).strip().rpartition(":")
    return int(m or 0) * 60 + float(s)


def load_queries(path=QUERIES_CSV, split=None):
    """-> [{id, question, type, windows: [(talk_id, start, end), ...]}, ...]
    split: "dev" (all windows on DEV_TALKS), "test" (the rest) or None (every question)."""
    by_id = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = by_id.setdefault(r["id"], {"id": r["id"], "question": r["question"], "type": r["type"], "windows": []})
            if r["talk_id"]:  # unanswerable (trap) questions have no window
                q["windows"].append((r["talk_id"], secs(r["start"]), secs(r["end"])))
    queries = list(by_id.values())
    if split:
        queries = [q for q in queries if all(t in DEV_TALKS for t, _, _ in q["windows"]) == (split == "dev")]
    return queries


def first_hit_rank(moments, windows, tol=15):
    """1-based rank of the first moment overlapping any window, or None."""
    for rank, m in enumerate(moments, 1):
        if any(m["talk_id"] == talk and m["start"] < end + tol and start - tol < m["end"] for talk, start, end in windows):
            return rank
    return None


def evaluate(search_fn, queries, ks=(1, 5, 10), tol=15):
    """search_fn(question) -> ranked moments (at least max(ks) of them). Returns
    {"all": {...}, <type>: {...}} with recall@k, MRR and n for each group. Trap questions are skipped."""
    ranks = [(q["type"], first_hit_rank(search_fn(q["question"]), q["windows"], tol)) for q in queries if q["windows"]]

    def summarize(rs):
        out = {f"R@{k}": sum(1 for r in rs if r and r <= k) / len(rs) for k in ks}
        out["MRR"] = sum(1 / r for r in rs if r) / len(rs)
        out["n"] = len(rs)
        return out

    result = {"all": summarize([r for _, r in ranks])}
    for t in sorted({t for t, _ in ranks}):
        result[t] = summarize([r for tt, r in ranks if tt == t])
    return result


def evaluate_answers(answer_fn, queries, tol=15):
    """answer_fn(question) -> mmrag.answer.answer() output. Returns (summary, per-question rows).

    For answerable questions: was a correct moment in the context, did the model answer, and did it
    cite a correct moment? A refusal while a correct moment *was* in the context is a false refusal.
    For trap questions (no window): did the model refuse?
    """
    rows = []
    for q in queries:
        r, w = answer_fn(q["question"]), q["windows"]
        rows.append({"id": q["id"], "type": q["type"], "status": r["status"],
                     "in_context": bool(w) and first_hit_rank(r["moments"], w, tol) is not None,
                     "cited_correct": bool(w) and first_hit_rank(r["citations"], w, tol) is not None,
                     "cost": r["cost"], "answer": r["answer"]})

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    answerable = [x for x, q in zip(rows, queries) if q["windows"]]
    traps = [x for x, q in zip(rows, queries) if not q["windows"]]
    summary = {
        "context hit": mean([x["in_context"] for x in answerable]),
        "answered": mean([x["status"] == "OK" for x in answerable]),
        "cited correct": mean([x["status"] == "OK" and x["cited_correct"] for x in answerable]),
        "false refusals": mean([x["status"] != "OK" for x in answerable if x["in_context"]]),
        "trap refusals": mean([x["status"] == "INSUFFICIENT_CONTEXT" for x in traps]),
        "cost per answer": mean([x["cost"] for x in rows]),
        "n": len(answerable),
        "n traps": len(traps),
    }
    return summary, rows
