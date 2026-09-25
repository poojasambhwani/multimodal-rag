"""Retrieval evaluation: recall@k and MRR over labelled questions in eval/queries.csv.

A question may have several valid windows (e.g. the slide and the moment it's explained): one CSV
row per window, sharing the same id. A retrieved moment is a hit if it overlaps any window of its
question, with `tol` seconds of slack at either end.
"""
import csv
from pathlib import Path

QUERIES_CSV = Path(__file__).resolve().parent.parent / "eval" / "queries.csv"


def secs(t):
    """'2:33' or '153' -> 153.0"""
    m, _, s = str(t).strip().rpartition(":")
    return int(m or 0) * 60 + float(s)


def load_queries(path=QUERIES_CSV):
    """-> [{id, question, type, windows: [(talk_id, start, end), ...]}, ...]"""
    by_id = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = by_id.setdefault(r["id"], {"id": r["id"], "question": r["question"], "type": r["type"], "windows": []})
            q["windows"].append((r["talk_id"], secs(r["start"]), secs(r["end"])))
    return list(by_id.values())


def first_hit_rank(moments, windows, tol=15):
    """1-based rank of the first moment overlapping any window, or None."""
    for rank, m in enumerate(moments, 1):
        if any(m["talk_id"] == talk and m["start"] < end + tol and start - tol < m["end"] for talk, start, end in windows):
            return rank
    return None


def evaluate(search_fn, queries, ks=(1, 5, 10), tol=15):
    """search_fn(question) -> ranked moments (at least max(ks) of them). Returns
    {"all": {...}, <type>: {...}} with recall@k, MRR and n for each group."""
    ranks = [(q["type"], first_hit_rank(search_fn(q["question"]), q["windows"], tol)) for q in queries]

    def summarize(rs):
        out = {f"R@{k}": sum(1 for r in rs if r and r <= k) / len(rs) for k in ks}
        out["MRR"] = sum(1 / r for r in rs if r) / len(rs)
        out["n"] = len(rs)
        return out

    result = {"all": summarize([r for _, r in ranks])}
    for t in sorted({t for t, _ in ranks}):
        result[t] = summarize([r for tt, r in ranks if tt == t])
    return result
