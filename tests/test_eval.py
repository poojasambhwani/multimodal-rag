import os
import tempfile

from mmrag.eval import evaluate, evaluate_answers, first_hit_rank, load_queries, secs


def m(talk, start, end):
    return {"talk_id": talk, "start": start, "end": end}


def test_secs():
    assert secs("2:33") == 153 and secs("153") == 153 and secs("0:05") == 5


def test_load_queries_groups_windows_and_splits_dev_test():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "q.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("id,question,talk_id,start,end,type\n"
                    "a,Q a,PD8WGF,0:10,0:40,both\n"
                    "a,Q a,PD8WGF,1:00,1:30,both\n"
                    'b,"Q, with a comma",CB7MBQ,2:00,2:30,slide\n'
                    "u,Unanswerable?,,,,unanswerable\n")
        dev, test = load_queries(path, split="dev"), load_queries(path, split="test")
    assert [q["id"] for q in dev] == ["a", "u"] and dev[0]["windows"] == [("PD8WGF", 10, 40), ("PD8WGF", 60, 90)]
    assert dev[1]["windows"] == []
    assert [q["question"] for q in test] == ["Q, with a comma"]


def test_hit_needs_same_talk_and_overlap_within_tolerance():
    windows = [("A", 100, 130)]
    assert first_hit_rank([m("B", 100, 130), m("A", 140, 160)], windows) == 2  # 140 < 130 + 15
    assert first_hit_rank([m("A", 150, 170)], windows) is None                  # 150 > 130 + 15
    assert first_hit_rank([m("A", 0, 30), m("A", 120, 125)], [("A", 0, 1), ("A", 500, 510)]) == 1


def test_evaluate_recall_and_mrr_by_type():
    queries = [
        {"question": "q1", "type": "slide", "windows": [("A", 0, 30)]},
        {"question": "q2", "type": "speech", "windows": [("A", 300, 330)]},
    ]
    results = {"q1": [m("A", 0, 30)], "q2": [m("A", 900, 930), m("A", 300, 330)]}
    trap = {"question": "trap", "type": "unanswerable", "windows": []}
    r = evaluate(lambda q: results[q], queries + [trap], ks=(1, 5))  # the trap is skipped
    assert r["all"] == {"R@1": 0.5, "R@5": 1.0, "MRR": 0.75, "n": 2}
    assert r["speech"]["R@1"] == 0.0 and r["slide"]["MRR"] == 1.0


def test_evaluate_answers_separates_false_refusals_from_retrieval_misses():
    hit, miss = m("A", 0, 30), m("A", 900, 930)
    replies = {
        "good":    {"status": "OK", "moments": [hit], "citations": [hit], "cost": 0.001, "answer": ""},
        "refused": {"status": "INSUFFICIENT_CONTEXT", "moments": [hit], "citations": [], "cost": 0.001, "answer": ""},
        "no_ctx":  {"status": "INSUFFICIENT_CONTEXT", "moments": [miss], "citations": [], "cost": 0.001, "answer": ""},
        "trap":    {"status": "INSUFFICIENT_CONTEXT", "moments": [miss], "citations": [], "cost": 0.001, "answer": ""},
    }
    queries = [{"id": q, "question": q, "type": "slide", "windows": [("A", 0, 30)]} for q in ("good", "refused", "no_ctx")]
    queries.append({"id": "trap", "question": "trap", "type": "unanswerable", "windows": []})
    s, rows = evaluate_answers(lambda q: replies[q], queries)
    assert s["context hit"] == 2 / 3 and s["answered"] == 1 / 3 and s["cited correct"] == 1 / 3
    assert s["false refusals"] == 0.5  # "refused" had the answer in context; "no_ctx" is a retrieval miss
    assert s["trap refusals"] == 1.0 and s["n"] == 3 and s["n traps"] == 1 and len(rows) == 4
