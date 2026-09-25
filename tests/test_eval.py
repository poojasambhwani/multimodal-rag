from mmrag.eval import evaluate, first_hit_rank, secs


def m(talk, start, end):
    return {"talk_id": talk, "start": start, "end": end}


def test_secs():
    assert secs("2:33") == 153 and secs("153") == 153 and secs("0:05") == 5


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
    r = evaluate(lambda q: results[q], queries, ks=(1, 5))
    assert r["all"] == {"R@1": 0.5, "R@5": 1.0, "MRR": 0.75, "n": 2}
    assert r["speech"]["R@1"] == 0.0 and r["slide"]["MRR"] == 1.0
