"""Fusion search over three indexes: speech chunks, slide text, and slide images (CLIP).

Scores from different embedding models aren't comparable (on this corpus CLIP cosines sit
around 0.2-0.35, bge around 0.55-0.8), so ranked lists are fused by position with Reciprocal
Rank Fusion.
"""
SOURCES = ("speech", "slide_text", "slide_clip")
RRF_K = 60  # the standard RRF constant: softens the gap between rank 1 and rank 2


def retrieve(query, source, n=10):
    """Top-n hits from one index, as dicts: talk_id, title, start, end, text, source, rank."""
    import pixeltable as pxt  # imported here so fuse() can be tested without pixeltable

    if source == "speech":
        t = pxt.get_table("mmrag.audio_chunks")
        sim = t.text.similarity(string=query)
        cols = dict(start=t.segment_start, end=t.segment_end, text=t.text)
    else:
        t = pxt.get_table("mmrag.keyframes")
        sim = (t.slide_doc if source == "slide_text" else t.slide).similarity(string=query)
        cols = dict(start=t.start_time, end=t.start_time + t.duration, text=t.slide_text)
    df = t.order_by(sim, asc=False).limit(n).select(t.talk_id, t.title, **cols).collect().to_pandas()
    return [dict(row, source=source, rank=i + 1) for i, row in enumerate(df.to_dict("records"))]


def _overlaps(a, b):
    return a["talk_id"] == b["talk_id"] and a["start"] < b["end"] and b["start"] < a["end"]


def fuse(hit_lists, k=5, rrf_k=RRF_K):
    """Rank individual hits (one speech chunk or one slide scene) and return the top-k moments.

    A hit scores 1 / (rrf_k + rank), plus, for each *other* index, the best such score among that
    index's hits overlapping it in time. Hits are never merged into longer spans (an earlier
    version did, and overlapping chunks and slides chained into multi-minute "moments"). A hit
    overlapping an already-chosen result is skipped, so the top-k are distinct moments.
    """
    hits = [dict(h, rrf=1 / (rrf_k + h["rank"])) for hl in hit_lists for h in hl]
    for h in hits:
        support = {}  # other source -> its best hit overlapping h
        for o in hits:
            if o["source"] != h["source"] and _overlaps(h, o) and o["rrf"] > support.get(o["source"], {"rrf": 0})["rrf"]:
                support[o["source"]] = o
        h["score"] = h["rrf"] + sum(o["rrf"] for o in support.values())
        h["sources"] = sorted([h["source"], *support])
        h["texts"] = {h["source"]: h["text"], **{s: o["text"] for s, o in support.items()}}

    chosen = []
    for h in sorted(hits, key=lambda h: h["score"], reverse=True):  # stable: on ties, speech (listed first) wins
        if not any(_overlaps(h, c) for c in chosen):
            chosen.append(h)
            if len(chosen) == k:
                break
    return [{"talk_id": h["talk_id"], "title": h.get("title"), "start": h["start"], "end": h["end"],
             "anchor": h["start"], "score": h["score"], "sources": h["sources"], "texts": h["texts"]}
            for h in chosen]


def search(query, k=5, n=10, sources=SOURCES, rrf_k=RRF_K):
    """Top-k moments for a question. `sources`, `n` and `rrf_k` are exposed for evaluation."""
    return fuse([retrieve(query, s, n) for s in sources], k=k, rrf_k=rrf_k)
