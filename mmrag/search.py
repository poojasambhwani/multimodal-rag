"""Fusion search over two indexes: speech chunks and slide text (the VLM's reading + description).

Ranked lists are fused by position with Reciprocal Rank Fusion. A CLIP index on the slide images was
evaluated and dropped: on the dev set it didn't improve R@10 or MRR over speech + slide text.
"""
SOURCES = ("speech", "slide_text")
RRF_K = 60  # the standard RRF constant: softens the gap between rank 1 and rank 2


def retrieve(query, source, n=10, vec=None):
    """Top-n hits from one index, as dicts: talk_id, title, start, end, text, source, rank.
    vec: a precomputed query embedding (e.g. from an ONNX encoder); if None, the index's own model embeds it."""
    import pixeltable as pxt  # imported here so fuse() can be tested without pixeltable

    if source == "speech":
        t = pxt.get_table("mmrag.audio_chunks")
        col, cols = t.text, dict(start=t.segment_start, end=t.segment_end, text=t.text)
    else:
        t = pxt.get_table("mmrag.keyframes")
        col, cols = t.slide_doc, dict(start=t.start_time, end=t.start_time + t.duration, text=t.slide_doc)
    sim = col.similarity(string=query) if vec is None else col.similarity(vector=vec)
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


def search(query, k=5, n=10, sources=SOURCES, rrf_k=RRF_K, encode=None):
    """Top-k moments for a question. `sources`, `n` and `rrf_k` are exposed for evaluation.
    encode: optional text -> vector function (both indexes use the same model, so it runs once)."""
    vec = encode(query) if encode else None
    return fuse([retrieve(query, s, n, vec) for s in sources], k=k, rrf_k=rrf_k)
