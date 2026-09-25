"""Fusion search over three indexes: speech chunks, slide text, and slide images (CLIP).

Scores from different embedding models aren't comparable (on this corpus CLIP cosines sit
around 0.2-0.35, bge around 0.55-0.8), so ranked lists are fused by position with Reciprocal
Rank Fusion. Hits in the same talk whose time ranges overlap are merged into one moment.
"""
SOURCES = ("speech", "slide_text", "slide_clip")
RRF_K = 60  # the standard RRF constant: softens the gap between rank 1 and rank 2


def retrieve(query, source, n=20):
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


def fuse(hit_lists, k=5, rrf_k=RRF_K):
    """Merge ranked hit lists into the top-k moments.

    Each hit is worth 1 / (rrf_k + rank). Hits in the same talk whose time ranges overlap merge
    into one moment (ranges that only touch don't, or contiguous speech chunks would chain into
    one). Within a moment each source counts once, via its best hit, so a long slide scene isn't
    rewarded just for overlapping several speech chunks.
    """
    hits = sorted((h for hl in hit_lists for h in hl), key=lambda h: (h["talk_id"], h["start"]))
    moments = []
    for h in hits:
        m = moments[-1] if moments else None
        if m and m["talk_id"] == h["talk_id"] and h["start"] < m["end"]:
            m["end"] = max(m["end"], h["end"])
            m["hits"].append(h)
        else:
            moments.append({"talk_id": h["talk_id"], "title": h.get("title"),
                            "start": h["start"], "end": h["end"], "hits": [h]})

    for m in moments:
        best = {}  # source -> (score, hit)
        for h in m.pop("hits"):
            s = 1 / (rrf_k + h["rank"])
            if s > best.get(h["source"], (0.0, None))[0]:
                best[h["source"]] = (s, h)
        m["score"] = sum(s for s, _ in best.values())
        m["sources"] = sorted(best)
        m["anchor"] = max(best.values(), key=lambda b: b[0])[1]["start"]  # where a citation link jumps to
        m["texts"] = {src: h["text"] for src, (_, h) in best.items()}
    return sorted(moments, key=lambda m: m["score"], reverse=True)[:k]


def search(query, k=5, n=20, sources=SOURCES):
    """Top-k moments for a question. `sources` selects indexes (used for ablations in evaluation)."""
    return fuse([retrieve(query, s, n) for s in sources], k=k)
