from mmrag.answer import build_context, parse_reply

MOMENTS = [
    {"talk_id": "PD8WGF", "title": "Talk", "start": 153, "end": 183, "anchor": 153,
     "texts": {"speech": "Paddler is our open source software.", "slide_text": "WHERE DOES PADDLER FIT IN?"}},
    {"talk_id": "NHNPMY", "title": "Other", "start": 140, "end": 187, "anchor": 140,
     "texts": {"slide_text": "Vanilla RAG struggles with: multi-hop reasoning"}},
]


def test_build_context_numbers_moments_and_labels_sources():
    ctx = build_context(MOMENTS)
    assert ctx.startswith('[1] "Talk" 2:33-3:03\nSaid: Paddler is our open source software.\nSlide: WHERE DOES PADDLER FIT IN?')
    assert '[2] "Other" 2:20-3:07\nSlide: Vanilla RAG' in ctx and "Said" not in ctx.split("[2]")[1]


def test_parse_reply_maps_citations_to_moments_with_links():
    r = parse_reply('{"status": "OK", "answer": "It is open source [1].", "citations": [1, 1, 7, "2"]}', MOMENTS)
    assert r["status"] == "OK"
    assert [c["n"] for c in r["citations"]] == [1]  # duplicates, out-of-range and non-int numbers are dropped
    assert r["citations"][0]["url"].endswith("PD8WGF-from_infrastructure_to_production_a_year_of_self-hosted_llms.mp4#t=153")


def test_parse_reply_survives_invalid_json():
    r = parse_reply("not json", MOMENTS)
    assert r["status"] == "ERROR" and r["citations"] == []
