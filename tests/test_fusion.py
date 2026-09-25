from mmrag.search import fuse


def hit(talk, start, end, source, rank):
    return {"talk_id": talk, "start": start, "end": end, "source": source, "rank": rank, "text": ""}


def test_overlapping_hits_from_other_indexes_add_support():
    speech = [hit("A", 30, 60, "speech", 1)]
    slides = [hit("A", 40, 140, "slide_text", 1)]
    [m] = fuse([speech, slides])  # the slide overlaps the chosen chunk, so it isn't returned separately
    assert (m["start"], m["end"]) == (30, 60)
    assert m["sources"] == ["slide_text", "speech"]
    assert m["score"] == 1 / 61 + 1 / 61


def test_no_chaining_into_long_spans():
    speech = [hit("A", 0, 30, "speech", 1), hit("A", 30, 60, "speech", 2)]
    slides = [hit("A", 20, 40, "slide_text", 1)]
    spans = [(m["start"], m["end"]) for m in fuse([speech, slides])]
    assert spans == [(0, 30), (30, 60)]


def test_agreement_across_indexes_beats_a_single_top_hit():
    speech = [hit("A", 0, 30, "speech", 1), hit("B", 0, 30, "speech", 2)]
    slides = [hit("B", 10, 50, "slide_text", 2)]
    assert fuse([speech, slides], k=1)[0]["talk_id"] == "B"


def test_single_index_keeps_its_own_order():
    speech = [hit("A", 0, 30, "speech", 1), hit("A", 60, 90, "speech", 2), hit("B", 0, 30, "speech", 3)]
    assert [(m["talk_id"], m["start"]) for m in fuse([speech])] == [("A", 0), ("A", 60), ("B", 0)]
