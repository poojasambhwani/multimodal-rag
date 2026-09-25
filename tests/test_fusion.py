from mmrag.search import fuse


def hit(talk, start, end, source, rank):
    return {"talk_id": talk, "start": start, "end": end, "source": source, "rank": rank, "text": ""}


def test_overlapping_hits_merge_and_each_source_counts_once():
    speech = [hit("A", 30, 60, "speech", 1), hit("A", 60, 90, "speech", 2)]
    slides = [hit("A", 40, 140, "slide_text", 1)]
    [m] = fuse([speech, slides])
    assert (m["start"], m["end"]) == (30, 140)
    assert m["sources"] == ["slide_text", "speech"]
    assert m["score"] == 1 / 61 + 1 / 61  # best speech hit + best slide hit; the 2nd speech hit adds nothing
    assert m["anchor"] == 30


def test_touching_chunks_stay_separate():
    speech = [hit("A", 0, 30, "speech", 1), hit("A", 30, 60, "speech", 2)]
    assert len(fuse([speech])) == 2


def test_agreement_across_indexes_beats_a_single_top_hit():
    speech = [hit("A", 0, 30, "speech", 1), hit("B", 0, 30, "speech", 2)]
    slides = [hit("B", 10, 50, "slide_text", 2)]
    assert fuse([speech, slides], k=1)[0]["talk_id"] == "B"
