"""The ingestion pipeline as code: Pixeltable tables, views, computed columns and indexes.

build() creates whatever is missing and leaves existing tables and columns untouched, so it is safe to
run on a restored database. It does not migrate: changing a definition here means dropping that column
(and the columns computed from it) first. ingest(talk_id) downloads one talk and inserts it, which
computes everything for that talk: transcripts, slide detection, slide reading and index entries.
"""
import csv
import urllib.request
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPUS_CSV = REPO / "corpus.csv"
DATA_DIR = REPO.parent / "data"  # downloaded media lives next to the repo, never in it

WHISPER_MODEL = "small.en"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # the query encoder (mmrag.encoder) must use this same model
VLM_MODEL = "gpt-4o-mini"
DETECT_BOX = (75, 50, 1425, 800)  # x, y, w, h: area used to detect slide changes (1920x1080 FOSDEM layout)
SLIDE_BOX = (20, 20, 1534, 862)  # x, y, w, h: FOSDEM's full slide area, cropped for the VLM
SLIDE_PROMPT = (
    "This image is one slide from a recorded tech talk. Reply with a JSON object with two keys. "
    '"text": the words visible on the slide, verbatim, including handwritten words, labels and names '
    "(for dense screenshots or code, give the title and the most prominent words, at most about 100 words). "
    '"description": 2-3 sentences on what the slide shows (drawings, diagrams, screenshots) and its main point.'
)


@cache
def corpus():
    """talk_id -> row of corpus.csv (title, speakers, video_url, slides_url, event_url)."""
    with open(CORPUS_CSV, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def download(talk_id, data_dir=DATA_DIR):
    """Download a talk's video and slide PDF, skipping files already there. -> (video_path, slides_path)"""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    row, paths = corpus()[talk_id], []
    for url, ext in [(row["video_url"], "mp4"), (row["slides_url"], "pdf")]:
        dest = Path(data_dir) / f"{talk_id}.{ext}"
        if not dest.exists():
            tmp = dest.with_suffix(".part")  # an interrupted download never looks complete
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
        paths.append(dest)
    return tuple(paths)


def build():
    """Create any missing table, view, column or index. -> (talks, chunks, keyframes)"""
    import pixeltable as pxt
    from pixeltable.functions import openai, whisper
    from pixeltable.functions.audio import audio_splitter
    from pixeltable.functions.huggingface import sentence_transformer
    from pixeltable.functions.json import list_iterator

    from mmrag.udfs import parse_json

    pxt.create_dir("mmrag", if_exists="ignore")
    talks = pxt.create_table("mmrag.talks", {"talk_id": pxt.String, "title": pxt.String, "video": pxt.Video},
                             primary_key="talk_id", if_exists="ignore")
    talks.add_computed_column(audio=talks.video.extract_audio(format="flac"), if_exists="ignore")
    x, y, w, h = DETECT_BOX
    talks.add_computed_column(slide_proxy=talks.video.ffmpeg_filter(vf=f"crop={w}:{h}:{x}:{y},scale=640:-2,fps=2"),
                              if_exists="ignore")
    talks.add_computed_column(scenes=talks.slide_proxy.scene_detect_content(min_scene_len=4, threshold=15.0, delta_edges=1.0),
                              if_exists="ignore")

    # speech: ~30 s chunks cut at pauses -> Whisper -> text
    chunks = pxt.create_view("mmrag.audio_chunks", talks, if_exists="ignore",
                             iterator=audio_splitter(talks.audio, duration=30.0, min_silence_len=0.5))
    chunks.add_computed_column(asr=whisper.transcribe(chunks.audio_segment, model=WHISPER_MODEL), if_exists="ignore")
    chunks.add_computed_column(text=chunks.asr.text.astype(pxt.String), if_exists="ignore")

    # slides: one row per detected scene -> middle frame cropped to the slide -> VLM reads and describes it
    keyframes = pxt.create_view("mmrag.keyframes", talks, iterator=list_iterator(talks.scenes), if_exists="ignore")
    x, y, w, h = SLIDE_BOX
    keyframes.add_computed_column(
        slide=keyframes.video.extract_frame(timestamp=keyframes.start_time + keyframes.duration / 2).crop((x, y, x + w, y + h)),
        if_exists="ignore")
    keyframes.add_computed_column(
        vlm=openai.chat_completions(
            messages=[{"role": "user", "content": [{"type": "text", "text": SLIDE_PROMPT},
                                                   {"type": "image_url", "image_url": keyframes.slide}]}],
            model=VLM_MODEL,
            model_kwargs={"response_format": {"type": "json_object"}, "max_completion_tokens": 1500, "temperature": 0}),
        if_exists="ignore")
    keyframes.add_computed_column(vlm_json=parse_json(keyframes.vlm.choices[0].message.content), if_exists="ignore")
    keyframes.add_computed_column(slide_text=keyframes.vlm_json.text.astype(pxt.String), if_exists="ignore")
    keyframes.add_computed_column(caption=keyframes.vlm_json.description.astype(pxt.String), if_exists="ignore")
    keyframes.add_computed_column(slide_doc=keyframes.slide_text + "\n" + keyframes.caption, if_exists="ignore")

    embed = sentence_transformer.using(model_id=EMBED_MODEL)
    chunks.add_embedding_index("text", idx_name="speech_idx", embedding=embed, if_exists="ignore")
    keyframes.add_embedding_index("slide_doc", idx_name="slide_text_idx", embedding=embed, if_exists="ignore")
    return talks, chunks, keyframes


def ingest(talk_id, data_dir=DATA_DIR):
    """Download and insert one talk; Pixeltable computes everything for it. Per-cell errors are stored,
    not raised (retry them with recompute_columns(..., errors_only=True)). Returns the insert status,
    or None if the talk was already ingested."""
    talks, _, _ = build()
    if talks.where(talks.talk_id == talk_id).count():
        return None
    video, _ = download(talk_id, data_dir)
    return talks.insert([{"talk_id": talk_id, "title": corpus()[talk_id]["title"], "video": str(video)}],
                        on_error="ignore")
