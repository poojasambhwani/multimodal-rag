"""Grounded answers: GPT-4o-mini answers only from the retrieved moments and cites them by number."""
import json

from mmrag.pipeline import corpus
from mmrag.search import search

MODEL = "gpt-4o-mini"
PRICE_IN, PRICE_OUT = 0.15e-6, 0.60e-6  # gpt-4o-mini list price per token when written; check before relying on it

SYSTEM_PROMPT = """You answer questions about recorded conference talks using ONLY the numbered context moments.
Each moment gives what the speaker said and/or the slide on screen (its text and a description).
Rules:
- Use only facts stated in the context, never outside knowledge.
- Cite the moment numbers that support the answer.
- If the context does not contain the answer, set status to INSUFFICIENT_CONTEXT and say briefly what is missing.
Reply with a JSON object: {"status": "OK" or "INSUFFICIENT_CONTEXT", "answer": "...", "citations": [moment numbers]}"""


def mmss(t):
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"


def build_context(moments):
    """Numbered context blocks, one per moment: title, time range, what was said, what the slide showed."""
    blocks = []
    for i, m in enumerate(moments, 1):
        lines = [f'[{i}] "{m["title"]}" {mmss(m["start"])}-{mmss(m["end"])}']
        if "speech" in m["texts"]:
            lines.append(f'Said: {m["texts"]["speech"]}')
        if "slide_text" in m["texts"]:
            lines.append(f'Slide: {m["texts"]["slide_text"]}')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def parse_reply(content, moments):
    """Model JSON -> {status, answer, citations: [{n, talk_id, title, start, end, url}]}.
    Citation numbers outside 1..len(moments) are dropped, never trusted."""
    try:
        reply = json.loads(content)
    except json.JSONDecodeError:
        return {"status": "ERROR", "answer": content, "citations": []}
    nums = [n for n in reply.get("citations", []) if isinstance(n, int) and 1 <= n <= len(moments)]
    citations = []
    for n in dict.fromkeys(nums):  # de-duplicate, keep order
        m = moments[n - 1]
        citations.append({"n": n, "talk_id": m["talk_id"], "title": m["title"], "start": m["start"], "end": m["end"],
                          "url": f'{corpus()[m["talk_id"]]["video_url"]}#t={int(m["anchor"])}'})
    return {"status": reply.get("status", "ERROR"), "answer": reply.get("answer", ""), "citations": citations}


def answer(question, k=10, client=None, system_prompt=SYSTEM_PROMPT):
    """Retrieve the top-k moments, ask the model, return the parsed reply plus the moments, usage and cost."""
    from openai import OpenAI  # imported here so the pure functions above are testable without it

    moments = search(question, k=k)
    resp = (client or OpenAI()).chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context moments:\n\n{build_context(moments)}\n\nQuestion: {question}"},
        ],
    )
    out = parse_reply(resp.choices[0].message.content, moments)
    out["moments"] = moments
    out["usage"] = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens}
    out["cost"] = resp.usage.prompt_tokens * PRICE_IN + resp.usage.completion_tokens * PRICE_OUT
    return out
