"""Pixeltable UDFs used by stored computed columns.

They live in an importable module, not a notebook cell, because Pixeltable stores a
computed column's UDF by its import path: a UDF defined in a notebook becomes
`__main__.<name>`, which no later session can resolve.
"""
import json

import pixeltable as pxt


@pxt.udf
def parse_json(s: str) -> pxt.Json:
    """Parse a model's JSON reply. A malformed reply (e.g. cut off at the token limit)
    keeps its raw text so the slide stays searchable, flagged with `parse_failed`."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"text": s, "description": "", "parse_failed": True}
