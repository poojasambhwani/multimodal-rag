"""Query encoders for search: the fp32 PyTorch model the indexes were built with, or ONNX copies of it.

Only the *query* side changes; the stored chunk and slide embeddings stay as they are.
"""
from pathlib import Path

MODEL_ID = "BAAI/bge-small-en-v1.5"


def export_onnx(out_dir, qconfig):
    """Save an fp32 ONNX copy of the model to out_dir, plus a dynamically quantized INT8 one
    (onnx/model_{qint8|quint8}_<qconfig>.onnx). qconfig: "avx2", "avx512", "avx512_vnni" or "arm64"."""
    from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model

    model = SentenceTransformer(MODEL_ID, backend="onnx")
    model.save_pretrained(out_dir)  # the quantizer only writes the .onnx file, so config + tokenizer must be there first
    export_dynamic_quantized_onnx_model(model, qconfig, out_dir)


def load_encoder(kind, onnx_dir=None, qconfig=None):
    """-> encode(text) returning a 1-D float32 vector. kind: "torch", "onnx" or "onnx-int8"."""
    from sentence_transformers import SentenceTransformer

    if kind == "torch":
        model = SentenceTransformer(MODEL_ID, backend="torch")
    elif kind == "onnx":
        model = SentenceTransformer(onnx_dir, backend="onnx", model_kwargs={"file_name": "onnx/model.onnx"})
    elif kind == "onnx-int8":
        # signed (qint8) or unsigned (quint8) depending on the quantization config, e.g. avx2 -> quint8
        int8 = next(Path(onnx_dir, "onnx").glob(f"model_q*int8_{qconfig}.onnx"))
        model = SentenceTransformer(onnx_dir, backend="onnx", model_kwargs={"file_name": f"onnx/{int8.name}"})
    else:
        raise ValueError(f"unknown encoder kind: {kind}")
    return model.encode
