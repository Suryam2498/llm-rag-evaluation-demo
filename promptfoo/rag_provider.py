"""Promptfoo custom provider: exposes the RAG pipeline so Promptfoo can run regression and safety tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_eval.llm import get_llm  # noqa: E402
from rag_eval.pipeline import RagPipeline  # noqa: E402
from rag_eval.retriever import Retriever, load_chunks  # noqa: E402

_pipeline = None


def call_api(prompt, options, context):
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline(get_llm(), Retriever(load_chunks(ROOT / "data" / "kb")))
    return {"output": _pipeline.answer(prompt).answer}
