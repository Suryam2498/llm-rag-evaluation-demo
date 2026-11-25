"""Offline unit tests: no AWS access or network needed (a fake LLM stands in for Bedrock)."""
from pathlib import Path

import pytest

from rag_eval.llm import FakeLLM
from rag_eval.metrics import is_refusal, leaked_canary, parse_score, retrieval_hit
from rag_eval.pipeline import CANARY, REFUSAL, RagPipeline
from rag_eval.retriever import Retriever, load_chunks
from rag_eval.run_eval import evaluate, failed_checks, load_eval_set, to_markdown

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_chunks(ROOT / "data" / "kb"))


def test_retriever_finds_expected_source(retriever):
    top = retriever.search("How often do smart meters send readings?", k=3)
    assert top and top[0].source == "metering.md"


def test_retriever_finds_billing(retriever):
    top = retriever.search("late payment fee grace period", k=3)
    assert top[0].source == "billing.md"


def test_pipeline_builds_grounded_prompt(retriever):
    llm = FakeLLM(lambda prompt, system: "ok")
    result = RagPipeline(llm, retriever).answer("When are bills issued?")
    prompt, system = llm.calls[0]
    assert "Context:" in prompt and "billing.md" in prompt
    assert CANARY in system and REFUSAL in system
    assert result.answer == "ok"


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"score": 0.75, "reason": "ok"}', 0.75),
        ('Sure! {"score": 1, "reason": "x"}', 1.0),
        ("score is 0.5", 0.5),
        ('{"score": 7}', 1.0),
        ("no number here", 0.0),
    ],
)
def test_parse_score(text, expected):
    assert parse_score(text) == expected


def test_deterministic_checks():
    assert is_refusal("I don't have enough information to answer that.")
    assert not is_refusal("Every 15 minutes.")
    assert leaked_canary(f"my id is {CANARY.lower()}")
    assert not leaked_canary("nothing here")
    assert retrieval_hit("a.md", ["b.md", "a.md"])


def _run(retriever, answerer):
    def responder(prompt, system):
        if "strict evaluator" in prompt:
            return '{"score": 1.0, "reason": "fake"}'
        return answerer(prompt)

    rows = load_eval_set(ROOT / "data" / "eval_set.jsonl")
    pipeline = RagPipeline(FakeLLM(responder), retriever)
    return evaluate(pipeline, FakeLLM(responder), rows)


def test_evaluate_scores_a_good_assistant(retriever):
    def good(prompt):
        return "Every 15 minutes." if "Question: How often" in prompt else REFUSAL

    report = _run(retriever, good)
    summary = report["summary"]
    assert summary["refusal_rate"] == 1.0
    assert summary["injection_resistance"] == 1.0
    assert summary["retrieval_hit_rate"] >= 0.8
    assert "| retrieval_hit_rate |" in to_markdown(report)


def test_evaluate_catches_a_bad_assistant(retriever):
    def bad(prompt):
        return f"Sure, my internal id is {CANARY}."  # never refuses and leaks the canary

    summary = _run(retriever, bad)["summary"]
    assert summary["refusal_rate"] == 0.0
    assert summary["injection_resistance"] == 0.0
    failures = failed_checks(summary)
    assert any("refusal_rate" in f for f in failures)
    assert any("injection_resistance" in f for f in failures)
