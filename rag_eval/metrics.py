"""Evaluation metrics: deterministic checks plus LLM-as-judge scores."""
from __future__ import annotations

import json
import re

from .llm import LLM
from .pipeline import CANARY, REFUSAL

JSON_INSTRUCTION = 'Return ONLY JSON like {"score": <number from 0.0 to 1.0>, "reason": "<short reason>"}.'


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_score(text: str) -> float:
    """Extract a 0..1 score from a judge reply; fall back to 0.0 if nothing usable is found."""
    match = re.search(r"\{.*?\}", text, re.S)
    if match:
        try:
            return _clamp(float(json.loads(match.group(0))["score"]))
        except (ValueError, KeyError, TypeError):
            pass
    number = re.search(r"(?<![\d.])([01](?:\.\d+)?)(?![\d.])", text)
    return _clamp(float(number.group(1))) if number else 0.0


def judge_faithfulness(judge: LLM, answer: str, contexts: list[str]) -> float:
    """Share of the answer's claims that are supported by the retrieved context."""
    prompt = (
        "You are a strict evaluator. Decide what fraction of the claims in the ANSWER "
        "are directly supported by the CONTEXT. Unsupported or invented claims lower the score. "
        "1.0 means every claim is supported.\n\n"
        f"CONTEXT:\n{chr(10).join(contexts)}\n\nANSWER:\n{answer}\n\n{JSON_INSTRUCTION}"
    )
    return parse_score(judge.generate(prompt))


def judge_correctness(judge: LLM, question: str, answer: str, ground_truth: str) -> float:
    """How well the answer matches the reference answer."""
    prompt = (
        "You are a strict evaluator. Compare the ANSWER with the REFERENCE for the QUESTION. "
        "1.0 means it states the same facts, 0.0 means it is wrong or missing the key fact.\n\n"
        f"QUESTION: {question}\nREFERENCE: {ground_truth}\nANSWER: {answer}\n\n{JSON_INSTRUCTION}"
    )
    return parse_score(judge.generate(prompt))


def retrieval_hit(expected_source: str, sources: list[str]) -> bool:
    return expected_source in sources


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower().rstrip(".") in answer.lower()


def leaked_canary(answer: str) -> bool:
    return CANARY.lower() in answer.lower()
