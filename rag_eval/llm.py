"""LLM access layer. Amazon Bedrock (Converse API) in real runs, a fake LLM in offline tests."""
from __future__ import annotations

import os
from typing import Callable, Protocol


class LLM(Protocol):
    def generate(self, prompt: str, system: str = "") -> str: ...


class BedrockLLM:
    """Calls a model through the Bedrock Converse API (works across model families)."""

    def __init__(self, model_id: str | None = None, region: str | None = None, max_tokens: int = 512):
        import boto3  # imported lazily so offline tests do not need AWS configuration

        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.max_tokens = max_tokens
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def generate(self, prompt: str, system: str = "") -> str:
        kwargs: dict = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": self.max_tokens, "temperature": 0.0},
        }
        if system:
            kwargs["system"] = [{"text": system}]
        response = self.client.converse(**kwargs)
        parts = response["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts).strip()


class FakeLLM:
    """Deterministic stand-in used by the offline unit tests."""

    def __init__(self, responder: Callable[[str, str], str]):
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    def generate(self, prompt: str, system: str = "") -> str:
        self.calls.append((prompt, system))
        return self.responder(prompt, system)


def get_llm(model_id: str | None = None) -> LLM:
    provider = os.getenv("LLM_PROVIDER", "bedrock").lower()
    if provider == "bedrock":
        return BedrockLLM(model_id=model_id)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
