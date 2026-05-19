from typing import NamedTuple

from pl_mocks_and_fakes import MockInUnitTests, MockReason
from pl_tiny_clients.fetch_chat_completion import PROVIDER_ANTHROPIC
from pydantic import BaseModel, Field

from pl_llm_blocks.fetch_structured_llm import fetch_structured_llm


class LLMClassificationError(Exception):
    pass


class ClassificationResult(NamedTuple):
    classification: bool
    thought_process: str
    cost: float | None


class ModelClassificationResult(BaseModel):
    classification: bool = Field(
        ..., description="The classification result: true or false."
    )


@MockInUnitTests(MockReason.UNINVESTIGATED)
def llm_binary_classifier(
    prompt: str, model: str, provider: str = PROVIDER_ANTHROPIC
) -> ClassificationResult:
    """Classify a binary classification prompt using an LLM. Classifies it into one of two categories: true or false."""
    response, thought_process, cost = fetch_structured_llm(
        prompt, ModelClassificationResult, model=model, provider=provider
    )

    return ClassificationResult(
        classification=response.classification,
        thought_process=thought_process,
        cost=cost,
    )
