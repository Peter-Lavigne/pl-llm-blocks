from textwrap import dedent
from typing import NamedTuple

from pl_mocks_and_fakes import MockInUnitTests, MockReason
from pl_tiny_clients.fetch_chat_completion import PROVIDER_ANTHROPIC
from pydantic import BaseModel, Field

from pl_llm_blocks.fetch_structured_llm import fetch_structured_llm


class LLMClassificationError(Exception):
    pass


class ClassificationResult(NamedTuple):
    classification_index: int
    thought_process: str
    cost: float | None


class ModelClassificationResult(BaseModel):
    classification_index: int = Field(
        ...,
        description="The 0-based index of the classification that the value falls into.",
    )


@MockInUnitTests(MockReason.UNINVESTIGATED)
def llm_classifier(
    context: str,
    value: str,
    classifications: list[str],
    model: str,
    provider: str = PROVIDER_ANTHROPIC,
) -> ClassificationResult:
    """
    Classify a value into exactly one of the given classifications using an LLM.

    Returns the index of the chosen classification, along with thought_process and cost.
    """
    if not classifications:
        msg = "classifications must be non-empty"
        raise ValueError(msg)

    categories_text = "\n".join(f"{i}: {c}" for i, c in enumerate(classifications))
    prompt = dedent("""
        {context}

        You are given a value and a list of classifications (each with a 0-based index).
        Your task is to choose exactly one classification that the value falls into.
        Return the index (0 to {max_index}) of that classification.

        Classifications:
        {categories_text}

        Value: "{value}"
    """).format(
        context=context,
        max_index=len(classifications) - 1,
        categories_text=categories_text,
        value=value,
    )

    response, thought_process, cost = fetch_structured_llm(
        prompt, ModelClassificationResult, model=model, provider=provider
    )

    idx = response.classification_index
    if idx < 0 or idx >= len(classifications):
        max_idx = len(classifications) - 1
        msg = f"LLM returned classification_index={idx}, which is out of range [0, {max_idx}]"
        raise LLMClassificationError(msg)

    return ClassificationResult(
        classification_index=idx,
        thought_process=thought_process,
        cost=cost,
    )
