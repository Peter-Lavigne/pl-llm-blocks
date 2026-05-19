from pl_tiny_clients.fetch_chat_completion import MODEL_HAIKU_4_5, PROVIDER_ANTHROPIC
from pydantic import BaseModel

from pl_llm_blocks.fetch_structured_llm import fetch_structured_llm

from .constants import PYTEST_INTEGRATION_MARKER

pytestmark = PYTEST_INTEGRATION_MARKER


def test_returns_specified_model() -> None:
    class QuestionAnswer(BaseModel):
        question: str
        answer: str

    result, _, _ = fetch_structured_llm(
        prompt="Extract a question and answer pair from the following text: 'What is the capital of France? The capital of France is Paris.'",
        return_type=QuestionAnswer,
        model=MODEL_HAIKU_4_5,
        provider=PROVIDER_ANTHROPIC,
    )

    assert result.question == "What is the capital of France?"
    assert result.answer == "The capital of France is Paris."


def test_raises_exception_on_invalid_response() -> None:
    class SimpleResponse(BaseModel):
        fakefield: int

    try:
        fetch_structured_llm(
            prompt="Provide a response that does not conform to the schema.",
            return_type=SimpleResponse,
            model=MODEL_HAIKU_4_5,
            provider=PROVIDER_ANTHROPIC,
        )
    except Exception as e:
        assert "fakefield" in str(
            e
        )  # Expecting a validation error mentioning 'fakefield'


def test_returns_cost() -> None:
    class SimpleResponse(BaseModel):
        message: str

    _, _, cost = fetch_structured_llm(
        prompt="Provide a simple message.",
        return_type=SimpleResponse,
        model=MODEL_HAIKU_4_5,
        provider=PROVIDER_ANTHROPIC,
    )

    assert cost is not None
    assert isinstance(cost, float)
    assert cost >= 0.0


def test_returns_thought_process() -> None:
    class QuestionAnswer(BaseModel):
        question: str
        answer: str

    _, thought_process, _ = fetch_structured_llm(
        prompt="Extract a question and answer pair from the following text: 'What is the capital of France? The capital of France is Paris.'",
        return_type=QuestionAnswer,
        model=MODEL_HAIKU_4_5,
        provider=PROVIDER_ANTHROPIC,
    )

    assert len(thought_process) > 0
