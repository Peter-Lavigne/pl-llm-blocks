from pl_tiny_clients.fetch_chat_completion import MODEL_DEEPSEEK_V3_2, PROVIDER_DEEPSEEK

from pl_llm_blocks.llm_classifier import (
    ClassificationResult,
    llm_classifier,
)
from pl_llm_blocks.testing.tests import run_classification_test

from .constants import PYTEST_INTEGRATION_MARKER

pytestmark = PYTEST_INTEGRATION_MARKER

CLASSIFICATIONS = ["bird", "mammal", "reptile", "fish", "insect"]

LABELED_EXAMPLES: list[tuple[str, int]] = [
    # (value, expected index into CLASSIFICATIONS)
    ("sparrow", 0),
    ("eagle", 0),
    ("penguin", 0),
    ("cat", 1),
    ("elephant", 1),
    ("dolphin", 1),
    ("snake", 2),
    ("turtle", 2),
    ("lizard", 2),
    ("salmon", 3),
    ("tuna", 3),
    ("goldfish", 3),
    ("butterfly", 4),
    ("ant", 4),
    ("bee", 4),
]


def _animal_classifier(value: str) -> ClassificationResult:
    return llm_classifier(
        "Which category of animal does the following animal fall into?",
        value,
        classifications=CLASSIFICATIONS,
        model=MODEL_DEEPSEEK_V3_2,
        provider=PROVIDER_DEEPSEEK,
    )


def test_classifies_values_into_given_classifications() -> None:
    # Classification Results:
    # Correct: 15 / 15
    # Accuracy: 1.000
    # Average cost per run: $0.0001
    # Total cost of test: $0.0020
    # Average time per run: 0.45 seconds
    # Total time of test: 6.81 seconds

    run_classification_test(
        classifier=_animal_classifier,
        classifications=CLASSIFICATIONS,
        labeled_examples=LABELED_EXAMPLES,
    )
