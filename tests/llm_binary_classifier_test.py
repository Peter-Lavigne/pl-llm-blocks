from textwrap import dedent

from pl_tiny_clients.fetch_chat_completion import MODEL_DEEPSEEK_V3_2, PROVIDER_DEEPSEEK

from pl_llm_blocks.llm_binary_classifier import (
    ClassificationResult,
    llm_binary_classifier,
)
from pl_llm_blocks.testing.tests import run_binary_classification_test

from .constants import PYTEST_INTEGRATION_MARKER

pytestmark = PYTEST_INTEGRATION_MARKER

BIRD_SUBJECTS = [
    "sparrow",
    "eagle",
    "penguin",
    "parrot",
    "hummingbird",
    "ostrich",
    "swan",
    "owl",
    "flamingo",
    "peacock",
    "crow",
    "dove",
    "pelican",
    "albatross",
    "kingfisher",
]

NON_BIRD_SUBJECTS = [
    "cat",
    "elephant",
    "dolphin",
    "butterfly",
    "snake",
    "turtle",
    "fish",
    "squirrel",
    "wolf",
    "zebra",
    "bear",
    "frog",
    "crab",
    "lizard",
    "ant",
]


def _bird_classifier(subject: str) -> ClassificationResult:
    prompt = dedent("""
        You are given a subject name. Your task is to determine whether the subject is a bird.

        Subject: "{subject}"
    """).format(subject=subject)

    return llm_binary_classifier(
        prompt, model=MODEL_DEEPSEEK_V3_2, provider=PROVIDER_DEEPSEEK
    )


def test_classifies_prompts_into_true_false_values() -> None:
    # Classification Results:
    # True Positives: 15
    # False Positives: 0
    # True Negatives: 15
    # False Negatives: 0
    # Precision: 1.000
    # Recall: 1.000
    # Accuracy: 1.000
    # Average cost per run: $0.0001
    # Total cost of test: $0.0026
    # Average time per run: 0.44 seconds
    # Total time of test: 13.27 seconds

    run_binary_classification_test(
        classifier=_bird_classifier,
        true_values=BIRD_SUBJECTS,
        false_values=NON_BIRD_SUBJECTS,
    )
