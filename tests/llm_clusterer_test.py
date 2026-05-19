from pl_tiny_clients.fetch_chat_completion import (
    MODEL_DEEPSEEK_V3_2,
    PROVIDER_DEEPSEEK,
)

from pl_llm_blocks.testing.tests import run_clustering_test

from .constants import PYTEST_INTEGRATION_MARKER

pytestmark = PYTEST_INTEGRATION_MARKER


CONTEXT = "Cluster the following country names by continent using the standard set of continents (Asia, etc.)."

VALUES = [
    "France",
    "Japan",
    "Brazil",
    "Egypt",
    "Australia",
    "Canada",
    "Germany",
    "India",
    "Argentina",
    "Nigeria",
    "New Zealand",
    "Mexico",
    "Italy",
    "China",
    "Chile",
    "South Africa",
]

EXPECTED_CLUSTERS = [
    ["France", "Germany", "Italy"],
    ["Japan", "India", "China"],
    ["Brazil", "Argentina", "Chile"],
    ["Egypt", "Nigeria", "South Africa"],
    ["Australia", "New Zealand"],
    ["Canada", "Mexico"],
]


def test_clusters_values_into_groups() -> None:
    # Clustering Results:
    # Clusters: 6 (expected 6)
    # Matches expected partition: True
    # Total cost: $0.0015
    # Total time: 70.09 seconds

    run_clustering_test(
        context=CONTEXT,
        values=VALUES,
        expected_clusters=EXPECTED_CLUSTERS,
        model=MODEL_DEEPSEEK_V3_2,
        provider=PROVIDER_DEEPSEEK,
    )
