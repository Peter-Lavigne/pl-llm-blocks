from textwrap import dedent
from typing import NamedTuple

from pl_mocks_and_fakes import MockInUnitTests, MockReason
from pl_tiny_clients.fetch_chat_completion import PROVIDER_ANTHROPIC

from pl_llm_blocks.llm_classifier import llm_classifier

# This is mostly a proof-of-concept. It's not particularly accurate and very slow. I don't recommend using it for non-trivial clustering tasks.


class ClusteringResult(NamedTuple):
    clusters: list[list[str]]
    thought_process: str
    cost: float | None


NONE_OF_THE_ABOVE = "None of the above (create new cluster)"


@MockInUnitTests(MockReason.UNINVESTIGATED)
def llm_clusterer(
    context: str,
    values: list[str],
    model: str,
    provider: str = PROVIDER_ANTHROPIC,
) -> ClusteringResult:
    """
    Cluster a list of values into groups using an LLM.

    Returns the clusters, combined thought_process, and total cost.
    """
    if not values:
        return ClusteringResult(clusters=[], thought_process="", cost=None)

    clusters: list[list[str]] = []
    thought_parts: list[str] = []
    total_cost: float = 0.0

    for value in values:
        if not clusters:
            clusters.append([value])
            thought_parts.append(f"Value {value!r}: (First item; started new cluster.)")
            continue

        categories = [f"Cluster: [{', '.join(cluster)}]" for cluster in clusters]
        categories.append(NONE_OF_THE_ABOVE)

        prompt = dedent(
            """\
            {context}

            Your task is to determine whether this value should be added to any of the existing clusters, or if it should start a new cluster on its own.

            Note: The below clusters are only partially complete, as you have not yet categorized the current value.
        """
        ).format(context=context)

        result = llm_classifier(
            context=prompt,
            value=value,
            classifications=categories,
            model=model,
            provider=provider,
        )
        thought_parts.append(f"Value {value!r}: {result.thought_process}")
        if result.cost is not None:
            total_cost += result.cost

        if result.classification_index == len(clusters):
            clusters.append([value])
        else:
            clusters[result.classification_index].append(value)

    thought_process = "\n\n---\n\n".join(thought_parts)
    return ClusteringResult(
        clusters=clusters,
        thought_process=thought_process,
        cost=total_cost or None,
    )
