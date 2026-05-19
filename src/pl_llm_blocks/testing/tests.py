from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from pl_tiny_clients.current_time import current_time
from pl_tiny_clients.fetch_chat_completion import (
    PROVIDER_DEEPSEEK,
)
from pl_user_io.display import display

from pl_llm_blocks.llm_binary_classifier import (
    ClassificationResult as BinaryClassificationResult,
)
from pl_llm_blocks.llm_classifier import (
    ClassificationResult as ClassifierClassificationResult,
)
from pl_llm_blocks.llm_clusterer import ClusteringResult, llm_clusterer


def run_binary_classification_test[T](
    classifier: Callable[[T], BinaryClassificationResult],
    true_values: list[T],
    false_values: list[T],
) -> None:
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    false_positives_details: list[tuple[T, str]] = []
    false_negatives_details: list[tuple[T, str]] = []
    total_cost: float = 0.0
    total_time: float = 0.0

    start = current_time()
    display("Classifying true values...")
    with ThreadPoolExecutor() as executor:
        positive_futures = {
            executor.submit(classifier, subject): subject for subject in true_values
        }
        for future in as_completed(positive_futures):
            subject = positive_futures[future]
            classification, reasoning, cost = future.result()
            if cost is not None:
                total_cost += cost
            if classification:
                true_positives += 1
            else:
                false_negatives += 1
                false_negatives_details.append((subject, reasoning))

    display("Classifying false values...")
    with ThreadPoolExecutor() as executor:
        negative_futures = {
            executor.submit(classifier, subject): subject for subject in false_values
        }
        for future in as_completed(negative_futures):
            subject = negative_futures[future]
            classification, reasoning, cost = future.result()
            if cost is not None:
                total_cost += cost
            if not classification:
                true_negatives += 1
            else:
                false_positives += 1
                false_positives_details.append((subject, reasoning))
    end = current_time()
    total_time = end - start

    if false_positives > 0:
        display("\nFalse Positives (incorrectly classified as true):")
        for subject, reasoning in false_positives_details:
            display(f"\n  {subject}: {reasoning}")

    if false_negatives > 0:
        display("\nFalse Negatives (incorrectly classified as false):")
        for subject, reasoning in false_negatives_details:
            display(f"\n  {subject}: {reasoning}")

    total_runs = true_positives + false_negatives + true_negatives + false_positives
    precision = (
        f"{true_positives / (true_positives + false_positives):.3f}"
        if true_positives + false_positives > 0
        else "N/A"
    )
    recall = (
        f"{true_positives / (true_positives + false_negatives):.3f}"
        if true_positives + false_negatives > 0
        else "N/A"
    )
    accuracy = (true_positives + true_negatives) / total_runs

    display()
    display("    # Classification Results:")
    display(f"    # True Positives: {true_positives}")
    display(f"    # False Positives: {false_positives}")
    display(f"    # True Negatives: {true_negatives}")
    display(f"    # False Negatives: {false_negatives}")
    display(f"    # Precision: {precision}")
    display(f"    # Recall: {recall}")
    display(f"    # Accuracy: {accuracy:.3f}")
    display(f"    # Average cost per run: ${total_cost / total_runs:.4f}")
    display(f"    # Total cost of test: ${total_cost:.4f}")
    display(f"    # Average time per run: {total_time / total_runs:.2f} seconds")
    display(f"    # Total time of test: {total_time:.2f} seconds")
    display()
    display("Test complete. Consider documenting these values.")


def run_classification_test(
    classifier: Callable[[str], ClassifierClassificationResult],
    classifications: list[str],
    labeled_examples: list[tuple[str, int]],
) -> None:
    """Run the classifier on each (value, expected_index) and report accuracy and cost."""
    correct = 0
    total_cost: float = 0.0
    mismatches: list[tuple[str, int, int, str]] = []  # value, expected, got, reasoning

    start = current_time()
    display("Classifying values...")
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(classifier, value): (value, expected_index)
            for value, expected_index in labeled_examples
        }
        for future in as_completed(futures):
            value, expected_index = futures[future]
            result = future.result()
            if result.cost is not None:
                total_cost += result.cost
            if result.classification_index == expected_index:
                correct += 1
            else:
                mismatches.append(
                    (
                        value,
                        expected_index,
                        result.classification_index,
                        result.thought_process,
                    )
                )
            print(".", end="", flush=True)
    end = current_time()
    total_time = end - start
    total_runs = len(labeled_examples)

    if mismatches:
        display("\nMismatches (value -> expected index, got index):")
        for value, expected, got, reasoning in mismatches:
            expected_label = classifications[expected]
            got_label = classifications[got] if 0 <= got < len(classifications) else "?"
            display(
                f"\n  {value!r}: expected {expected} ({expected_label}), got {got} ({got_label})"
            )
            display(f"    Reasoning: {reasoning}")

    accuracy = correct / total_runs if total_runs else 0.0
    display()
    display("    # Classification Results:")
    display(f"    # Correct: {correct} / {total_runs}")
    display(f"    # Accuracy: {accuracy:.3f}")
    display(f"    # Average cost per run: ${total_cost / total_runs:.4f}")
    display(f"    # Total cost of test: ${total_cost:.4f}")
    display(f"    # Average time per run: {total_time / total_runs:.2f} seconds")
    display(f"    # Total time of test: {total_time:.2f} seconds")
    display()
    display("Test complete. Consider documenting these values.")


def _clustering_partition(clusters: list[list[str]]) -> list[frozenset[str]]:
    """Return the partition of items as a list of frozensets for comparison."""
    return [frozenset(c) for c in clusters]


def _clusterings_equivalent(actual: list[list[str]], expected: list[list[str]]) -> bool:
    """Return True if both clusterings form the same partition of the same set of items."""
    actual_part = sorted(_clustering_partition(actual), key=len)
    expected_part = sorted(_clustering_partition(expected), key=len)
    return actual_part == expected_part


def _assert_clustering_result(
    result: ClusteringResult,
    values: list[str],
    expected_clusters: list[list[str]],
    total_time: float,
) -> None:
    clusters = result.clusters
    equivalent = _clusterings_equivalent(clusters, expected_clusters)

    # Sanity: every value appears exactly once
    all_items = [x for c in clusters for x in c]
    counts = Counter(all_items)
    duplicates = [k for k, v in counts.items() if v > 1]
    missing = [v for v in values if v not in counts]

    if duplicates:
        display(f"\n  Duplicates in clusters: {duplicates}")
    if missing:
        display(f"\n  Missing from clusters: {missing}")

    if not equivalent:
        display("\n  Expected clusters:")
        for i, c in enumerate(expected_clusters):
            display(f"    {i + 1}: {c}")
        display("  Actual clusters:")
        for i, c in enumerate(clusters):
            display(f"    {i + 1}: {c}")

    cost_str = f"${result.cost:.4f}" if result.cost is not None else "N/A"
    display()
    display("    # Clustering Results:")
    display(f"    # Clusters: {len(clusters)} (expected {len(expected_clusters)})")
    display(f"    # Matches expected partition: {equivalent}")
    display(f"    # Total cost: {cost_str}")
    display(f"    # Total time: {total_time:.2f} seconds")
    display()
    display("Test complete. Consider documenting these values.")

    assert not duplicates, f"Duplicates in clusters: {duplicates}"
    assert not missing, f"Missing from clusters: {missing}"
    assert equivalent, "Clustering did not match expected partition"


def run_clustering_test(
    context: str,
    values: list[str],
    expected_clusters: list[list[str]],
    model: str,
    provider: str = PROVIDER_DEEPSEEK,
) -> None:
    """Run the clusterer and report whether result matches expected grouping, plus cost and time."""
    start = current_time()
    display("Clustering values...")
    result = llm_clusterer(
        context=context,
        values=values,
        model=model,
        provider=provider,
    )
    end = current_time()
    total_time = end - start
    _assert_clustering_result(result, values, expected_clusters, total_time)


def run_clustering_test_with_clusterer(
    clusterer: Callable[[list[str]], ClusteringResult],
    values: list[str],
    expected_clusters: list[list[str]],
) -> None:
    """Run a clusterer function and report whether result matches expected grouping, plus cost and time."""
    start = current_time()
    display("Clustering values...")
    result = clusterer(values)
    end = current_time()
    total_time = end - start
    _assert_clustering_result(result, values, expected_clusters, total_time)
