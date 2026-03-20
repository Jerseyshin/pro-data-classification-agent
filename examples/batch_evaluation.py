"""
Example: Batch evaluation on data.csv

This example demonstrates:
1. How to read data.csv with ground truth labels
2. How to run batch evaluation
3. How to calculate accuracy metrics
4. How to export results to CSV/Markdown for inspection

Speed options:
- Set FAST_MODE=True in .env or pass fast_mode=True to agent:
  merges feature analysis + preliminary classification into one LLM call
  → ~33% faster (16 samples → 16 LLM calls instead of 32)
- Set MAX_CONCURRENCY=N > 1 for concurrent processing:
  multiple samples processed in parallel via async
  → even faster depending on your API rate limits
"""
import os
import asyncio
from typing import List, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
from classification_agent import ClassificationAgent
from classification_agent.llm.openai_wrapper import OpenAILLM
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
from classification_agent.utils.data_reader import load_data_csv_with_gt, TableFieldInputWithGroundTruth
from classification_agent.utils.result_exporter import export_batch_results
from classification_agent.types.schemas import EvaluationResult, ClassificationResult, TableFieldInput

# Load environment variables
load_dotenv()

# Get LLM credentials
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL", "MiniMax-M2.5")

# Initialize LLM
llm = OpenAILLM(
    api_key=api_key,
    model=model,
    base_url=base_url,
)


def run_batch_evaluation_sync(
    agent: ClassificationAgent,
    data: list[TableFieldInputWithGroundTruth],
    limit: int | None = None,
) -> tuple[EvaluationResult, list[TableFieldInput], list[ClassificationResult]]:
    """Run batch evaluation sequentially (synchronous, one sample at a time)"""
    total_samples = len(data[:limit]) if limit else len(data)
    exact_match_count = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_sample_results = []
    all_results: list[ClassificationResult] = []
    inputs: list[TableFieldInput] = []

    print(f"Running sequential batch evaluation on {total_samples} samples...")

    for item in tqdm(data[:limit], total=total_samples):
        result = agent.classify(
            field_input=item.input,
            ground_truth_data_items=item.ground_truth_data_items,
        )
        all_results.append(result)
        inputs.append(item.input)

        # Get evaluation from result state
        # Since we provided ground truth, evaluation is already done by the graph
        if "evaluation" in result and result["evaluation"]:
            per_sample = result["evaluation"]["per_sample_results"][0]
            exact_match_count += 1 if per_sample["exact_match"] else 0
            total_tp += len(per_sample["correct_predictions"])
            total_fp += len(per_sample["wrong_predictions"])
            total_fn += len(per_sample["missing_predictions"])
            per_sample_results.append(per_sample)

    # Aggregate metrics
    exact_match_accuracy = exact_match_count / total_samples if total_samples > 0 else 0.0

    # Macro precision
    if (total_tp + total_fp) == 0:
        macro_precision = 0.0
    else:
        macro_precision = total_tp / (total_tp + total_fp)

    # Macro recall
    if (total_tp + total_fn) == 0:
        macro_recall = 0.0
    else:
        macro_recall = total_tp / (total_tp + total_fn)

    # Macro F1
    if (macro_precision + macro_recall) == 0:
        macro_f1 = 0.0
    else:
        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    evaluation = EvaluationResult(
        total_samples=total_samples,
        exact_match_count=exact_match_count,
        exact_match_accuracy=exact_match_accuracy,
        total_true_positives=total_tp,
        total_false_positives=total_fp,
        total_false_negatives=total_fn,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_sample_results=per_sample_results,
    )

    return evaluation, inputs, all_results


async def classify_single_async(
    agent: ClassificationAgent,
    item: TableFieldInputWithGroundTruth,
) -> ClassificationResult:
    """Async wrapper for single classification"""
    # Note: agent.classify is sync, we just run it in a thread
    # to get concurrency on I/O waits (API network calls)
    import anyio
    result = await anyio.to_thread.run_sync(
        agent.classify,
        item.input,
        None,
        None,
        None,
        None,
        item.ground_truth_data_items,
    )
    return result


async def run_batch_evaluation_concurrent(
    agent: ClassificationAgent,
    data: list[TableFieldInputWithGroundTruth],
    limit: int | None = None,
    max_concurrency: int = 5,
) -> tuple[EvaluationResult, list[TableFieldInput], list[ClassificationResult]]:
    """Run batch evaluation with concurrent processing (multiple samples in parallel)

    Good for I/O bound workloads (waiting for API responses).
    Uses semaphore to limit concurrency to avoid hitting API rate limits.
    """
    from collections import deque

    data_slice = data[:limit] if limit else data
    total_samples = len(data_slice)
    semaphore = asyncio.Semaphore(max_concurrency)

    exact_match_count = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_sample_results = []
    all_results: list[ClassificationResult] = []
    inputs: list[TableFieldInput] = []

    print(f"Running concurrent batch evaluation on {total_samples} samples, max concurrency={max_concurrency}...")

    async def process_item(item: TableFieldInputWithGroundTruth) -> ClassificationResult:
        async with semaphore:
            return await classify_single_async(agent, item)

    # Create tasks with indices to preserve original order
    async def process_item_with_index(item: TableFieldInputWithGroundTruth, index: int) -> tuple[int, ClassificationResult]:
        result = await process_item(item)
        return index, result

    tasks = [process_item_with_index(item, idx) for idx, item in enumerate(data_slice)]

    # Run with progress bar
    results: List[tuple[int, ClassificationResult]] = []
    pbar = tqdm(total=total_samples)

    for task in asyncio.as_completed(tasks):
        index_result = await task
        results.append(index_result)
        pbar.update(1)

    pbar.close()

    # Collect results in original order
    all_results = [None] * total_samples
    inputs = [None] * total_samples
    for idx, result in results:
        all_results[idx] = result
        inputs[idx] = data_slice[idx].input

    # Now collect per-sample evaluation in original order
    for idx, (item, result) in enumerate(zip(data_slice, all_results)):

        if "evaluation" in result and result["evaluation"]:
            per_sample = result["evaluation"]["per_sample_results"][0]
            if per_sample["exact_match"]:
                exact_match_count += 1
            total_tp += len(per_sample["correct_predictions"])
            total_fp += len(per_sample["wrong_predictions"])
            total_fn += len(per_sample["missing_predictions"])
            per_sample_results.append(per_sample)

    # Aggregate metrics
    exact_match_accuracy = exact_match_count / total_samples if total_samples > 0 else 0.0

    # Macro precision
    if (total_tp + total_fp) == 0:
        macro_precision = 0.0
    else:
        macro_precision = total_tp / (total_tp + total_fp)

    # Macro recall
    if (total_tp + total_fn) == 0:
        macro_recall = 0.0
    else:
        macro_recall = total_tp / (total_tp + total_fn)

    # Macro F1
    if (macro_precision + macro_recall) == 0:
        macro_f1 = 0.0
    else:
        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    evaluation = EvaluationResult(
        total_samples=total_samples,
        exact_match_count=exact_match_count,
        exact_match_accuracy=exact_match_accuracy,
        total_true_positives=total_tp,
        total_false_positives=total_fp,
        total_false_negatives=total_fn,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_sample_results=per_sample_results,
    )

    return evaluation, inputs, all_results


def main():
    print("=" * 60)
    print("Batch Evaluation Example on data.csv")
    print("=" * 60)

    # Read data from data.csv
    csv_path = "data.csv"
    print(f"\nLoading data from: {csv_path}")
    data = load_data_csv_with_gt(csv_path, skip_empty_gt=True)
    print(f"Loaded {len(data)} samples with non-empty ground truth")

    if not data:
        print("No data loaded, exiting")
        return

    # Create agent (RAG optional, configure via .env)
    print("\nInitializing ClassificationAgent...")
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=DEFAULT_DATA_CATEGORIES,
        confidence_threshold=0.7,
        allow_multiple=True,
    )

    # Get configuration from environment or set defaults
    LIMIT = int(os.getenv("BATCH_LIMIT", "0")) or None  # 0 means no limit
    MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))

    # Create agent (RAG optional, configure via .env)
    print("\nInitializing ClassificationAgent...")
    # FAST_MODE can be enabled via .env: FAST_MODE=true
    # This merges feature analysis + preliminary classification into one LLM call
    # → ~33% faster processing
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=DEFAULT_DATA_CATEGORIES,
        confidence_threshold=0.7,
        allow_multiple=True,
    )

    # Run batch evaluation
    if MAX_CONCURRENCY > 1:
        # Concurrent processing (faster for multiple samples)
        import asyncio
        evaluation, inputs, all_results = asyncio.run(
            run_batch_evaluation_concurrent(agent, data, limit=LIMIT, max_concurrency=MAX_CONCURRENCY)
        )
    else:
        # Sequential processing
        evaluation, inputs, all_results = run_batch_evaluation_sync(agent, data, limit=LIMIT)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Result:")
    print("=" * 60)
    print(f"Total samples: {evaluation['total_samples']}")
    print(f"Exact matches: {evaluation['exact_match_count']}")
    print(f"Exact match accuracy: {evaluation['exact_match_accuracy']:.4f}")
    print(f"Macro precision: {evaluation['macro_precision']:.4f}")
    print(f"Macro recall: {evaluation['macro_recall']:.4f}")
    print(f"Macro F1: {evaluation['macro_f1']:.4f}")
    print(f"\nBreakdown:")
    print(f"  True positives: {evaluation['total_true_positives']}")
    print(f"  False positives: {evaluation['total_false_positives']}")
    print(f"  False negatives: {evaluation['total_false_negatives']}")

    # Show some examples of wrong predictions
    wrong_samples = [
        (idx, res) for idx, res in enumerate(evaluation['per_sample_results'])
        if not res['exact_match']
    ]
    if wrong_samples:
        print(f"\nFirst 5 misclassified examples:")
        for i, (idx, res) in enumerate(wrong_samples[:5]):
            sample = data[idx]
            print(f"\n  {i+1}. {sample.input['table_name']}.{sample.input['field_name']}")
            print(f"    Predicted: {res['predicted_data_items']}")
            print(f"    Ground truth: {res['ground_truth_data_items']}")
            if res['wrong_predictions']:
                print(f"    Wrong: {res['wrong_predictions']}")
            if res['missing_predictions']:
                print(f"    Missing: {res['missing_predictions']}")

    # Export results to CSV and Markdown
    print("\nExporting results...")
    csv_path, md_path = export_batch_results(
        outputs_dir="outputs",
        inputs=inputs,
        results=all_results,
        evaluation=evaluation,
        base_name="data_evaluation_results",
    )
    print(f"Results exported to:")
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
