"""
Example: Batch evaluation on data.csv

This example demonstrates:
1. How to read data.csv with ground truth labels
2. How to run batch evaluation
3. How to calculate accuracy metrics
"""
import os
from tqdm import tqdm
from dotenv import load_dotenv
from classification_agent import ClassificationAgent
from classification_agent.llm.openai_wrapper import OpenAILLM
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
from classification_agent.utils.data_reader import load_data_csv_with_gt, TableFieldInputWithGroundTruth
from classification_agent.types.schemas import EvaluationResult

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


def run_batch_evaluation(
    agent: ClassificationAgent,
    data: list[TableFieldInputWithGroundTruth],
    limit: int | None = None,
) -> EvaluationResult:
    """Run batch evaluation on a list of data with ground truth"""
    total_samples = len(data[:limit]) if limit else len(data)
    exact_match_count = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_sample_results = []

    print(f"Running batch evaluation on {total_samples} samples...")

    for i, item in enumerate(tqdm(data[:limit], total=total_samples)):
        result = agent.classify(
            field_input=item.input,
            ground_truth_data_items=item.ground_truth_data_items,
        )

        # Get evaluation from result state
        # Since we provided ground truth, evaluation is already done by the graph
        if "evaluation" in result:
            eval_result = result["evaluation"]
            if eval_result:
                per_sample = eval_result["per_sample_results"][0]
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

    result = EvaluationResult(
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

    return result


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

    # Run batch evaluation
    # Set limit to a smaller number for quick testing
    # Remove limit to evaluate entire dataset
    LIMIT = None  # 10 for testing
    result = run_batch_evaluation(agent, data, limit=LIMIT)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Result:")
    print("=" * 60)
    print(f"Total samples: {result['total_samples']}")
    print(f"Exact matches: {result['exact_match_count']}")
    print(f"Exact match accuracy: {result['exact_match_accuracy']:.4f}")
    print(f"Macro precision: {result['macro_precision']:.4f}")
    print(f"Macro recall: {result['macro_recall']:.4f}")
    print(f"Macro F1: {result['macro_f1']:.4f}")
    print(f"\nBreakdown:")
    print(f"  True positives: {result['total_true_positives']}")
    print(f"  False positives: {result['total_false_positives']}")
    print(f"  False negatives: {result['total_false_negatives']}")

    # Show some examples of wrong predictions
    wrong_samples = [
        (idx, res) for idx, res in enumerate(result['per_sample_results'])
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


if __name__ == "__main__":
    main()
