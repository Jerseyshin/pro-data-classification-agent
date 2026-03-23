"""
Example: Whole table classification with context analysis.

This example demonstrates:
1. How to load all fields from a table
2. How to enable table-level context analysis
3. How to classify all fields in one LLM call (much faster)
4. How to export results to CSV/Markdown
"""
import os
from dotenv import load_dotenv
from classification_agent import ClassificationAgent
from classification_agent.llm.openai_wrapper import OpenAILLM
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
from classification_agent.utils.data_reader import load_data_csv
from classification_agent.utils.result_exporter import export_batch_results

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


def main():
    print("=" * 60)
    print("Whole Table Classification Example with Context Analysis")
    print("=" * 60)
    print("\nThis example classifies ALL fields of a table in ONE LLM call")
    print("after doing table-level context analysis first.\n")

    # Read all fields from data.csv
    # We'll just take all fields from the same table as an example
    csv_path = "data.csv"
    print(f"Loading all fields from: {csv_path}")
    inputs, _ = load_data_csv(csv_path, include_ground_truth=False)
    print(f"Loaded {len(inputs)} fields total")

    if not inputs:
        print("No data loaded, exiting")
        return

    # Group by table (if there are multiple tables, process them one by one)
    # In this example, we just take the first table
    first_table_name = inputs[0]["table_name"].split(" - ")[0]
    table_inputs = [inp for inp in inputs if first_table_name in inp["table_name"]]
    table_cn_name = inputs[0]["table_name"].split(" - ")[-1] if " - " in inputs[0]["table_name"] else None

    print(f"\nProcessing table: {first_table_name}")
    if table_cn_name:
        print(f"Table chinese name: {table_cn_name}")
    print(f"Number of fields: {len(table_inputs)}")

    # Create agent
    # - enable_table_context=True (default): do table context analysis first
    # - fast_mode=True: combined feature+classification if single field mode
    #   (whole-table mode is already fast enough with one call)
    print("\nInitializing ClassificationAgent...")
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=DEFAULT_DATA_CATEGORIES,
        confidence_threshold=0.7,
        allow_multiple=True,
        enable_table_context=True,  # <- Enable table-level context analysis (default is True)
        fast_mode=True,
    )

    # Get ground truth if available
    import csv
    ground_truth_list = None
    try:
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)
            from classification_agent.utils.data_reader import parse_ground_truth
            gts = []
            for row in reader:
                if len(row) >= 6 and row[5].strip():
                    gts.append(parse_ground_truth(row[5].strip()))
                else:
                    gts.append([])
            if len(gts) == len(table_inputs):
                ground_truth_list = gts
                print(f"Found ground truth for {len([g for g in gts if g])} fields")
    except Exception:
        pass

    # Classify all fields in ONE LLM call
    print(f"\nStarting classification: {len(table_inputs)} fields in 1 LLM call...")
    import time
    start_time = time.time()

    results = agent.classify_table(
        fields=table_inputs,
        table_chinese_name=table_cn_name,
        ground_truth_list=ground_truth_list,
    )

    elapsed = time.time() - start_time
    print(f"\nDone! Classification took {elapsed:.2f} seconds")
    print(f"Average {elapsed/len(table_inputs):.2f} seconds per field")

    # Print summary
    print("\n" + "=" * 60)
    print("Classification Summary:")
    print("=" * 60)
    for i, (inp, result) in enumerate(zip(table_inputs, results)):
        print(f"  {i+1}. {inp['field_name']:<30} -> {result['final_labels']}")

    # Export results to CSV and Markdown
    if ground_truth_list:
        # Calculate aggregated evaluation metrics
        total_samples = len([g for g in ground_truth_list if g])
        exact_matches = sum(
            1 for res in results
            if res["evaluation"] and res["evaluation"]["exact_match_accuracy"] > 0.99
        )
        if total_samples > 0:
            accuracy = exact_matches / total_samples
            print(f"\nEvaluation on {total_samples} labeled fields:")
            print(f"Exact match accuracy: {accuracy:.4f}")

    print("\nExporting results...")
    csv_path_out, md_path = export_batch_results(
        outputs_dir="outputs",
        inputs=table_inputs,
        results=results,
        evaluation=None,  # We already have per-sample evaluation in results
        base_name="whole_table_results",
    )
    print(f"Results exported to:")
    print(f"  CSV: {csv_path_out}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
