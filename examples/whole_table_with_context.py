"""
Example: Whole table classification with context analysis, separate nodes, no RAG.

This example demonstrates:
1. Table-level context analysis (enabled)
2. Separate feature_analysis + preliminary_classification nodes (normal mode, not fast mode)
3. No RAG enabled
4. Classify all fields in one LLM call
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
    print("Whole Table Classification")
    print("  - With table context analysis: ENABLED")
    print("  - Feature + Classification nodes: SEPARATE")
    print("  - RAG: DISABLED")
    print("=" * 60)

    # Read all fields from data.csv
    csv_path = "data.csv"
    print(f"\nLoading fields from: {csv_path}")
    inputs, _ = load_data_csv(csv_path, include_ground_truth=False)
    print(f"Loaded {len(inputs)} fields total")

    if not inputs:
        print("No data loaded, exiting")
        return

    # Group by the first table (all from same table in data.csv example)
    first_table_name = inputs[0]["table_name"].split(" - ")[0]
    table_cn_name = inputs[0]["table_name"].split(" - ")[-1] if " - " in inputs[0]["table_name"] else None
    table_inputs = [inp for inp in inputs if first_table_name in inp["table_name"]]

    print(f"\nProcessing table: {first_table_name}")
    if table_cn_name:
        print(f"Table Chinese name: {table_cn_name}")
    print(f"Number of fields: {len(table_inputs)}")

    # Create agent with:
    # - enable_table_context = True  (do table context analysis)
    # - fast_mode = False            (keep feature and classification separate)
    # - enable_rag = False           (no RAG)
    print("\nInitializing ClassificationAgent...")
    print("  enable_table_context = True (table context analysis enabled)")
    print("  fast_mode = False (feature_analysis and preliminary_classification are separate nodes)")
    print("  enable_rag = False (RAG disabled)")

    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=DEFAULT_DATA_CATEGORIES,
        confidence_threshold=0.7,
        allow_multiple=True,
        enable_table_context=True,  # <- Enable table context analysis
        fast_mode=False,            # <- Keep separate nodes (not combined)
        enable_rag=False,           # <- Disable RAG
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
                print(f"\nFound ground truth for {len([g for g in gts if g])} fields")
    except Exception as e:
        print(f"\nFailed to read ground truth: {e}")

    # Classify all fields in ONE LLM call
    print(f"\nStarting classification...")
    print("1. context_analysis node will analyze the whole table first")
    print("2. Then batch classify all fields in one LLM call")
    print("   (feature_analysis + preliminary_classification for all fields)")

    import time
    start_time = time.time()

    results = agent.classify_table(
        fields=table_inputs,
        table_chinese_name=table_cn_name,
        ground_truth_list=ground_truth_list,
    )

    elapsed = time.time() - start_time
    print(f"\n✅ Done! Total time: {elapsed:.2f} seconds")
    print(f"   Average per field: {elapsed/len(table_inputs):.2f} seconds")

    # Print table context analysis result
    if len(results) > 0 and "table_context_analysis" in results[0]:
        ctx = results[0]["table_context_analysis"]
        print(f"\n📊 Table Context Analysis Result:")
        print(f"   Inferred purpose: {ctx['inferred_purpose']}")
        print(f"   Key business concepts: {', '.join(ctx['key_business_concepts'])}")
        print(f"   Overall data category: {ctx['overall_data_category']}")

    # Print classification summary
    print("\n" + "=" * 60)
    print("Classification Summary:")
    print("=" * 60)
    for i, (inp, result) in enumerate(zip(table_inputs, results)):
        print(f"  {i+1:2d}. {inp['field_name']:<30} -> {result['final_labels']}")

    # Print evaluation if we have ground truth
    if ground_truth_list:
        print("\n" + "=" * 60)
        print("Evaluation:")
        print("=" * 60)
        total_labeled = sum(1 for gt in ground_truth_list if gt)
        if total_labeled > 0:
            exact_matches = sum(
                1 for res in results
                if res.get("evaluation") and res["evaluation"]["per_sample_results"]
                and res["evaluation"]["per_sample_results"][0]["exact_match"]
            )
            accuracy = exact_matches / total_labeled
            print(f"Total labeled fields: {total_labeled}")
            print(f"Exact matches: {exact_matches}")
            print(f"Exact match accuracy: {accuracy:.4f}")

    # Export results
    print("\nExporting results to outputs/...")
    csv_path_out, md_path = export_batch_results(
        outputs_dir="outputs",
        inputs=table_inputs,
        results=results,
        evaluation=None,
        base_name="whole_table_with_context",
    )
    print(f"Results exported:")
    print(f"  CSV: {csv_path_out}")
    print(f"  Markdown: {md_path}")

    print("\n✅ Example completed successfully!")
    print("\nSummary of this configuration:")
    print("- Table context analysis: ✓ Enabled")
    print("- Feature analysis and preliminary classification: ✓ Separate nodes")
    print("- RAG: ✗ Disabled")
    print("- All fields classified in: ✓ ONE LLM call (massive speedup)")


if __name__ == "__main__":
    main()
