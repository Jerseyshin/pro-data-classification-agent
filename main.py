#!/usr/bin/env python3
"""
CLI入口 - 整张表字段批量分类工具
Usage:
    python main.py --input data.csv [--output output] [--encoding GB18030] [--with-ground-truth]

Requirements:
    1. Uses context_analysis: 启用表级上下文分析（默认开启）
    2. Whole table prediction: 对整张表所有字段批量分类
    3. Separates feature_analysis and preliminary_classification: 不使用fast_mode，两个节点分离
    4. Output result documents: 输出 CSV 和 Markdown 结果文件
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from classification_agent import ClassificationAgent
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
from classification_agent.llm.openai_wrapper import OpenAILLM
from classification_agent.utils.data_reader import load_data_csv
from classification_agent.utils.result_exporter import export_batch_results

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description='Whole table hierarchical classification with LangGraph\n'
                    'Requirements: context_analysis enabled, separate feature_analysis and preliminary_classification'
    )
    parser.add_argument('--input', required=True, help='Input CSV file with table fields to classify')
    parser.add_argument('--input-encoding', default='utf-8-sig', help='Input file encoding (default: utf-8-sig, can use GB18030)')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for results')
    parser.add_argument('--output-base-name', default='classification_results', help='Base name for output files')
    parser.add_argument('--output-encoding', default='utf-8', help='Output file encoding (default: utf-8, can use GB18030)')
    parser.add_argument('--with-ground-truth', action='store_true', help='Input CSV contains ground truth, enable evaluation')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--allow-multiple', action='store_true', default=True, help='Allow multiple labels per field')
    parser.add_argument('--no-multiple', dest='allow_multiple', action='store_false', help='Disable multiple labels')
    parser.add_argument('--enable-rag', action='store_true', help='Enable RAG retrieval')

    args = parser.parse_args()

    # Initialize LLM from environment
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable must be set in .env")
        exit(1)

    llm = OpenAILLM(api_key=api_key, model=model, base_url=base_url)

    # Load default categories
    categories = DEFAULT_DATA_CATEGORIES

    # Initialize agent with requested configuration:
    # - enable_table_context = True  → uses context_analysis (requirement 1)
    # - fast_mode = False           → separates feature_analysis and preliminary_classification (requirement 3)
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=categories,
        confidence_threshold=args.confidence_threshold,
        allow_multiple=args.allow_multiple,
        enable_table_context=True,        # requirement 1: use context_analysis
        fast_mode=False,                  # requirement 3: separate feature and classification nodes
        enable_rag=args.enable_rag,
    )

    # Load input data from CSV
    print(f"Loading input from {args.input} (encoding={args.input_encoding})...")
    if args.with_ground_truth:
        inputs, ground_truth_list = load_data_csv(
            args.input,
            include_ground_truth=True,
            skip_empty_gt=True,
            encoding=args.input_encoding,
        )
    else:
        inputs, ground_truth_list = load_data_csv(
            args.input,
            include_ground_truth=False,
            encoding=args.input_encoding,
        )

    if not inputs:
        print("Error: No valid input fields loaded from CSV")
        exit(1)

    print(f"Loaded {len(inputs)} fields for classification")

    # Run whole table classification (requirement 2: whole table prediction)
    print("Starting classification...")
    results = agent.classify_table(
        fields=inputs,
        ground_truth_list=ground_truth_list,
    )

    print(f"Classification completed: {len(results)} fields processed")

    # Export results to CSV and Markdown (requirement 4: output result documents)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get overall evaluation if we have ground truth
    evaluation = None
    if len(results) > 0 and results[0].get("evaluation"):
        evaluation = results[0]["evaluation"]
        if evaluation:
            print(f"\nEvaluation results:")
            print(f"  Exact match accuracy: {evaluation['exact_match_accuracy']:.4f}")
            print(f"  Macro F1: {evaluation['macro_f1']:.4f}")

    # Export both CSV and Markdown
    csv_path, md_path = export_batch_results(
        outputs_dir=output_dir,
        inputs=inputs,
        results=results,
        evaluation=evaluation,
        base_name=args.output_base_name,
        encoding=args.output_encoding,
    )

    print(f"\nResults exported:")
    print(f"  CSV:  {csv_path}")
    print(f"  Markdown:  {md_path}")


if __name__ == "__main__":
    main()
