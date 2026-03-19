#!/usr/bin/env python3
"""
CLI入口 - 表格字段分类工具
Usage:
    python main.py --table "表名" --field "字段名" [--description "字段描述"]
"""

import argparse
import os
import json
from dotenv import load_dotenv
from classification_agent import ClassificationAgent
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
from classification_agent.llm.openai_wrapper import OpenAILLM

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description='Table field hierarchical classification agent')
    parser.add_argument('--table', required=True, help='Table name')
    parser.add_argument('--field', required=True, help='Field name')
    parser.add_argument('--description', help='Field description')
    parser.add_argument('--categories', help='JSON file with custom categories')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--no-multiple', action='store_true', help='Disable multiple labels')
    parser.add_argument('--output', help='Output file for result')

    args = parser.parse_args()

    # Initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable must be set")
        exit(1)

    llm = OpenAILLM(api_key=api_key, model=model)

    # Load categories
    if args.categories:
        with open(args.categories, 'r', encoding='utf-8') as f:
            categories = json.load(f)
    else:
        categories = DEFAULT_DATA_CATEGORIES

    # Initialize agent
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=categories,
        confidence_threshold=args.threshold,
        allow_multiple=not args.no_multiple
    )

    # Classify
    input_data = {
        "table_name": args.table,
        "field_name": args.field,
        "field_description": args.description
    }

    result = agent.classify(input_data)

    # Print result
    print("\n" + "="*60)
    print("Classification Result")
    print("="*60)
    print(f"Input: {args.table}.{args.field}")
    if args.description:
        print(f"Description: {args.description}")
    print()
    print(f"Final labels: {result['final_labels']}")
    print(f"Final confidence: {result['final_confidence']:.2f}")
    print()
    print("Feature Analysis:")
    print(f"  Table keywords: {result['feature_analysis']['table_name_keywords']}")
    print(f"  Field keywords: {result['feature_analysis']['field_name_keywords']}")
    print(f"  Dominant source: {result['feature_analysis']['dominant_source']}")
    print()
    print("Verification:")
    print(f"  Cross validation: {result['verification_result']['cross_validation_note']}")
    print()

    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Full result saved to {args.output}")


if __name__ == "__main__":
    main()
