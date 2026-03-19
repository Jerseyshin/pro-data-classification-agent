#!/usr/bin/env python3
"""
基础使用示例：使用预定义的层级分类体系进行分类
"""
import os
from dotenv import load_dotenv
from classification_agent import ClassificationAgent
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
from classification_agent.llm.openai_wrapper import OpenAILLM

load_dotenv()


def main():
    # Initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    llm = OpenAILLM(api_key=api_key, model=model)

    # Initialize agent with pre-defined hierarchical categories
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=DEFAULT_DATA_CATEGORIES,
        confidence_threshold=0.7,
        allow_multiple=True
    )

    # Example 1: The tricky deviceidtail case
    print("=" * 60)
    print("Example 1: deviceidtail in '框架应用市场下载结果日志'")
    print("=" * 60)

    input_data = {
        "table_name": "框架应用市场下载结果日志",
        "field_name": "deviceidtail",
        "field_description": "设备ID尾部"
    }

    result = agent.classify(input_data)

    print(f"\nInput:")
    print(f"  Table: {input_data['table_name']}")
    print(f"  Field: {input_data['field_name']}")
    print(f"  Description: {input_data['field_description']}")

    print(f"\nResult:")
    print(f"  Final labels: {result['final_labels']}")
    print(f"  Confidence: {result['final_confidence']:.2f}")
    print(f"  Dominant source: {result['feature_analysis']['dominant_source']}")
    print(f"  Cross-validation: {result['verification_result']['cross_validation_note']}")
    print()

    # Example 2: The clear network case
    print("=" * 60)
    print("Example 2: network field in any table")
    print("=" * 60)

    input_data = {
        "table_name": "框架应用市场下载结果日志",
        "field_name": "network",
        "field_description": None
    }

    result = agent.classify(input_data)

    print(f"\nInput:")
    print(f"  Table: {input_data['table_name']}")
    print(f"  Field: {input_data['field_name']}")

    print(f"\nResult:")
    print(f"  Final labels: {result['final_labels']}")
    print(f"  Confidence: {result['final_confidence']:.2f}")
    print(f"  Dominant source: {result['feature_analysis']['dominant_source']}")
    print()


if __name__ == "__main__":
    main()
