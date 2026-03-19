#!/usr/bin/env python3
import os
import json
from dotenv import load_dotenv
from classification_agent import ClassificationAgent
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
from classification_agent.llm.openai_wrapper import OpenAILLM

load_dotenv()


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "MiniMax-M2.5")
    base_url = os.getenv("OPENAI_BASE_URL")

    llm = OpenAILLM(api_key=api_key, model=model, base_url=base_url)

    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=DEFAULT_DATA_CATEGORIES,
        confidence_threshold=0.7,
        allow_multiple=True
    )

    input_data = {
        "table_name": "框架应用市场下载结果日志",
        "field_name": "deviceidtail",
        "field_description": "设备ID尾部"
    }

    # Debug: manually build and run step by step
    from langgraph.graph import StateGraph
    from classification_agent.graph.builder import build_graph

    graph = build_graph(llm).compile()

    initial_state = {
        "input": input_data,
        "hierarchical_categories": DEFAULT_DATA_CATEGORIES,
        "confidence_threshold": 0.7,
        "allow_multiple": True,
        "feature_analysis": None,
        "preliminary_classification": None,
        "verification": None,
        "reclassification_count": 0
    }

    print("Running full graph...")
    result = graph.invoke(initial_state)

    print("\n=== FINAL RESULT STATE ===")
    print(f"_final_predictions exists: {'_final_predictions' in result}")
    if '_final_predictions' in result:
        print(f"Number of predictions: {len(result['_final_predictions'])}")
        for pred in result['_final_predictions']:
            print(f"  - {pred['data_item']} (confidence: {pred['confidence']})")

    print(f"\n_final_labels: {result.get('_final_labels', [])}")
    print(f"\nVerification:")
    if 'verification' in result:
        verif = result['verification']
        print(f"  verified_predictions: {len(verif.get('verified_predictions', []))}")
        print(f"  added_missing: {len(verif.get('added_missing', []))}")
        for added in verif.get('added_missing', []):
            print(f"    + {added['data_item']} (confidence: {added['confidence']})")


if __name__ == "__main__":
    main()
