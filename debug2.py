#!/usr/bin/env python3
import os
import json
from dotenv import load_dotenv
from classification_agent import ClassificationAgent
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
from classification_agent.llm.openai_wrapper import OpenAILLM
from classification_agent.nodes.feature_analysis import FeatureAnalysisNode
from classification_agent.nodes.preliminary_classification import PreliminaryClassificationNode
from classification_agent.nodes.self_verification import SelfVerificationNode
from classification_agent.nodes.final_result import FinalResultNode

load_dotenv()


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "MiniMax-M2.5")
    base_url = os.getenv("OPENAI_BASE_URL")

    llm = OpenAILLM(api_key=api_key, model=model, base_url=base_url)

    input_data = {
        "table_name": "框架应用市场下载结果日志",
        "field_name": "deviceidtail",
        "field_description": "设备ID尾部"
    }

    state = {
        "input": input_data,
        "hierarchical_categories": DEFAULT_DATA_CATEGORIES,
        "confidence_threshold": 0.7,
        "allow_multiple": True,
        "feature_analysis": None,
        "preliminary_classification": None,
        "verification": None,
        "reclassification_count": 0
    }

    # Step 1: feature analysis
    print("=== Step 1: Feature Analysis ===")
    node = FeatureAnalysisNode(llm)
    update = node.process(state)
    state.update(update)
    print(f"Dominant source: {state['feature_analysis']['dominant_source']}")

    # Step 2: preliminary classification
    print("\n=== Step 2: Preliminary Classification ===")
    node = PreliminaryClassificationNode(llm)
    update = node.process(state)
    state.update(update)
    print(f"Predictions: {[p['data_item'] for p in state['preliminary_classification']['predictions']]}")
    print(f"Total confidence: {state['preliminary_classification']['total_confidence']}")

    # Step 3: self verification
    print("\n=== Step 3: Self Verification ===")
    node = SelfVerificationNode(llm)
    update = node.process(state)
    state.update(update)
    verif = state["verification"]
    print(f"verified_predictions: {len(verif['verified_predictions'])}")
    print(f"removed_false_positives: {verif['removed_false_positives']}")
    print(f"added_missing: {[p['data_item'] for p in verif['added_missing']] if verif['added_missing'] else '[]'}")
    print(f"average_confidence: {verif['average_confidence']}")
    print(f"suggests_reclassification: {verif['suggests_reclassification']}")
    print(f"reclassification_count: {state['reclassification_count']}")
    print(f"cross_validation_note: {verif['cross_validation_note']}")

    if verif["suggests_reclassification"]:
        print("\n=== Step 4: Reclassification (preliminary again) ===")
        node = PreliminaryClassificationNode(llm)
        update = node.process(state)
        state.update(update)
        print(f"New predictions: {[p['data_item'] for p in state['preliminary_classification']['predictions']]}")

        print("\n=== Step 5: Verification again ===")
        node = SelfVerificationNode(llm)
        update = node.process(state)
        state.update(update)
        verif = state["verification"]
        print(f"verified_predictions: {len(verif['verified_predictions'])}")
        print(f"added_missing: {[p['data_item'] for p in verif['added_missing']] if verif['added_missing'] else '[]'}")
        print(f"average_confidence: {verif['average_confidence']}")
        print(f"suggests_reclassification: {verif['suggests_reclassification']}")
        print(f"reclassification_count: {state['reclassification_count']}")

    # Step final result
    print("\n=== Step 6: Final Result ===")
    node = FinalResultNode(llm)
    update = node.process(state)
    print(f"_final_predictions in update: {'_final_predictions' in update}")
    if '_final_predictions' in update:
        print(f"Count: {len(update['_final_predictions'])}")
        for p in update['_final_predictions']:
            print(f"  - {p['data_item']} ({p['confidence']:.2f})")


if __name__ == "__main__":
    main()
