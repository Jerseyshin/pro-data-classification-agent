"""Integration test for the full classification flow"""
import json
import pytest
from classification_agent import ClassificationAgent
from classification_agent.llm.openai_wrapper import OpenAILLM


@pytest.mark.integration
def test_full_classification_deviceidtail(openai_api_key, sample_categories):
    """Test the specific case: deviceidtail in app download log"""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")

    llm = OpenAILLM(api_key=openai_api_key, model="gpt-4o-mini")
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=sample_categories,
        confidence_threshold=0.7,
        allow_multiple=True
    )

    input_data = {
        "table_name": "框架应用市场下载结果日志",
        "field_name": "deviceidtail",
        "field_description": "设备ID尾部"
    }

    result = agent.classify(input_data)

    assert "final_labels" in result
    assert len(result["final_labels"]) > 0
    assert "final_confidence" in result
    assert 0 <= result["final_confidence"] <= 1
    assert "feature_analysis" in result
    assert "preliminary_result" in result
    assert "verification_result" in result

    # Check that dominant_source is set
    assert "dominant_source" in result["feature_analysis"]

    print("\nResult:")
    print(f"Final labels: {result['final_labels']}")
    print(f"Confidence: {result['final_confidence']}")
    print(f"Dominant source: {result['feature_analysis']['dominant_source']}")
    print(f"Cross-validation note: {result['verification_result']['cross_validation_note']}")


@pytest.mark.integration
def test_full_classification_network(openai_api_key, sample_categories):
    """Test the specific case: network field in any table"""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")

    llm = OpenAILLM(api_key=openai_api_key, model="gpt-4o-mini")
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=sample_categories,
        confidence_threshold=0.7,
        allow_multiple=True
    )

    input_data = {
        "table_name": "框架应用市场下载结果日志",
        "field_name": "network",
        "field_description": None
    }

    result = agent.classify(input_data)

    assert "final_labels" in result
    assert len(result["final_labels"]) > 0
    # Should be classified as network basic information
    assert "网络基本信息" in result["final_labels"]

    print("\nResult:")
    print(f"Final labels: {result['final_labels']}")
    print(f"Confidence: {result['final_confidence']}")
    print(f"Dominant source: {result['feature_analysis']['dominant_source']}")


@pytest.mark.integration
@pytest.mark.parametrize("test_case_file", ["tests/fixtures/test_cases.json"])
def test_all_test_cases(openai_api_key, sample_categories, test_case_file):
    """Run all test cases from the fixture file"""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")

    with open(test_case_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    llm = OpenAILLM(api_key=openai_api_key, model="gpt-4o-mini")
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=sample_categories,
        confidence_threshold=0.7,
        allow_multiple=True
    )

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {case['name']}")
        print(f"Description: {case['description']}")

        result = agent.classify(case["input"], sample_categories)

        print(f"Input: {case['input']}")
        print(f"Expected: {case['expected_labels']}")
        print(f"Got: {result['final_labels']}")
        print(f"Confidence: {result['final_confidence']:.2f}")

        # Check that we got at least one expected label
        found_expected = any(label in result["final_labels"] for label in case["expected_labels"])
        assert found_expected, f"Expected at least one of {case['expected_labels']}, got {result['final_labels']}"
