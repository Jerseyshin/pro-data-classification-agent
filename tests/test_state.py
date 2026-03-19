"""Test state schema definitions"""
import pytest
from classification_agent.types.schemas import TableFieldInput, HierarchicalCategory
from classification_agent.graph.state import ClassificationState


def test_table_field_input():
    """Test TableFieldInput structure"""
    input_data: TableFieldInput = {
        "table_name": "test_table",
        "field_name": "test_field",
        "field_description": "test description"
    }
    assert input_data["table_name"] == "test_table"
    assert input_data["field_name"] == "test_field"
    assert input_data["field_description"] == "test description"


def test_hierarchical_category():
    """Test HierarchicalCategory structure"""
    category: HierarchicalCategory = {
        "level1": "level1",
        "level2": "level2",
        "data_item": "data_item",
        "data_subitems": [
            {"name": "sub1", "description": "desc1"}
        ]
    }
    assert category["level1"] == "level1"
    assert len(category["data_subitems"]) == 1


def test_classification_state():
    """Test ClassificationState initialization"""
    from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES

    input_data: TableFieldInput = {
        "table_name": "test_table",
        "field_name": "test_field",
        "field_description": "test description"
    }

    state: ClassificationState = {
        "input": input_data,
        "hierarchical_categories": DEFAULT_DATA_CATEGORIES,
        "confidence_threshold": 0.7,
        "allow_multiple": True,
        "feature_analysis": None,
        "preliminary_classification": None,
        "verification": None,
        "reclassification_count": 0
    }

    assert state["input"] == input_data
    assert len(state["hierarchical_categories"]) > 0
