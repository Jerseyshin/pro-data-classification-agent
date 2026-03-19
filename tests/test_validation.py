"""Test validation utilities"""
import pytest
from classification_agent.utils.validation import validate_input, validate_categories


def test_validate_input_ok():
    """Test validate input with valid data"""
    from classification_agent.types.schemas import TableFieldInput
    input_data: TableFieldInput = {
        "table_name": "test_table",
        "field_name": "test_field",
        "field_description": "description"
    }
    # Should not raise
    validate_input(input_data)


def test_validate_input_empty_table_name():
    """Test validate input with empty table name"""
    from classification_agent.types.schemas import TableFieldInput
    input_data: TableFieldInput = {
        "table_name": "",
        "field_name": "test_field",
        "field_description": "description"
    }
    with pytest.raises(ValueError, match="table_name cannot be empty"):
        validate_input(input_data)


def test_validate_categories_ok(sample_categories):
    """Test validate categories with valid data"""
    # Should not raise
    validate_categories(sample_categories)


def test_validate_categories_empty():
    """Test validate categories with empty list"""
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_categories([])


def test_validate_categories_duplicate():
    """Test validate categories with duplicate data_item"""
    from classification_agent.types.schemas import HierarchicalCategory
    categories: list[HierarchicalCategory] = [
        {
            "level1": "A",
            "level2": "B",
            "data_item": "test",
            "data_subitems": []
        },
        {
            "level1": "A",
            "level2": "C",
            "data_item": "test",
            "data_subitems": []
        }
    ]
    with pytest.raises(ValueError, match="Duplicate data_item"):
        validate_categories(categories)
