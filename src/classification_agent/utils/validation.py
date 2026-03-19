from typing import List
from classification_agent.types.schemas import TableFieldInput, HierarchicalCategory


def validate_input(input_data: TableFieldInput) -> None:
    """Validate input data"""
    if not input_data.get("table_name") or not input_data.get("table_name").strip():
        raise ValueError("table_name cannot be empty")
    if not input_data.get("field_name") or not input_data.get("field_name").strip():
        raise ValueError("field_name cannot be empty")


def validate_categories(categories: List[HierarchicalCategory]) -> None:
    """Validate hierarchical categories"""
    if not categories:
        raise ValueError("hierarchical_categories cannot be empty")

    if len(categories) < 2:
        raise ValueError("At least 2 categories are required for classification")

    # 三元组 (level1, level2, data_item) 必须唯一，允许不同level下data_item重名
    seen_keys = set()
    for i, cat in enumerate(categories):
        if not cat.get("level1") or not cat.get("level1").strip():
            raise ValueError(f"Category {i}: level1 cannot be empty")
        if not cat.get("level2") or not cat.get("level2").strip():
            raise ValueError(f"Category {i}: level2 cannot be empty")
        if not cat.get("data_item") or not cat.get("data_item").strip():
            raise ValueError(f"Category {i}: data_item cannot be empty")

        key = (cat["level1"].strip(), cat["level2"].strip(), cat["data_item"].strip())
        if key in seen_keys:
            raise ValueError(f"Duplicate category: level1={key[0]}, level2={key[1]}, data_item={key[2]}")
        seen_keys.add(key)

        if "data_subitems" not in cat or not isinstance(cat["data_subitems"], list):
            raise ValueError(f"Category {i}: data_subitems must be a list")
