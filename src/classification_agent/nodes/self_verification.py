from typing import Any, Dict, Set, List
from classification_agent.types.schemas import PredictedItem, HierarchicalCategory

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.prompts.loader import load_prompt
from classification_agent.utils.logging import get_logger
from .base_node import BaseNode

logger = get_logger(__name__)


def deterministic_hallucination_check(
    predictions: List[PredictedItem],
    hierarchical_categories: List[HierarchicalCategory],
) -> List[PredictedItem]:
    """Deterministic hallucination check - remove predictions that violate the classification hierarchy:

    Checks:
    1. Does the predicted data_item actually exist in the classification system?
    2. Does each predicted matching_data_subitem actually belong to that data_item?
    3. Is the level1-level2-data_item hierarchy correct (matches what's defined)?

    Returns:
        List of predictions that passed the check (hallucinations are removed before LLM check)
    """
    # Build lookup tables for fast checking
    # 1. lookup by full (level1, level2, data_item) key
    category_key_map: Set[str] = set()
    # 2. lookup data_item existence (allow same data_item name under different levels)
    data_item_existence: Dict[str, Set[str]] = {}  # data_item -> set of (level1, level2) keys
    # 3. lookup allowed subitems for each (level1, level2, data_item)
    subitem_map: Dict[str, Set[str]] = {}

    for cat in hierarchical_categories:
        key = f"{cat['level1'].strip().lower()}||{cat['level2'].strip().lower()}||{cat['data_item'].strip().lower()}"
        category_key_map.add(key)
        data_item_key = cat['data_item'].strip().lower()
        if data_item_key not in data_item_existence:
            data_item_existence[data_item_key] = set()
        data_item_existence[data_item_key].add(key)
        # Add allowed subitems
        for subitem in cat['data_subitems']:
            subitem_key = cat['data_item'].strip().lower() + "||" + subitem['name'].strip().lower()
            subitem_map[subitem_key] = set()

    passed: List[PredictedItem] = []
    removed_count = 0

    for pred in predictions:
        # Check 1: Full hierarchy matches
        level1 = pred['level1'].strip().lower()
        level2 = pred['level2'].strip().lower()
        data_item = pred['data_item'].strip().lower()
        key = f"{level1}||{level2}||{data_item}"

        if key not in category_key_map:
            # Hallucination: hierarchy doesn't match defined classification
            logger.debug(
                "deterministic_hallucination_check: removing prediction because hierarchy not found - "
                "level1=%s, level2=%s, data_item=%s",
                pred['level1'], pred['level2'], pred['data_item']
            )
            removed_count += 1
            continue

        # Check 2: All predicted subitems exist for this data_item
        valid_subitems = True
        for subitem_name in pred['matching_data_subitems']:
            subitem_key = data_item + "||" + subitem_name.strip().lower()
            if subitem_key not in subitem_map:
                # Subitem doesn't belong to this data_item
                logger.debug(
                    "deterministic_hallucination_check: removing prediction because subitem doesn't belong to data_item - "
                    "data_item=%s, subitem=%s",
                    pred['data_item'], subitem_name
                )
                valid_subitems = False
                break

        if not valid_subitems:
            removed_count += 1
            continue

        # All checks passed
        passed.append(pred)

    if removed_count > 0:
        logger.info(
            "deterministic_hallucination_check: removed %d/%d predictions due to hallucination",
            removed_count, len(predictions)
        )

    return passed


class SelfVerificationNode(BaseNode):
    """自我验证节点 - 交叉验证每个预测，检查仅字段名vs结合表名是否一致，剔除误匹配。
    First does deterministic hallucination check (remove impossible predictions) before LLM check.
    """

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        # FIRST: Do deterministic hallucination check
        # Remove predictions that are definitely impossible
        cleaned_predictions = deterministic_hallucination_check(
            state["preliminary_classification"]["predictions"],
            state["hierarchical_categories"],
        )

        # Replace with cleaned predictions for LLM to process
        preliminary_classification = dict(state["preliminary_classification"])
        preliminary_classification["predictions"] = cleaned_predictions

        # Collect names of hallucinated data_items that were removed
        # These will be warned against in reclassification
        hallucinated_data_items: List[str] = []
        original_predictions = state["preliminary_classification"]["predictions"]
        for pred in original_predictions:
            if pred not in cleaned_predictions:
                hallucinated_data_items.append(pred["data_item"])

        if hallucinated_data_items:
            logger.info(
                "deterministic_hallucination_check: collected %d hallucinated data_items to warn against: %s",
                len(hallucinated_data_items),
                ", ".join(hallucinated_data_items)
            )

        prompt = load_prompt(
            "self_verification.jinja2",
            input=state["input"],
            feature_analysis=state["feature_analysis"],
            preliminary_classification=preliminary_classification,
            hierarchical_categories=state["hierarchical_categories"],
            confidence_threshold=state["confidence_threshold"],
            table_context=state.get("table_context_analysis"),
        )

        result = self.llm.generate_json(prompt)

        required_keys = [
            "verified_predictions",
            "removed_false_positives",
            "added_missing",
            "average_confidence",
            "cross_validation_note",
            "suggests_reclassification"
        ]
        for key in required_keys:
            if key not in result:
                if key in ["verified_predictions", "removed_false_positives", "added_missing"]:
                    result[key] = []
                elif key == "average_confidence":
                    result[key] = 0.0
                elif key == "cross_validation_note":
                    result[key] = ""
                else:
                    result[key] = False

        # 增加重分类计数
        current_count = state.get("reclassification_count", 0)
        # Handle case where LLM returns string "true"/"false" instead of boolean
        suggested = result.get("suggests_reclassification", False)
        if isinstance(suggested, str):
            suggested = suggested.lower() in ["true", "1", "yes"]

        if suggested and current_count < 1:
            # 允许最多一次重分类
            result["suggests_reclassification"] = True
            current_count += 1
            self.logger.info(
                "验证建议重分类，平均置信度: %.2f，原因: %s",
                result.get("average_confidence", 0.0),
                result.get("cross_validation_note", ""),
            )
        else:
            result["suggests_reclassification"] = False
            self.logger.info(
                "验证通过，保留 %d 个预测，移除 %d 个，新增 %d 个，平均置信度: %.2f",
                sum(1 for v in result.get("verified_predictions", []) if v.get("is_kept")),
                len(result.get("removed_false_positives", [])),
                len(result.get("added_missing", [])),
                result.get("average_confidence", 0.0),
            )

        # If we already had previous hallucinations from earlier iterations, append to them
        existing_hallucinations = state.get("hallucinated_data_items", [])
        existing_hallucinations.extend(hallucinated_data_items)
        # Deduplicate
        existing_hallucinations = list(dict.fromkeys(existing_hallucinated_data_items))

        return {
            "verification": result,
            "reclassification_count": current_count,
            "hallucinated_data_items": existing_hallucinations,
        }
