import json
from typing import Any, Dict, List, Set

from classification_agent.graph.state import ClassificationState
from classification_agent.types.schemas import (
    PredictedItem,
    HierarchicalCategory,
    VerificationResult,
    BulkVerificationResult,
)
from classification_agent.llm.base import BaseLLM
from classification_agent.prompts.loader import load_prompt
from classification_agent.utils.logging import get_logger
from .base_node import BaseNode

logger = get_logger(__name__)


class _CacheManager:
    """Cache manager for hierarchical categories with version control"""

    def __init__(self):
        self.cache_version = None
        self.category_key_map = set()
        self.data_item_set = set()  # Stores all data_item names
        self.subitem_map = {}  # subitem_key -> parent data_item

    def refresh_cache(self, hierarchical_categories):
        """Precompute lookups and update cache version"""
        # Compute current hash
        current_hash = self._compute_hash(hierarchical_categories)

        # Only update if categories have changed
        if current_hash != self.cache_version:
            self._precompute_lookups(hierarchical_categories)
            self.cache_version = current_hash
            logger.debug("Cache refreshed with version: %s", self.cache_version)

    def _compute_hash(self, hierarchical_categories):
        """Compute hash of the classification structure"""
        import hashlib

        sorted_str = json.dumps(hierarchical_categories, sort_keys=True)
        return hashlib.sha256(sorted_str.encode()).hexdigest()

    def _precompute_lookups(self, hierarchical_categories):
        """Precompute all lookup tables"""
        # Reset caches
        self.category_key_map = set()
        self.data_item_set = set()
        self.subitem_map = {}

        # Precompute all lookups
        for cat in hierarchical_categories:
            # Category key (level1||level2||data_item)
            key = f"{cat['level1'].strip().lower()}||{cat['level2'].strip().lower()}||{cat['data_item'].strip().lower()}"
            self.category_key_map.add(key)

            # Data item existence
            data_item_key = cat["data_item"].strip().lower()
            self.data_item_set.add(data_item_key)

            # Subitem relationships
            for subitem in cat["data_subitems"]:
                subitem_key = f"{data_item_key}||{subitem['name'].strip().lower()}"
                # Store parent data_item for each subitem
                self.subitem_map[subitem_key] = data_item_key


def deterministic_hallucination_check_bulk(
    field_predictions: List[List[PredictedItem]],
    hierarchical_categories: List[HierarchicalCategory],
) -> List[List[PredictedItem]]:
    """Deterministic hallucination check with O(n) complexity using cached lookups"""
    # Initialize cache manager
    if not hasattr(deterministic_hallucination_check_bulk, "cache_manager"):
        deterministic_hallucination_check_bulk.cache_manager = _CacheManager()

    # Refresh cache if needed
    cache_manager = deterministic_hallucination_check_bulk.cache_manager
    cache_manager.refresh_cache(hierarchical_categories)

    cleaned_results: List[List[PredictedItem]] = []
    total_removed = 0
    total_original = 0

    for predictions in field_predictions:
        total_original += len(predictions)
        passed: List[PredictedItem] = []

        for pred in predictions:
            level1 = pred["level1"].strip().lower()
            level2 = pred["level2"].strip().lower()
            data_item = pred["data_item"].strip().lower()
            key = f"{level1}||{level2}||{data_item}"

            # 1. Check category existence (O(1))
            if key not in cache_manager.category_key_map:
                logger.debug(
                    "Removing prediction: Hierarchy not found - %s||%s||%s",
                    level1,
                    level2,
                    data_item,
                )
                total_removed += 1
                continue

            # 2. Check all subitems exist and have valid parents (O(k))
            valid_subitems = True
            for subitem_name in pred["matching_data_subitems"]:
                subitem_key = f"{data_item}||{subitem_name.strip().lower()}"

                # Check subitem exists and belongs to the same data_item
                if subitem_key not in cache_manager.subitem_map:
                    valid_subitems = False
                    break

                # Verify parent-child relationship
                if cache_manager.subitem_map[subitem_key] != data_item:
                    valid_subitems = False
                    break

            if not valid_subitems:
                total_removed += 1
                continue

            passed.append(pred)

        cleaned_results.append(passed)

    if total_removed > 0:
        logger.info(
            "Removed %d/%d predictions due to hallucination",
            total_removed,
            total_original,
        )

    return cleaned_results

    cleaned_results: List[List[PredictedItem]] = []
    total_removed = 0
    total_original = 0

    for predictions in field_predictions:
        total_original += len(predictions)
        passed: List[PredictedItem] = []
        for pred in predictions:
            level1 = pred["level1"].strip().lower()
            level2 = pred["level2"].strip().lower()
            data_item = pred["data_item"].strip().lower()
            key = f"{level1}||{level2}||{data_item}"

            if key not in category_key_map:
                logger.debug(
                    "deterministic_hallucination_check_bulk: removing prediction because hierarchy not found - "
                    "level1=%s, level2=%s, data_item=%s",
                    pred["level1"],
                    pred["level2"],
                    pred["data_item"],
                )
                total_removed += 1
                continue

            valid_subitems = True
            for subitem_name in pred["matching_data_subitems"]:
                subitem_key = data_item + "||" + subitem_name.strip().lower()
                if subitem_key not in subitem_map:
                    valid_subitems = False
                    break

            if not valid_subitems:
                total_removed += 1
                continue

            passed.append(pred)

        cleaned_results.append(passed)

    if total_removed > 0:
        logger.info(
            "deterministic_hallucination_check_bulk: removed %d/%d predictions due to hallucination",
            total_removed,
            total_original,
        )

    return cleaned_results


class BulkSelfVerificationNode(BaseNode):
    """批量自我验证节点 - 一次性验证所有字段的所有预测，只调用一次LLM"""

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        inputs = state["inputs"]
        bulk_feature_analysis = state["bulk_feature_analysis"]
        bulk_preliminary = state["bulk_preliminary_classification"]

        if not inputs or not bulk_feature_analysis or not bulk_preliminary:
            return {
                "bulk_verification": None,
            }

        # Extract field analyses and preliminary predictions
        field_analyses = bulk_feature_analysis["field_analyses"]
        field_preliminary = bulk_preliminary["field_results"]

        # Do deterministic hallucination check first for all fields
        # Collect cleaned predictions for each field
        all_predictions: List[List[PredictedItem]] = [
            fr.get("predictions", []) for fr in field_preliminary
        ]
        cleaned_predictions = deterministic_hallucination_check_bulk(
            all_predictions,
            state["hierarchical_categories"],
        )

        # Create cleaned preliminary results
        cleaned_preliminary: List[Dict[str, Any]] = []
        for i, fr in enumerate(field_preliminary):
            cleaned = dict(fr)
            cleaned["predictions"] = cleaned_predictions[i]
            cleaned_preliminary.append(cleaned)

        prompt = load_prompt(
            "bulk_self_verification.jinja2",
            inputs=inputs,
            field_analyses=field_analyses,
            field_preliminary=cleaned_preliminary,
            hierarchical_categories=state["hierarchical_categories"],
            confidence_threshold=state["confidence_threshold"],
            table_context=state.get("table_context_analysis"),
        )

        result = self.llm.generate_json(prompt)

        # Validate result structure
        required_keys = ["field_verifications"]
        if "field_verifications" not in result:
            result["field_verifications"] = []

        # Ensure each field verification has all required keys
        required_field_keys = [
            "verified_predictions",
            "removed_false_positives",
            "added_missing",
            "average_confidence",
            "cross_validation_note",
            "suggests_reclassification",
        ]
        processed_verifications: List[VerificationResult] = []
        total_kept = 0
        total_removed = 0
        total_added = 0

        for idx, verification in enumerate(result["field_verifications"]):
            for key in required_field_keys:
                if key not in verification:
                    if key in [
                        "verified_predictions",
                        "removed_false_positives",
                        "added_missing",
                    ]:
                        verification[key] = []
                    elif key == "average_confidence":
                        verification[key] = 0.0
                    elif key == "cross_validation_note":
                        verification[key] = ""
                    else:
                        verification[key] = False

            # Handle boolean conversion
            suggested = verification.get("suggests_reclassification", False)
            if isinstance(suggested, str):
                suggested = suggested.lower() in ["true", "1", "yes"]
            verification["suggests_reclassification"] = suggested

            # Count statistics
            kept_count = sum(
                1
                for v in verification.get("verified_predictions", [])
                if v.get("is_kept")
            )
            rm_count = len(verification.get("removed_false_positives", []))
            add_count = len(verification.get("added_missing", []))
            total_kept += kept_count
            total_removed += rm_count
            total_added += add_count

            if idx < len(inputs):
                field_name = inputs[idx]["field_name"]
                avg_conf = verification.get("average_confidence", 0.0)
                if suggested:
                    logger.info(
                        "字段[%s] 验证建议重分类，平均置信度: %.2f",
                        field_name,
                        avg_conf,
                    )
                else:
                    logger.info(
                        "字段[%s] 验证通过，保留 %d 个预测，移除 %d 个，新增 %d 个，平均置信度: %.2f",
                        field_name,
                        kept_count,
                        rm_count,
                        add_count,
                        avg_conf,
                    )

            processed_verifications.append(VerificationResult(**verification))

        bulk_result: BulkVerificationResult = {
            "field_verifications": processed_verifications,
        }

        self.logger.info(
            "批量验证完成: %d 个字段，保留 %d 预测，移除 %d，新增 %d",
            len(processed_verifications),
            total_kept,
            total_removed,
            total_added,
        )

        return {
            "bulk_verification": bulk_result,
        }
