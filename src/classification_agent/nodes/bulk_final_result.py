from typing import Any, Dict, List

from classification_agent.graph.state import ClassificationState
from classification_agent.types.schemas import (
    ClassificationResult,
    FeatureAnalysisResult,
    PreliminaryResult,
    VerificationResult,
    PredictedItem,
)
from classification_agent.llm.base import BaseLLM
from .base_node import BaseNode


class BulkFinalResultNode(BaseNode):
    """批量最终结果汇总节点 - 汇总所有字段的分析、分类、验证结果，生成完整的结果列表"""

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        inputs = state["inputs"]
        bulk_feature = state["bulk_feature_analysis"]
        bulk_preliminary = state["bulk_preliminary_classification"]
        bulk_verification = state["bulk_verification"]

        if not inputs or not bulk_feature or not bulk_preliminary or not bulk_verification:
            return {
                "bulk_final_results": [],
            }

        feature_list = bulk_feature["field_analyses"]
        preliminary_list = bulk_preliminary["field_results"]
        verification_list = bulk_verification["field_verifications"]
        table_context = state.get("table_context_analysis")

        # Known keys for filtering (keep the same as single mode)
        known_feature_keys = {"table_name_keywords", "field_name_keywords", "description_keywords",
                              "semantic_summary", "consistency_analysis", "dominant_source"}
        known_preliminary_keys = {"predictions", "total_confidence"}
        known_verification_keys = {"verified_predictions", "removed_false_positives", "added_missing",
                                   "average_confidence", "cross_validation_note", "suggests_reclassification"}

        final_results: List[ClassificationResult] = []

        for idx, _input in enumerate(inputs):
            feature_analysis = feature_list[idx] if idx < len(feature_list) else None
            preliminary = preliminary_list[idx] if idx < len(preliminary_list) else None
            verification = verification_list[idx] if idx < len(verification_list) else None

            # Extract final predictions from verification
            final_predictions: List[PredictedItem] = []
            final_labels: List[str] = []
            final_avg_confidence = 0.0

            if verification:
                # Rebuild final predictions from verified items
                for verified in verification.get("verified_predictions", []):
                    if verified.get("is_kept") and verified.get("original_prediction"):
                        final_predictions.append(verified["original_prediction"])

                final_labels = [p["data_item"] for p in final_predictions]
                final_avg_confidence = verification.get("average_confidence", 0.0)

            # Build classification result
            result: ClassificationResult = {
                "final_predictions": final_predictions,
                "final_labels": final_labels,
                "final_confidence": final_avg_confidence,
                "reasoning_chain": [],  # Bulk mode doesn't track per-field reasoning chain in state
                "feature_analysis": FeatureAnalysisResult(
                    **{k: v for k, v in (feature_analysis or {}).items() if k in known_feature_keys}
                ) if feature_analysis else None,
                "preliminary_result": PreliminaryResult(
                    **{k: v for k, v in (preliminary or {}).items() if k in known_preliminary_keys}
                ) if preliminary else None,
                "verification_result": VerificationResult(
                    **{k: v for k, v in (verification or {}).items() if k in known_verification_keys}
                ) if verification else None,
                "evaluation": None,  # Evaluation done later by separate node
                "table_context_analysis": table_context,
            }

            final_results.append(result)

        self.logger.info(
            "批量最终结果汇总完成: %d 个字段处理完毕",
            len(final_results)
        )

        return {
            "bulk_final_results": final_results,
        }
