from typing import Any, Dict, List

from classification_agent.graph.state import ClassificationState
from classification_agent.types.schemas import PredictedItem
from classification_agent.llm.base import BaseLLM
from .base_node import BaseNode


class FinalResultNode(BaseNode):
    """最终结果节点 - 汇总验证后的结果输出"""

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        verification = state["verification"]
        preliminary = state["preliminary_classification"]

        # Collect all predictions: keep verified ones + add new ones
        final_predictions: List[PredictedItem] = []

        # Process verified predictions from preliminary
        for verified in verification["verified_predictions"]:
            # Check if is_kept exists (LLM output format), default to keeping if missing
            if verified.get("is_kept", True):
                if "original_prediction" in verified:
                    pred = verified["original_prediction"].copy()
                    if "adjusted_confidence" in verified:
                        pred["confidence"] = verified["adjusted_confidence"]
                    final_predictions.append(pred)
                elif "level1" in verified and "level2" in verified and "data_item" in verified:
                    # Direct prediction format from LLM, add as new prediction
                    final_predictions.append(verified)

        # Add any new predictions discovered during verification
        for added in verification.get("added_missing", []):
            # 防御性校验：确保 added_missing 中的条目包含必需字段
            required_fields = ["level1", "level2", "data_item", "confidence", "matching_data_subitems", "reasoning"]
            if all(field in added for field in required_fields):
                final_predictions.append(added)

        # If no predictions kept after verification, try to keep the best from original
        if not final_predictions and preliminary.get("predictions"):
            # Sort by confidence and take top
            sorted_preds = sorted(
                preliminary["predictions"],
                key=lambda p: p.get("confidence", 0),
                reverse=True
            )
            if sorted_preds:
                final_predictions = [sorted_preds[0]]

        # 如果不允许多标签，只保留置信度最高的
        if not state["allow_multiple"] and len(final_predictions) > 1:
            final_predictions.sort(key=lambda p: p["confidence"], reverse=True)
            final_predictions = [final_predictions[0]]

        # Extract labels for convenient access
        final_labels = [p["data_item"] for p in final_predictions]

        # Calculate average confidence
        if final_predictions:
            avg_confidence = sum(p["confidence"] for p in final_predictions) / len(final_predictions)
        else:
            avg_confidence = 0.0

        # Build reasoning chain（过滤空字符串条目）
        reasoning_chain = [
            f"Feature analysis: {state['feature_analysis']['semantic_summary']}",
            f"Dominant source: {state['feature_analysis']['dominant_source']}",
            f"Preliminary predictions: {len(preliminary.get('predictions', []))} found",
            f"Verification: {len(verification.get('removed_false_positives', []))} removed, {len(verification.get('added_missing', []))} added",
        ]
        cross_note = verification.get("cross_validation_note", "")
        if cross_note:
            reasoning_chain.append(cross_note)

        self.logger.info(
            "最终结果: %d 个标签 %s，平均置信度: %.2f",
            len(final_labels),
            final_labels,
            avg_confidence,
        )

        return {
            # 最终结果存在这里，ClassificationAgent会提取出去构建ClassificationResult
            "_final_predictions": final_predictions,
            "_final_labels": final_labels,
            "_final_avg_confidence": avg_confidence,
            "_reasoning_chain": reasoning_chain
        }
