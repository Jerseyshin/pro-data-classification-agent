from typing import Any, Dict, List

from classification_agent.graph.state import ClassificationState
from classification_agent.types.schemas import PreliminaryResult, BulkPreliminaryResult
from classification_agent.llm.base import BaseLLM
from classification_agent.prompts.loader import load_prompt
from .base_node import BaseNode


class BulkPreliminaryClassificationNode(BaseNode):
    """批量初步分类节点 - 基于所有字段的特征分析，一次性给出所有分类预测，只调用一次LLM"""

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        inputs = state["inputs"]
        bulk_feature_analysis = state["bulk_feature_analysis"]

        if not inputs or not bulk_feature_analysis:
            return {
                "bulk_preliminary_classification": None,
            }

        # Check if this is a reclassification (after verification suggested reclassification)
        # We need to increment reclassification_count if we have hallucinated_data_items
        # or if this is not the first pass
        hallucinated_items = state.get("hallucinated_data_items", [])
        reclassification_count = state.get("reclassification_count", 0)

        # If we have hallucinated items and this is the first reclassification, increment count
        if hallucinated_items and reclassification_count == 0:
            # This is a reclassification pass
            reclassification_count = 1
            self.logger.info(
                "开始重新分类循环，处理 %d 个幻觉数据项", len(hallucinated_items)
            )

        # Get the feature analyses for all fields
        field_analyses = bulk_feature_analysis["field_analyses"]

        prompt = load_prompt(
            "bulk_preliminary_classification.jinja2",
            inputs=inputs,
            field_analyses=field_analyses,
            hierarchical_categories=state["hierarchical_categories"],
            allow_multiple=state["allow_multiple"],
            table_context=state.get("table_context_analysis"),
            retrieved_examples=state.get("retrieved_examples"),
            hallucinated_data_items=state.get(
                "hallucinated_data_items", []
            ),  # Filter known hallucinations
        )

        result = self.llm.generate_json(prompt)

        # Validate result structure
        required_keys = ["field_results"]
        if "field_results" not in result:
            result["field_results"] = []

        # Ensure each field result has all required keys
        required_field_keys = ["predictions", "total_confidence"]
        processed_results: List[PreliminaryResult] = []
        total_predictions = 0
        for idx, field_result in enumerate(result["field_results"]):
            for key in required_field_keys:
                if key not in field_result:
                    field_result[key] = [] if key == "predictions" else 0.0
            if idx < len(inputs):
                field_name = inputs[idx]["field_name"]
                preds = field_result.get("predictions", [])
                total_predictions += len(preds)
                avg_conf = field_result.get("total_confidence", 0.0)
                self.logger.info(
                    "字段[%s] 初步预测 %d 个分类，总置信度: %.2f",
                    field_name,
                    len(preds),
                    avg_conf,
                )
            processed_results.append(PreliminaryResult(**field_result))

        bulk_result: BulkPreliminaryResult = {
            "field_results": processed_results,
        }

        self.logger.info(
            "批量初步分类完成: %d 个字段，总计 %d 个预测",
            len(processed_results),
            total_predictions,
        )

        return {
            "bulk_preliminary_classification": bulk_result,
            "reclassification_count": reclassification_count,
        }
