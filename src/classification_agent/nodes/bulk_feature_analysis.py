from typing import Any, Dict, List

from classification_agent.graph.state import ClassificationState
from classification_agent.types.schemas import FeatureAnalysisResult, BulkFeatureAnalysisResult
from classification_agent.llm.base import BaseLLM
from classification_agent.prompts.loader import load_prompt
from .base_node import BaseNode


class BulkFeatureAnalysisNode(BaseNode):
    """批量特征分析节点 - 一次性分析整张表所有字段，只调用一次LLM"""

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        inputs = state["inputs"]
        if not inputs:
            return {
                "bulk_feature_analysis": None,
            }

        prompt = load_prompt(
            "bulk_feature_analysis.jinja2",
            inputs=inputs,
            hierarchical_categories=state["hierarchical_categories"],
            table_context=state.get("table_context_analysis"),
        )

        result = self.llm.generate_json(prompt)

        # 验证结果结构
        required_keys = ["field_analyses"]
        if "field_analyses" not in result:
            result["field_analyses"] = []

        # Ensure each field analysis has all required keys
        required_field_keys = [
            "table_name_keywords",
            "field_name_keywords",
            "description_keywords",
            "semantic_summary",
            "consistency_analysis",
            "dominant_source"
        ]
        processed_analyses: List[FeatureAnalysisResult] = []
        for idx, analysis in enumerate(result["field_analyses"]):
            for key in required_field_keys:
                if key not in analysis:
                    analysis[key] = [] if key.endswith("keywords") else ""
            # Get the input field for this analysis to log
            if idx < len(inputs):
                field_name = inputs[idx]["field_name"]
                self.logger.info(
                    "字段[%s] 语义摘要: %s | 主导来源: %s",
                    field_name,
                    analysis.get("semantic_summary", ""),
                    analysis.get("dominant_source", ""),
                )
            processed_analyses.append(FeatureAnalysisResult(**analysis))

        bulk_result: BulkFeatureAnalysisResult = {
            "field_analyses": processed_analyses,
        }

        return {
            "bulk_feature_analysis": bulk_result,
        }
