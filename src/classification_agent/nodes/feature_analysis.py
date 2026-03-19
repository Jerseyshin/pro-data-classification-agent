from typing import Any, Dict

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.prompts.loader import load_prompt
from .base_node import BaseNode


class FeatureAnalysisNode(BaseNode):
    """特征分析节点 - 分别分析表名、字段名、描述，提取关键词并分析一致性"""

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        prompt = load_prompt(
            "feature_analysis.jinja2",
            input=state["input"],
            hierarchical_categories=state["hierarchical_categories"]
        )

        result = self.llm.generate_json(prompt)

        # 验证结果结构
        required_keys = [
            "table_name_keywords",
            "field_name_keywords",
            "description_keywords",
            "semantic_summary",
            "consistency_analysis",
            "dominant_source"
        ]
        for key in required_keys:
            if key not in result:
                result[key] = [] if key.endswith("keywords") else ""

        self.logger.info(
            "语义摘要: %s | 主导来源: %s",
            result.get("semantic_summary", ""),
            result.get("dominant_source", ""),
        )

        return {
            "feature_analysis": result,
            "reclassification_count": state.get("reclassification_count", 0)
        }
