from typing import Any, Dict

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.prompts.loader import load_prompt
from .base_node import BaseNode


class PreliminaryClassificationNode(BaseNode):
    """初步分类节点 - 基于特征分析给出初步预测"""

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        # 重分类时将上次验证失败的原因注入 prompt，避免 LLM 重复相同错误
        previous_verification = state.get("verification")
        reclassification_note = None
        if previous_verification and state.get("reclassification_count", 0) > 0:
            reclassification_note = previous_verification.get("cross_validation_note", "")

        # 获取之前被发现是幻觉的数据项，重分类时不要重复预测
        hallucinated_data_items = state.get("hallucinated_data_items", [])

        prompt = load_prompt(
            "preliminary_classification.jinja2",
            input=state["input"],
            feature_analysis=state["feature_analysis"],
            hierarchical_categories=state["hierarchical_categories"],
            allow_multiple=state["allow_multiple"],
            reclassification_note=reclassification_note,
            hallucinated_data_items=hallucinated_data_items,
            table_context=state.get("table_context_analysis"),
        )

        result = self.llm.generate_json(prompt)

        required_keys = ["predictions", "total_confidence"]
        for key in required_keys:
            if key not in result:
                if key == "predictions":
                    result[key] = []
                else:
                    result[key] = 0.0

        self.logger.info(
            "初步预测 %d 个分类，总置信度: %.2f",
            len(result.get("predictions", [])),
            result.get("total_confidence", 0.0),
        )

        return {
            "preliminary_classification": result,
            "reclassification_count": state.get("reclassification_count", 0)
        }
