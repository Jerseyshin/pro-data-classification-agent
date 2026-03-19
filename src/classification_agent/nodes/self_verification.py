from typing import Any, Dict

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.prompts.loader import load_prompt
from .base_node import BaseNode


class SelfVerificationNode(BaseNode):
    """自我验证节点 - 交叉验证每个预测，检查仅字段名vs结合表名是否一致，剔除误匹配"""

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        prompt = load_prompt(
            "self_verification.jinja2",
            input=state["input"],
            feature_analysis=state["feature_analysis"],
            preliminary_classification=state["preliminary_classification"],
            hierarchical_categories=state["hierarchical_categories"],
            confidence_threshold=state["confidence_threshold"]
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

        return {
            "verification": result,
            "reclassification_count": current_count
        }
