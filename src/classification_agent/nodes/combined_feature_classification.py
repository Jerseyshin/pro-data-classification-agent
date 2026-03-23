"""
Combined Feature Analysis + Preliminary Classification node for fast mode.

This node merges two sequential LLM calls into one, reducing network round-trips
by ~33% and significantly speeds up batch processing. The tradeoff is slightly
longer prompt but still within context limits for most models.
"""
from typing import Any, Dict

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.prompts.loader import load_prompt
from classification_agent.utils.logging import get_logger
from .base_node import BaseNode

logger = get_logger(__name__)


class CombinedFeatureAndClassificationNode(BaseNode):
    """Combined node: Feature Analysis + Preliminary Classification in one LLM call.

    Fast mode: reduces one network round-trip per sample, cutting total
    processing time by ~33% with only minimal accuracy impact.
    """

    def __init__(self, llm: BaseLLM):
        super().__init__(llm)

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        """Process both feature analysis and preliminary classification in one step"""
        prompt = load_prompt(
            "feature_classification_combined.jinja2",
            input=state["input"],
            hierarchical_categories=state["hierarchical_categories"],
            retrieved_examples=state.get("retrieved_examples"),
            allow_multiple=state["allow_multiple"],
            table_context=state.get("table_context_analysis"),
            hallucinated_data_items=state.get("hallucinated_data_items"),
        )

        result = self.llm.generate_json(prompt)

        # Extract the two parts from the combined output
        feature_analysis = result.get("feature_analysis", {})
        preliminary_classification = result.get("preliminary_classification", {})

        # Log
        avg_conf = preliminary_classification.get("total_confidence", 0.0)
        num_pred = len(preliminary_classification.get("predictions", []))
        logger.info(
            "CombinedFeatureClassificationNode - combined analysis + classification done, "
            "%d predictions, total confidence: %.2f",
            num_pred, avg_conf
        )

        return {
            "feature_analysis": feature_analysis,
            "preliminary_classification": preliminary_classification,
        }
