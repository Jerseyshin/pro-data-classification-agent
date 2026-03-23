"""
Context Analysis Node - Analyze whole table context before field-level classification.

This node:
1. Takes the whole table information (table name, table chinese name, all fields)
2. Infers the overall business purpose of the table
3. Extracts key business concepts
4. Identifies what broad data category the table belongs to
5. Provides global context for subsequent field-level classification
"""
from typing import Any, Dict, Optional

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.prompts.loader import load_prompt
from classification_agent.utils.logging import get_logger
from .base_node import BaseNode

logger = get_logger(__name__)


class ContextAnalysisNode(BaseNode):
    """Table-level context analysis node.

    Analyze the whole table to infer its business purpose and overall
    data category, providing global context for subsequent field-level
    classification. This improves accuracy when all fields in the
    table share a common business context.
    """

    def __init__(self, llm: BaseLLM):
        super().__init__(llm)

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        """Process table context analysis

        If multiple fields (inputs list) is provided, use all fields
        to understand the table's overall purpose.
        """
        # Get table information
        if state.get("inputs") and len(state["inputs"]) > 0:
            # Multiple fields (whole table)
            table_name = state["inputs"][0]["table_name"]
            fields = [
                {"field_name": f["field_name"], "field_description": f.get("field_description")}
                for f in state["inputs"]
            ]
        else:
            # Single field - still do context analysis on the table
            table_name = state["input"]["table_name"]
            fields = [
                {"field_name": state["input"]["field_name"],
                 "field_description": state["input"].get("field_description")}
            ]

        table_chinese_name = state.get("table_chinese_name")

        prompt = load_prompt(
            "context_analysis.jinja2",
            table_name=table_name,
            table_chinese_name=table_chinese_name,
            fields=fields,
        )

        result = self.llm.generate_json(prompt)

        # Validate required keys
        required_keys = [
            "inferred_purpose",
            "key_business_concepts",
            "overall_data_category",
        ]
        for key in required_keys:
            if key not in result:
                result[key] = result.get(key, "")

        if not isinstance(result.get("key_business_concepts"), list):
            # Fallback: split string into list
            if isinstance(result["key_business_concepts"], str):
                result["key_business_concepts"] = [
                    kw.strip() for kw in result["key_business_concepts"].split(",") if kw.strip()
                ]
            else:
                result["key_business_concepts"] = []

        context_analysis = {
            "table_name": table_name,
            "table_chinese_name": table_chinese_name,
            "inferred_purpose": result["inferred_purpose"],
            "key_business_concepts": result["key_business_concepts"],
            "overall_data_category": result["overall_data_category"],
        }

        logger.info(
            "ContextAnalysisNode - Table context analyzed: purpose=%s, category=%s, %d keywords",
            context_analysis["inferred_purpose"][:50] + "..." if len(context_analysis["inferred_purpose"]) > 50 else context_analysis["inferred_purpose"],
            context_analysis["overall_data_category"],
            len(context_analysis["key_business_concepts"]),
        )

        return {
            "table_context_analysis": context_analysis,
        }
