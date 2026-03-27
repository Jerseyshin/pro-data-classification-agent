"""
Context Analysis Node - Analyze whole table context before field-level classification.

This node:
1. Takes the whole table information (table name, table chinese name, all fields)
2. Performs comprehensive business context analysis in 5 aspects
3. Provides global context for subsequent field-level classification
"""
from typing import Any, Dict, Optional, List

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.prompts.loader import load_prompt
from classification_agent.utils.logging import get_logger
from .base_node import BaseNode

logger = get_logger(__name__)


class ContextAnalysisNode(BaseNode):
    """Table-level context analysis node.

    Analyze the whole table to infer comprehensive business context in 5 aspects,
    providing global context for subsequent field-level classification.
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

        # Validate required keys for new format
        required_keys = [
            "business_scenario",
            "table_type",
            "core_business_objects",
            "key_business_concepts",
            "overall_description",
        ]
        for key in required_keys:
            if key not in result:
                result[key] = result.get(key, "")

        # Ensure lists are proper lists
        def ensure_list(value, default=None):
            if default is None:
                default = []
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                # Split by comma or other delimiters
                items = [item.strip() for item in value.replace('，', ',').split(',') if item.strip()]
                return items
            else:
                return default

        core_business_objects = ensure_list(result.get("core_business_objects"))
        key_business_concepts = ensure_list(result.get("key_business_concepts"))

        context_analysis = {
            "table_name": table_name,
            "table_chinese_name": table_chinese_name,
            "business_scenario": result["business_scenario"],
            "table_type": result["table_type"],
            "core_business_objects": core_business_objects,
            "key_business_concepts": key_business_concepts,
            "overall_description": result["overall_description"],
        }

        logger.info(
            "ContextAnalysisNode - Table context analyzed: scenario='%s', type=%s, %d core objects, %d concepts",
            context_analysis["business_scenario"][:60] + "..." if len(context_analysis["business_scenario"]) > 60 else context_analysis["business_scenario"],
            context_analysis["table_type"],
            len(context_analysis["core_business_objects"]),
            len(context_analysis["key_business_concepts"]),
        )

        return {
            "table_context_analysis": context_analysis,
        }
