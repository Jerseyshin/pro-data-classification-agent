# pro-data-classification-agent

Hierarchical table field classification agent with thinking-verification architecture, built with LangGraph.

## Overview

This agent performs table field classification with a 4-stage thinking-verification architecture:
1. **Feature Analysis**: Separately analyze table name, field name, and field description. Extract keywords, analyze consistency, and identify which source should be more dominant.
2. **Preliminary Classification**: Give initial multi-label predictions based on feature analysis.
3. **Self-Verification**: Cross-validate each prediction by checking "field-name-only" vs "combined with table name". Remove false positives caused by keyword ambiguity.
4. **Final Result**: Output verified classification with confidence.

Key feature: **Dynamic weighting** of table name vs field name based on context:
- When field name is ambiguous (e.g., `deviceidtail`), table name context is more important
- When field name is clear (e.g., `network`), field name itself is sufficient

## Classification Hierarchy

The agent supports 4-level hierarchical classification:
- **Level 1**: Most general category
- **Level 2**: More specific middle category
- **Data Item**: **Target label to predict** (this is what you care about)
- **Data Subitem**: Fine-grained leaf nodes under data item (multiple allowed)

Example:
```
个人基本信息 (一级) → 自然人基本信息 (二级) → 人口属性 (数据项，目标) → 生日 (数据子项)
```

## Installation

```bash
poetry install
```

Or with pip:
```bash
pip install langgraph langchain-core openai pydantic jinja2 python-dotenv
```

## Configuration

Create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=optional-base-url
```

## Usage

### Basic Usage (with predefined categories)

```python
from classification_agent import ClassificationAgent
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
from classification_agent.llm.openai_wrapper import OpenAILLM
import os
from dotenv import load_dotenv

load_dotenv()
llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"))

agent = ClassificationAgent(
    llm=llm,
    hierarchical_categories=DEFAULT_DATA_CATEGORIES,
    confidence_threshold=0.7,
    allow_multiple=True  # allow multiple labels per field
)

input_data = {
    "table_name": "框架应用市场下载结果日志",
    "field_name": "deviceidtail",
    "field_description": "设备ID尾部"
}

result = agent.classify(input_data)
print(result["final_labels"])  # ["用户标识"]
print(result["final_confidence"])  # 0.85
print(result["feature_analysis"]["dominant_source"])  # "table_name"
```

### Dynamic Categories

You can pass custom hierarchical categories at classification time:

```python
custom_categories = [
    {
        "level1": "工单分类",
        "level2": "问题类型",
        "data_item": "Bug报告",
        "data_subitems": [
            {"name": "崩溃", "description": "应用崩溃异常退出"},
            {"name": "功能异常", "description": "功能不按预期工作"}
        ]
    },
    # ... more categories
]

result = agent.classify(input_data, hierarchical_categories=custom_categories)
```

### CLI Usage

```bash
python main.py --table "框架应用市场下载结果日志" --field "deviceidtail" --description "设备ID尾部"
```

## Examples

See the `examples/` directory:
- `basic_usage.py` - Basic usage with predefined categories
- `dynamic_categories.py` - Using dynamically provided categories

## Testing

```bash
# Unit tests
pytest tests/ -xvs

# Integration tests (requires OpenAI API key)
pytest tests/test_integration.py -xvs -m integration
```

## Project Structure

```
src/classification_agent/
├── agent.py                 # ClassificationAgent facade
├── graph/
│   ├── state.py             # State definition
│   ├── edges.py             # Routing logic
│   └── builder.py           # LangGraph builder
├── nodes/
│   ├── base_node.py         # Base node class
│   ├── feature_analysis.py  # Feature analysis node
│   ├── preliminary_classification.py  # Preliminary classification
│   ├── self_verification.py   # Self-verification node
│   └── final_result.py      # Final result aggregation
├── prompts/
│   ├── feature_analysis.jinja2
│   ├── preliminary_classification.jinja2
│   └── self_verification.jinja2
├── config/
│   ├── settings.py          # Settings
│   └── default_categories.py  # Example categories
├── llm/
│   ├── base.py              # LLM abstraction
│   └── openai_wrapper.py    # OpenAI implementation
├── types/
│   └── schemas.py           # Type definitions
└── utils/
    ├── logging.py           # Logging
    └── validation.py        # Input validation
```

## License

MIT
