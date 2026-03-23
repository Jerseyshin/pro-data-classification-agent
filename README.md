# pro-data-classification-agent

Hierarchical table field classification agent with thinking-verification architecture, built with LangGraph.

## Overview

This agent performs table field classification with a thinking-verification architecture:
- **Table Context Analysis** *(optional)*: Analyze the whole table first to infer overall business purpose, key concepts, and data category. Provides global context for better field-level classification.
- **Feature Analysis**: Separately analyze table name, field name, and field description. Extract keywords, analyze consistency, and identify which source should be more dominant.
- **Preliminary Classification**: Give initial multi-label predictions based on feature analysis.
- **Self-Verification**: Cross-validate each prediction by checking "field-name-only" vs "combined with table name". Remove false positives caused by keyword ambiguity.
- **Final Result**: Output verified classification with confidence.
- **Evaluation** *(optional)*: Calculate accuracy metrics when ground truth is provided.

Key features:
- **Dynamic weighting** of table name vs field name based on context:
  - When field name is ambiguous (e.g., `deviceidtail`), table name context is more important
  - When field name is clear (e.g., `network`), field name itself is sufficient
- **Whole-table batch classification**: Classify all fields in one LLM call for massive speedup
- **RAG support**: Retrieve similar labeled examples to improve accuracy
- **Table-level context analysis**: Better accuracy when classifying multiple fields from the same table

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

# Speed optimization
FAST_MODE=false  # merge feature+classification into one LLM call → ~33% faster
MAX_CONCURRENCY=5  # concurrent processing for batch evaluation

# RAG settings
ENABLE_RAG=false
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.5
RAG_EMBEDDING_PROVIDER=openai  # or "bge" for local BAAI/bge-large-zh-v1.5
RAG_EMBEDDING_MODEL=text-embedding-3-small
# Optional: separate OpenAI credentials for RAG embeddings
# RAG_OPENAI_API_KEY=your-rag-api-key
# RAG_OPENAI_BASE_URL=https://api.openai.com/v1
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

### Whole Table Classification (batch mode)

Classify all fields of a table in **one LLM call** (much faster than one-by-one):

```python
# Load all fields from your table
inputs = [
    {"table_name": "user_table", "field_name": "user_id", "field_description": "user unique id"},
    {"table_name": "user_table", "field_name": "user_name", "field_description": "user full name"},
    {"table_name": "user_table", "field_name": "email", "field_description": "user contact email"},
    # ... more fields
]

# Classify all fields in ONE LLM call
results = agent.classify_table(
    fields=inputs,
    table_chinese_name="用户基础信息表",  # optional
)

# results is List[ClassificationResult], one per field
for field, result in zip(inputs, results):
    print(field["field_name"], result["final_labels"])
```

The `classify_table` method automatically:
1. Does table-level context analysis (if enabled)
2. Does feature analysis + classification for all fields in one call
3. Returns a list of results in the same order as input
4. Includes evaluation if you provide `ground_truth_list`

### Load RAG Training Data from CSV

Load labeled examples from CSV and add to RAG:

```python
from classification_agent.utils.data_reader import load_rag_training_data
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES

# Load from CSV (same format as data.csv)
# Columns: 数据域,表名,表中文名,表字段,字段描述,字段隐私四级分类
rag_examples = load_rag_training_data(
    csv_path="your_training_data.csv",
    hierarchical_categories=DEFAULT_DATA_CATEGORIES,
    skip_if_not_found=True,
)

# Add to agent
agent.add_rag_examples(rag_examples)
```

### CLI Usage

```bash
python main.py --table "框架应用市场下载结果日志" --field "deviceidtail" --description "设备ID尾部"
```

## Examples

See the `examples/` directory:
- `basic_usage.py` - Basic usage with predefined categories
- `dynamic_categories.py` - Using dynamically provided categories
- `rag_usage.py` - Using RAG with OpenAI embeddings
- `bge_rag_usage.py` - Using RAG with local BGE embeddings (BAAI/bge-large-zh-v1.5)
- `batch_evaluation.py` - Batch evaluation on `data.csv` with accuracy metrics
- `whole_table_classification.py` - Classify all fields of a table in one LLM call
- `load_rag_from_csv.py` - Load RAG training data from CSV and add to agent

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
│   ├── context_analysis.py  # Table-level context analysis (NEW)
│   ├── feature_analysis.py  # Feature analysis node
│   ├── preliminary_classification.py  # Preliminary classification
│   ├── combined_feature_classification.py  # Combined node for fast mode
│   ├── self_verification.py   # Self-verification node
│   ├── final_result.py      # Final result aggregation
│   ├── rag_retrieval.py     # RAG similarity retrieval
│   └── evaluation.py        # Evaluation when ground truth provided
├── prompts/
│   ├── context_analysis.jinja2  # Table context prompt (NEW)
│   ├── feature_analysis.jinja2
│   ├── preliminary_classification.jinja2
│   ├── feature_classification_combined.jinja2  # Combined prompt for fast mode
│   ├── batch_table_classification.jinja2  # Whole-table batch prompt (NEW)
│   └── self_verification.jinja2
├── config/
│   ├── settings.py          # Settings
│   └── default_categories.py  # Example categories loaded from label.csv
├── llm/
│   ├── base.py              # LLM abstraction
│   └── openai_wrapper.py    # OpenAI implementation
├── rag/
│   ├── embeddings.py        # Embedding abstraction + OpenAI + BGE
│   └── vector_store.py      # In-memory vector store
├── types/
│   └── schemas.py           # Type definitions
└── utils/
    ├── logging.py           # Logging
    ├── validation.py        # Input validation
    ├── data_reader.py       # Load data.csv + load RAG training data
    └── result_exporter.py   # Export results to CSV/Markdown
```

## Speed Optimization

For faster batch processing:

| Optimization | Speedup | Description |
|-------------|---------|-------------|
| `FAST_MODE=true` | **~33% faster** | Merge feature analysis + preliminary classification into one LLM call, reduces network round-trips |
| `MAX_CONCURRENCY=N` (N > 1) | **~Nx faster** | Process multiple samples concurrently |
| `classify_table()` | **~Nx faster** | Classify all N fields of a table in **one** LLM call instead of N separate calls |

## RAG (Retrieval-Augmented Generation)

Improve accuracy by retrieving similar labeled examples:

```python
# Enable RAG when creating agent
agent = ClassificationAgent(
    llm=llm,
    hierarchical_categories=DEFAULT_DATA_CATEGORIES,
    enable_rag=True,
    rag_top_k=5,
)

# Add labeled examples
agent.add_rag_examples([
    (input_field, correct_category),
    ...
])

# Or load from CSV
from classification_agent.utils.data_reader import load_rag_training_data
rag_examples = load_rag_training_data("training.csv", DEFAULT_DATA_CATEGORIES)
agent.add_rag_examples(rag_examples)
```

Supports two embedding providers:
- `openai`: Use OpenAI API embeddings (good quality, requires API key)
- `bge`: Use local BAAI/bge-large-zh-v1.5 model (good for Chinese, runs locally, no API key needed)

## License

MIT
