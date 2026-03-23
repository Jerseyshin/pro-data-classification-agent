"""
Example: Load RAG training data from CSV and add to agent.

This example demonstrates:
1. How to use load_rag_training_data to load labeled examples from CSV
2. How to add them to the agent's RAG vector store
3. How to run classification with RAG
"""
import os
from dotenv import load_dotenv
from classification_agent import ClassificationAgent
from classification_agent.llm.openai_wrapper import OpenAILLM
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
from classification_agent.utils.data_reader import load_rag_training_data

# Load environment variables
load_dotenv()

# Get LLM credentials
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL", "MiniMax-M2.5")

# Initialize LLM
llm = OpenAILLM(
    api_key=api_key,
    model=model,
    base_url=base_url,
)


def main():
    print("=" * 60)
    print("Example: Load RAG Training Data from CSV")
    print("=" * 60)
    print("\nThis example shows how to load labeled examples from CSV\n"
          "and add them to the RAG vector store for similarity retrieval.\n")

    # Path to your RAG training CSV
    # Format is exactly the same as data.csv:
    #   Columns: 数据域,表名,表中文名,表字段,字段描述,字段隐私四级分类
    # Each labeled example becomes a (input, correct_label) tuple for RAG
    rag_csv_path = "data.csv"  # You can use your own CSV
    print(f"Loading RAG training data from: {rag_csv_path}")

    # Load RAG training data
    # This automatically matches the labels with hierarchical_categories
    rag_examples = load_rag_training_data(
        csv_path=rag_csv_path,
        hierarchical_categories=DEFAULT_DATA_CATEGORIES,
        skip_if_not_found=True,  # Skip labels that aren't in the category list
    )

    print(f"\nLoaded {len(rag_examples)} RAG training examples")
    if rag_examples:
        print(f"First example:")
        print(f"  Input: table={rag_examples[0][0]['table_name']}, field={rag_examples[0][0]['field_name']}")
        print(f"  Label: data_item={rag_examples[0][1]['data_item']}")

    if not rag_examples:
        print("\nNo examples loaded, exiting")
        return

    # Create agent with RAG enabled
    print("\nCreating ClassificationAgent with RAG enabled...")
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=DEFAULT_DATA_CATEGORIES,
        confidence_threshold=0.7,
        allow_multiple=True,
        enable_rag=True,
        rag_examples=rag_examples,  # <- Add loaded examples here
        rag_top_k=5,
        rag_similarity_threshold=0.5,
    )

    print(f"\nRAG initialized with {len(agent.get_rag_examples())} labeled examples")

    # Try a classification
    test_input = {
        "table_name": "dwd_browser_search_hispace_sur_base_joint_pay_dm - 搜索团队基础表-交易记录表",
        "field_name": "user_id",
        "field_description": "用户唯一标识",
    }

    print(f"\nTesting classification with RAG:")
    print(f"  Table: {test_input['table_name']}")
    print(f"  Field: {test_input['field_name']}")
    print(f"  Description: {test_input['field_description']}")

    result = agent.classify(**test_input)

    print(f"\nResult:")
    print(f"  Final labels: {result['final_labels']}")
    print(f"  Confidence: {result['final_confidence']:.4f}")
    print(f"  Dominant source: {result['feature_analysis']['dominant_source']}")

    print("\n✅ Done! RAG is ready to use with your labeled data from CSV.")


if __name__ == "__main__":
    main()
