"""
Example: Using BGE Local Embeddings with RAG

This example shows how to use BAAI/bge-large-zh-v1.5 local embedding model
instead of OpenAI API embeddings for RAG.

Requirements:
    pip install transformers torch
"""
import os
from dotenv import load_dotenv
from classification_agent import ClassificationAgent
from classification_agent.llm.openai_wrapper import OpenAILLM
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES

# Load environment variables
load_dotenv()

# Get main LLM credentials (can be MiniMax or OpenAI)
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL", "MiniMax-M2.5")

# Initialize main LLM (still needs an LLM for classification, only embeddings are local)
llm = OpenAILLM(
    api_key=api_key,
    model=model,
    base_url=base_url,
)


def main():
    print("=" * 60)
    print("Example: Classification with RAG using local BGE embeddings")
    print("=" * 60)

    # Some manually labeled examples to add to RAG
    # Format: [(input_field, correct_label), ...]
    labeled_examples = [
        (
            {
                "table_name": "用户登录日志",
                "field_name": "device_id",
                "field_description": "设备唯一标识符",
            },
            {
                "level1": "个人基本资料",
                "level2": "自然人设备标识",
                "data_item": "设备标识符",
                "data_subitems": [],
            },
        ),
        (
            {
                "table_name": "订单支付表",
                "field_name": "payment_amount",
                "field_description": "支付金额",
            },
            {
                "level1": "交易数据",
                "level2": "交易金额",
                "data_item": "交易金额",
                "data_subitems": [],
            },
        ),
        (
            {
                "table_name": "用户收货地址",
                "field_name": "receiver_phone",
                "field_description": "收货人联系电话",
            },
            {
                "level1": "个人基本资料",
                "level2": "自然人联系方式",
                "data_item": "电话号码",
                "data_subitems": [],
            },
        ),
    ]

    # Create agent with RAG enabled using BGE local embeddings
    # You can configure these via .env file as well
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=DEFAULT_DATA_CATEGORIES,
        enable_rag=True,             # Enable RAG
        rag_embedding_provider="bge", # Use local BGE instead of OpenAI
        rag_top_k=3,                 # Return top 3 most similar examples
        rag_similarity_threshold=0.5, # Only return examples with similarity >= 0.5
        rag_examples=labeled_examples, # Add initial labeled examples
        # BGE specific options (optional, defaults shown below):
        # bge_model_name="BAAI/bge-large-zh-v1.5",  # Can use local path
        # bge_device="cpu",                         # Use "cuda" for GPU
        # bge_use_fp16=True,                        # Half precision for faster inference
        confidence_threshold=0.7,
        allow_multiple=True,
    )

    print(f"\nRAG initialized with {len(agent.get_rag_examples())} labeled examples")
    print(f"Embedding provider: BGE ({agent.embeddings.model_name}) on {agent.embeddings.device}")
    print("-" * 60)

    # Example 1: Classify a similar field to what's in RAG
    print("\nExample 1: Classify customer phone number with BGE RAG")
    print("-" * 40)
    input_data = {
        "table_name": "用户订单表",
        "field_name": "customer_mobile",
        "field_description": "下单客户手机号码",
    }
    result = agent.classify(input_data, enable_rag=True)

    print(f"\nInput:")
    print(f"  Table: {input_data['table_name']}")
    print(f"  Field: {input_data['field_name']}")
    if input_data.get('field_description'):
        print(f"  Description: {input_data['field_description']}")

    print(f"\nResult:")
    print(f"  Final labels: {result['final_labels']}")
    print(f"  Confidence: {result['final_confidence']:.2f}")
    print(f"  Dominant source: {result['feature_analysis']['dominant_source']}")

    if "rag_retrieved_examples" in result:
        print(f"\nRetrieved {len(result['rag_retrieved_examples'])} similar examples from RAG:")
        for i, item in enumerate(result['rag_retrieved_examples'], 1):
            ex_input = item['input']
            ex_label = item['label']
            score = item['similarity_score']
            print(f"  {i}. [{score:.3f}] {ex_input['table_name']}.{ex_input['field_name']} → {ex_label['data_item']}")

    print("\n" + "=" * 60)

    # Example 2: Classify without RAG (dynamic disable)
    print("\nExample 2: Classify without RAG (dynamic disable)")
    print("-" * 40)
    input_data2 = {
        "table_name": "交易明细表",
        "field_name": "trade_price",
        "field_description": "交易商品价格",
    }
    result2 = agent.classify(input_data2, enable_rag=False)

    print(f"\nInput:")
    print(f"  Table: {input_data2['table_name']}")
    print(f"  Field: {input_data2['field_name']}")

    print(f"\nResult:")
    print(f"  Final labels: {result2['final_labels']}")
    print(f"  Confidence: {result2['final_confidence']:.2f}")


if __name__ == "__main__":
    main()
