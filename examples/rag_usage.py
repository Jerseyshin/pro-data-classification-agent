#!/usr/bin/env python3
"""
Example: Classification with RAG similarity search
Demonstrates how to:
- Initialize agent with RAG enabled
- Add labeled examples
- Classify with RAG assistance
"""

import os
from dotenv import load_dotenv
from classification_agent import ClassificationAgent
from classification_agent.llm.openai_wrapper import OpenAILLM
from classification_agent.types.schemas import TableFieldInput, HierarchicalCategory

# Load environment variables
load_dotenv()


def main():
    # Initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable must be set")
        exit(1)

    llm = OpenAILLM(api_key=api_key, model=model)

    # Use default categories from the package
    from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
    categories = DEFAULT_DATA_CATEGORIES

    # Your labeled examples (these would be manually labeled gold examples)
    # Format: List[Tuple[TableFieldInput, HierarchicalCategory]]
    labeled_examples = [
        (
            {
                "table_name": "user_login_log",
                "field_name": "device_id",
                "field_description": "Device unique identifier for login tracking",
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
                "table_name": "order_payment",
                "field_name": "payment_amount",
                "field_description": "Transaction payment amount in CNY",
            },
            {
                "level1": "交易数据",
                "level2": "交易主体",
                "data_item": "交易金额",
                "data_subitems": [],
            },
        ),
        (
            {
                "table_name": "user_profile",
                "field_name": "phone_number",
                "field_description": "User's mobile phone number for contact",
            },
            {
                "level1": "个人基本资料",
                "level2": "自然人联系方式",
                "data_item": "电话号码",
                "data_subitems": [],
            },
        ),
    ]

    # Create agent with RAG enabled
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=categories,
        enable_rag=True,             # Enable RAG
        rag_top_k=3,                 # Return top 3 similar examples
        rag_similarity_threshold=0.5,  # Only return examples with similarity >= 0.5
        rag_examples=labeled_examples, # Initial examples to add
        confidence_threshold=0.7,
        allow_multiple=True,
    )

    # You can also add more examples after initialization:
    # agent.add_rag_examples(more_examples)

    print("=" * 60)
    print("Example 1: Classify 'customer_id' in 'customer_order' table")
    print("=" * 60)

    # Classify a new field - RAG will activate automatically
    result = agent.classify({
        "table_name": "customer_order",
        "field_name": "customer_phone",
        "field_description": "Customer's contact phone number",
    })

    print(f"\nInput:")
    print(f"  Table: customer_order")
    print(f"  Field: customer_phone")
    print(f"  Description: Customer's contact phone number")

    print(f"\nResult:")
    print(f"  Final labels: {result['final_labels']}")
    print(f"  Confidence: {result['final_confidence']:.2f}")
    print(f"  Dominant source: {result['feature_analysis']['dominant_source']}")

    print("\n" + "=" * 60)
    print("Example 2: Classify 'transaction_value' without RAG (dynamic disable)")
    print("=" * 60)

    # You can dynamically disable RAG for specific classifications
    result2 = agent.classify(
        {
            "table_name": "payment_transaction",
            "field_name": "transaction_value",
            "field_description": "Value of the payment transaction",
        },
        enable_rag=False,
    )

    print(f"\nInput:")
    print(f"  Table: payment_transaction")
    print(f"  Field: transaction_value")

    print(f"\nResult:")
    print(f"  Final labels: {result2['final_labels']}")
    print(f"  Confidence: {result2['final_confidence']:.2f}")


if __name__ == "__main__":
    main()
