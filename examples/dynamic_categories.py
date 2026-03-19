#!/usr/bin/env python3
"""
动态分类示例：在分类时动态传入层级分类体系
"""
import os
from dotenv import load_dotenv
from classification_agent import ClassificationAgent
from classification_agent.llm.openai_wrapper import OpenAILLM

load_dotenv()


def main():
    # Initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    llm = OpenAILLM(api_key=api_key, model=model)

    # Initialize agent without pre-defined categories
    agent = ClassificationAgent(
        llm=llm,
        confidence_threshold=0.7,
        allow_multiple=True
    )

    # Define custom hierarchical categories dynamically
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
        {
            "level1": "工单分类",
            "level2": "问题类型",
            "data_item": "功能请求",
            "data_subitems": [
                {"name": "新功能", "description": "请求新增功能"},
                {"name": "改进建议", "description": "对现有功能改进建议"}
            ]
        },
        {
            "level1": "工单分类",
            "level2": "问题类型",
            "data_item": "使用咨询",
            "data_subitems": [
                {"name": "使用方法", "description": "询问如何使用功能"},
                {"name": "技术问题", "description": "技术相关问题"}
            ]
        }
    ]

    # Classify with dynamic categories
    print("=" * 60)
    print("Dynamic categories example - classifying support ticket")
    print("=" * 60)

    input_data = {
        "table_name": "用户工单表",
        "field_name": "ticket_content",
        "field_description": "工单内容描述"
    }

    input_text = "我的应用在点击提交按钮的时候总是崩溃退出"
    print(f"\nTicket content: {input_text}")

    # Note: In this example, the input field content is in field_description
    # Actually for this case, you would classify each ticket as a separate input
    input_data = {
        "table_name": "用户工单表",
        "field_name": "content",
        "field_description": input_text
    }

    result = agent.classify(input_data, hierarchical_categories=custom_categories)

    print(f"\nResult:")
    print(f"  Final labels: {result['final_labels']}")
    print(f"  Confidence: {result['final_confidence']:.2f}")
    print(f"  Reasoning chain length: {len(result['reasoning_chain'])}")
    print()


if __name__ == "__main__":
    main()
