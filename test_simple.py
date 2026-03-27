#!/usr/bin/env python3
"""
简化测试脚本 - 测试分类功能（不使用RAG）
"""

import sys
import os
from pathlib import Path

# 添加 src 目录到 Python 路径
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classification_agent import ClassificationAgent
from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
from classification_agent.llm.openai_wrapper import OpenAILLM
from classification_agent.utils.data_reader import load_data_csv
from dotenv import load_dotenv

load_dotenv()


def main():
    # 初始化 LLM
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("错误: OPENAI_API_KEY 环境变量未设置")
        return

    llm = OpenAILLM(api_key=api_key, model=model, base_url=base_url)

    # 初始化分类代理
    print("初始化 ClassificationAgent...")
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=DEFAULT_DATA_CATEGORIES,
        confidence_threshold=0.7,
        allow_multiple=True,
        enable_table_context=True,
        enable_rag=False,  # 禁用 RAG
    )

    # 加载测试数据（只加载前3个字段以加快测试）
    print("加载测试数据...")
    inputs, ground_truth_list = load_data_csv(
        "data.csv",
        include_ground_truth=True,
        skip_empty_gt=True,
        encoding="utf-8-sig",
    )

    # 只测试前3个字段
    test_inputs = inputs[:3]
    test_ground_truth = ground_truth_list[:3] if ground_truth_list else None

    print(f"测试 {len(test_inputs)} 个字段...")

    # 运行分类
    result = agent.classify_table(
        fields=test_inputs,
        ground_truth_list=test_ground_truth,
    )

    # 输出结果
    results = result.get("results", [])
    evaluation = result.get("evaluation")

    print(f"\n分类完成: {len(results)} 个字段处理完毕")

    if evaluation:
        print(f"\n评估结果:")
        print(f"  总样本数: {evaluation['total_samples']}")
        print(f"  精确匹配准确率: {evaluation['exact_match_accuracy']:.4f}")
        print(f"  Macro F1: {evaluation['macro_f1']:.4f}")

    # 显示每个字段的结果
    print(f"\n详细结果:")
    for i, (input_field, result_field) in enumerate(zip(test_inputs, results)):
        print(f"\n字段 {i + 1}: {input_field['field_name']}")
        print(f"  最终标签: {result_field.get('final_labels', [])}")
        print(f"  最终置信度: {result_field.get('final_avg_confidence', 0.0):.2f}")
        if test_ground_truth:
            print(f"  真实标签: {test_ground_truth[i]}")
            match = result_field.get("evaluation_match", False)
            print(f"  匹配: {'是' if match else '否'}")


if __name__ == "__main__":
    main()
