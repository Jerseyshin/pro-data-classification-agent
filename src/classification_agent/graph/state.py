from typing import TypedDict, List, Optional
from classification_agent.types.schemas import (
    TableFieldInput,
    HierarchicalCategory,
    FeatureAnalysisResult,
    PreliminaryResult,
    VerificationResult,
    PredictedItem,
    RetrievedExample,
    EvaluationResult,
)


class ClassificationState(TypedDict):
    """LangGraph agent 状态"""
    # 输入
    input: TableFieldInput                              # 输入（表名 + 字段名 + 字段描述）

    # 分类体系配置
    hierarchical_categories: List[HierarchicalCategory]  # 完整层级分类体系（所有候选类别）
    confidence_threshold: float                         # 置信度阈值，低于此建议重分类
    allow_multiple: bool                                 # 是否允许多标签输出
    rag_enabled: bool                                    # 是否启用RAG检索

    # 评估：真实标签（可选，提供后会在最后运行评估）
    ground_truth_data_items: Optional[List[str]]        # 真实数据项标签，提供则运行评估节点
    evaluation: Optional[EvaluationResult]              # 评估结果（由 EvaluationNode 写入）

    # 中间结果
    retrieved_examples: Optional[List[RetrievedExample]]  # RAG检索到的相似标注样例

    # 中间结果
    feature_analysis: Optional[FeatureAnalysisResult]    # 特征分析结果
    preliminary_classification: Optional[PreliminaryResult]  # 初步分类结果
    verification: Optional[VerificationResult]           # 自我验证结果

    # 迭代计数（防止无限循环）
    reclassification_count: int                          # 已经重新分类的次数，限制最多1次

    # 最终输出（由 FinalResultNode 写入）
    _final_predictions: Optional[List[PredictedItem]]    # 最终预测列表
    _final_labels: Optional[List[str]]                   # 最终标签列表
    _final_avg_confidence: Optional[float]               # 最终平均置信度
    _reasoning_chain: Optional[List[str]]                # 完整推理链
