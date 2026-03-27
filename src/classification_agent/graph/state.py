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
    ClassificationResult,
    TableContextAnalysis,
)


# Bulk mode results (table-level processing, all fields in one go)
class BulkFeatureAnalysisResult(TypedDict):
    """批量特征分析结果 - 整张表所有字段一次性分析完成"""
    # 每个字段对应一个特征分析结果
    field_analyses: List[FeatureAnalysisResult]


class BulkPreliminaryResult(TypedDict):
    """批量初步分类结果 - 整张表所有字段一次性分类完成"""
    # 每个字段对应一个初步分类结果
    field_results: List[PreliminaryResult]


class BulkVerificationResult(TypedDict):
    """批量验证结果 - 整张表所有字段一次性验证完成"""
    # 每个字段对应一个验证结果
    field_verifications: List[VerificationResult]


class ClassificationState(TypedDict):
    """LangGraph agent 状态"""
    # 输入：可以是单个字段或整张表多个字段（批量模式）
    input: TableFieldInput                              # 当前正在处理的单个输入字段（保持兼容单字段模式）
    inputs: Optional[List[TableFieldInput]]            # 整张表所有输入字段（批量模式原始输入）
    remaining_inputs: Optional[List[TableFieldInput]] # 批量模式中剩余待处理字段（兼容原有串行批量）
    table_chinese_name: Optional[str]                  # 表中文名（整表上下文分析用）

    # 分类体系配置
    hierarchical_categories: List[HierarchicalCategory]  # 完整层级分类体系（所有候选类别）
    confidence_threshold: float                         # 置信度阈值，低于此建议重分类
    allow_multiple: bool                                 # 是否允许多标签输出
    rag_enabled: bool                                    # 是否启用RAG检索
    bulk_mode: bool                                      # 是否启用表级批量模式（一次处理所有字段）

    # 评估：真实标签（可选，提供后会在最后运行评估）
    # 对于批量模式：每个输入都有对应的ground truth
    ground_truth_data_items: Optional[List[str]]        # 当前字段真实数据项标签
    ground_truth_list: Optional[List[List[str]]]       # 批量输入：所有字段的真实标签列表
    remaining_ground_truth: Optional[List[List[str]]]  # 批量模式中剩余待处理真实标签

    # 表级上下文分析（在单个字段分析前先分析整张表）
    table_context_analysis: Optional[TableContextAnalysis] # 表级上下文分析结果

    evaluation: Optional[EvaluationResult]              # 最终评估结果（由 EvaluationNode 写入）

    # 中间结果（共享）
    retrieved_examples: Optional[List[RetrievedExample]]  # RAG检索到的相似标注样例

    # 单字段模式（原有逻辑保留）
    feature_analysis: Optional[FeatureAnalysisResult]    # 特征分析结果（单字段）
    preliminary_classification: Optional[PreliminaryResult]  # 初步分类结果（单字段）
    verification: Optional[VerificationResult]           # 自我验证结果（单字段）

    # 批量模式：整张表所有字段一次性处理结果
    bulk_feature_analysis: Optional[BulkFeatureAnalysisResult]  # 批量特征分析（所有字段一次LLM）
    bulk_preliminary_classification: Optional[BulkPreliminaryResult] # 批量初步分类（一次LLM）
    bulk_verification: Optional[BulkVerificationResult]  # 批量验证（一次LLM）

    # 批量模式（原有串行）：已处理完的所有字段结果
    completed_results: Optional[List[ClassificationResult]] # 已完成的分类结果列表

    # 迭代计数（防止无限循环）

    reclassification_count: int                          # 已经重新分类的次数，限制最多1次（针对当前字段）



    # 记录之前被确定性检查剔除的幻觉预测，重分类时不要重复预测

    hallucinated_data_items: List[str]                  # 之前被发现是幻觉的数据项名称列表



    # 最终输出（单字段/批量都用这个汇总）

    _final_predictions: Optional[List[PredictedItem]]    # 最终预测列表（单字段）

    _final_labels: Optional[List[str]]                   # 最终标签列表（单字段）

    _final_avg_confidence: Optional[float]               # 最终平均置信度（单字段）



    # 批量模式最终结果：所有字段的结果

    bulk_final_results: Optional[List[ClassificationResult]] # 批量最终结果（所有字段）
