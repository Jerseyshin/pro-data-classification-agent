from typing import TypedDict, List, Optional


class TableFieldInput(TypedDict):
    """输入：表格字段元数据"""
    table_name: str        # 表名称，例如 "搜索团队基础表-交易记录表"
    field_name: str        # 字段名称，例如 "user_id"
    field_description: Optional[str]  # 字段描述，例如 "账号ID"


class DataSubitem(TypedDict):
    """数据子项：最细粒度叶子节点"""
    name: str              # 数据子项名称
    description: Optional[str]  # 数据子项描述


class HierarchicalCategory(TypedDict):
    """层级分类结构
    level1 -> level2 -> data_item -> [data_subitem*]
    预测目标是 data_item
    """
    level1: str                          # 一级分类，最概括的上层分类
    level2: str                          # 二级分类，更具体的中层分类
    data_item: str                       # 数据项，目标预测标签
    data_subitems: List[DataSubitem]     # 数据子项列表，多个细粒度叶子节点


class FeatureAnalysisResult(TypedDict):
    """特征分析结果"""
    table_name_keywords: List[str]      # 从表名提取的业务场景关键词
    field_name_keywords: List[str]      # 从字段名提取的关键词/词根
    description_keywords: List[str]     # 从字段描述提取的关键词
    semantic_summary: str               # 综合推断的业务含义
    consistency_analysis: str           # 分析不同来源信息是否一致
    dominant_source: str                # 哪个来源更有决定性："table_name", "field_name", "description"


class PredictedItem(TypedDict):
    """单个预测结果"""
    level1: str                      # 一级分类
    level2: str                      # 二级分类
    data_item: str                   # 预测的数据项（目标标签）
    matching_data_subitems: List[str]  # 匹配的数据子项名称
    confidence: float                 # 置信度 0-1
    reasoning: str                    # 推理过程


class PreliminaryResult(TypedDict):
    """初步分类结果"""
    predictions: List[PredictedItem]    # 预测列表
    total_confidence: float             # 整体平均置信度


class VerifiedItem(TypedDict):
    """验证后的单个预测结果"""
    original_prediction: PredictedItem    # 原始预测
    is_kept: bool                          # 是否保留
    adjusted_confidence: float             # 调整后的置信度
    verification_note: str                  # 验证说明


class VerificationResult(TypedDict):
    """自我验证结果"""
    verified_predictions: List[VerifiedItem]    # 验证后的预测列表
    removed_false_positives: List[str]           # 被剔除的误匹配数据项名称
    added_missing: List[PredictedItem]           # 补充遗漏的匹配
    average_confidence: float                    # 验证后的平均置信度
    cross_validation_note: str                   # 交叉验证（仅字段名 vs 结合表名）的结论
    suggests_reclassification: bool             # 是否建议重新分类


class RetrievedExample(TypedDict):
    """Retrieved similar labeled example from RAG"""
    input: TableFieldInput          # Original input (table name, field name, description)
    label: HierarchicalCategory     # The correct label for this example
    similarity_score: float         # Similarity score (0-1)


class SingleEvaluationResult(TypedDict):
    """单个样本的评估结果"""
    predicted_data_items: List[str]      # 预测的数据项标签
    ground_truth_data_items: List[str]   # 真实数据项标签
    correct_predictions: List[str]       # 预测正确的标签
    wrong_predictions: List[str]         # 预测错误（预测了但真实没有）
    missing_predictions: List[str]       # 缺失（真实有但没预测出来）
    exact_match: bool                    # 是否完全匹配（集合相等）


class EvaluationResult(TypedDict):
    """批量评估总体结果"""
    total_samples: int                       # 总样本数
    exact_match_count: int                   # 完全匹配样本数
    exact_match_accuracy: float              # 完全匹配准确率 = 完全匹配样本数 / 总样本数
    total_true_positives: int                # 总共真阳性（预测对了）
    total_false_positives: int               # 总共假阳性（预测错了）
    total_false_negatives: int               # 总共假阴性（漏了）
    macro_precision: float                   # 宏平均精确率 = TP / (TP + FP)
    macro_recall: float                      # 宏平均召回率 = TP / (TP + FN)
    macro_f1: float                          # 宏平均 F1 = 2 * P * R / (P + R)
    per_sample_results: List[SingleEvaluationResult]  # 每个样本的详细结果


class ClassificationResult(TypedDict):
    """最终分类结果"""
    final_predictions: List[PredictedItem]     # 最终预测列表
    final_labels: List[str]                    # 最终数据项标签列表
    final_confidence: float                    # 最终整体置信度（平均）
    reasoning_chain: List[str]                 # 完整推理链
    feature_analysis: FeatureAnalysisResult    # 特征分析结果（可追溯）
    preliminary_result: PreliminaryResult      # 初步分类结果（可追溯）
    verification_result: VerificationResult    # 验证结果（可追溯）
    evaluation: Optional[EvaluationResult]    # 评估结果（仅当提供ground_truth时存在）
