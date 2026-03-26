from typing import Optional
from langgraph.graph import StateGraph, END

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.nodes import (
    ContextAnalysisNode,
    RAGRetrievalNode,
    EvaluationNode,
)

# Import bulk processing nodes
from classification_agent.nodes.bulk_feature_analysis import BulkFeatureAnalysisNode
from classification_agent.nodes.bulk_preliminary_classification import (
    BulkPreliminaryClassificationNode,
)
from classification_agent.nodes.bulk_self_verification import BulkSelfVerificationNode
from classification_agent.nodes.bulk_final_result import BulkFinalResultNode
from classification_agent.nodes.base_node import BaseNode
from classification_agent.config.settings import Settings
from classification_agent.rag.embeddings import BaseEmbeddings
from classification_agent.rag.vector_store import InMemoryVectorStore


def should_evaluate_router(state: ClassificationState) -> str:
    """Router: 如果有 ground_truth，就走评估节点，否则直接结束"""
    if state.get("ground_truth_list") is not None:
        return "evaluation_node"
    else:
        return "end"


def next_after_context_router(state: ClassificationState) -> str:
    """After context analysis, decide next step based on RAG enabled"""
    if state.get("rag_enabled") and state.get("retrieved_examples") is None:
        return "rag_retrieval_node"
    else:
        return "bulk_feature_analysis_node"


def after_rag_router(state: ClassificationState) -> str:
    """After RAG retrieval, where to go next"""
    return "bulk_feature_analysis_node"


def reclassification_router(state: ClassificationState) -> str:
    """检查是否需要重新分类

    条件:
    1. 有字段的 suggests_reclassification = True
    2. reclassification_count < 1 (允许最多1次重新分类)
    """
    # 检查是否有字段建议重新分类
    bulk_verification = state.get("bulk_verification")
    if not bulk_verification:
        return "bulk_final_result_node"

    field_verifications = bulk_verification.get("field_verifications", [])

    # 检查是否有字段建议重新分类
    suggests_reclassification = any(
        v.get("suggests_reclassification", False) for v in field_verifications
    )

    # 检查重新分类次数
    reclassification_count = state.get("reclassification_count", 0)

    if suggests_reclassification and reclassification_count < 1:
        return "bulk_preliminary_classification_node"
    else:
        return "bulk_final_result_node"


def build_graph(
    llm: BaseLLM,
    settings: Settings,
    vector_store: Optional[InMemoryVectorStore] = None,
    embeddings: Optional[BaseEmbeddings] = None,
    enable_table_context: bool = True,
) -> StateGraph:
    """构建LangGraph - 仅支持Bulk模式（整表全字段一次性处理）

    **Bulk模式流程（唯一支持的模式）：**
    1. context_analysis_node - 分析整表上下文（一次）
    2. rag_retrieval_node (可选) - 基于表格上下文检索相似示例
    3. bulk_feature_analysis_node - 一次性分析所有字段（一个LLM调用）
    4. bulk_preliminary_classification_node - 一次性对所有字段进行初步分类（一个LLM调用）
    5. bulk_self_verification_node - 一次性验证所有字段（一个LLM调用）
    6. bulk_final_result_node - 聚合所有结果
    7. evaluation_node (如果提供ground truth，则进行评估) -> END

    **Bulk模式LLM调用总数：3-4次，与字段数量无关**
    """

    # 创建节点
    context_analysis = ContextAnalysisNode(llm)
    evaluation = EvaluationNode(llm)

    # 创建批量处理节点（表级一次性处理）
    bulk_feature_analysis = BulkFeatureAnalysisNode(llm)
    bulk_preliminary_classification = BulkPreliminaryClassificationNode(llm)
    bulk_self_verification = BulkSelfVerificationNode(llm)
    bulk_final_result = BulkFinalResultNode(llm)

    # 构建图
    workflow = StateGraph(ClassificationState)

    # RAG节点
    if vector_store and embeddings:
        rag_retrieval = RAGRetrievalNode(llm, settings, vector_store, embeddings)
    else:
        # 创建虚拟RAG节点（不实际执行）
        from typing import Any, Dict

        class DummyRAGNode(BaseNode):
            def process(self, state: ClassificationState) -> Dict[str, Any]:
                return {"retrieved_examples": None}

        rag_retrieval = DummyRAGNode(llm)
    workflow.add_node("rag_retrieval_node", rag_retrieval.process)

    # 添加所有Bulk处理节点
    workflow.add_node("bulk_feature_analysis_node", bulk_feature_analysis.process)
    workflow.add_node(
        "bulk_preliminary_classification_node", bulk_preliminary_classification.process
    )
    workflow.add_node("bulk_self_verification_node", bulk_self_verification.process)
    workflow.add_node("bulk_final_result_node", bulk_final_result.process)

    # 添加通用节点
    workflow.add_node("context_analysis_node", context_analysis.process)
    workflow.add_node("evaluation_node", evaluation.process)

    # Bulk流程边：bulk_feature -> bulk_preliminary -> bulk_verification -> bulk_final
    workflow.add_edge(
        "bulk_feature_analysis_node", "bulk_preliminary_classification_node"
    )
    workflow.add_edge(
        "bulk_preliminary_classification_node", "bulk_self_verification_node"
    )
    # 验证后根据是否需要重新分类决定下一步
    workflow.add_conditional_edges(
        "bulk_self_verification_node",
        reclassification_router,
        {
            "bulk_preliminary_classification_node": "bulk_preliminary_classification_node",
            "bulk_final_result_node": "bulk_final_result_node",
        },
    )

    # After bulk final -> 根据ground truth决定评估或结束
    workflow.add_conditional_edges(
        "bulk_final_result_node",
        should_evaluate_router,
        {
            "evaluation_node": "evaluation_node",
            "end": END,
        },
    )

    if enable_table_context:
        # 启用表格上下文：从context analysis开始
        workflow.set_entry_point("context_analysis_node")

        # After context analysis -> 根据RAG是否启用决定下一步
        workflow.add_conditional_edges(
            "context_analysis_node",
            next_after_context_router,
            {
                "rag_retrieval_node": "rag_retrieval_node",
                "bulk_feature_analysis_node": "bulk_feature_analysis_node",
            },
        )

        # After RAG retrieval -> 进入bulk feature analysis
        workflow.add_conditional_edges(
            "rag_retrieval_node",
            after_rag_router,
            {
                "bulk_feature_analysis_node": "bulk_feature_analysis_node",
            },
        )
    else:
        # 不启用表格上下文：直接从bulk feature analysis开始
        workflow.set_entry_point("bulk_feature_analysis_node")

    # Evaluation always ends
    workflow.add_edge("evaluation_node", END)

    return workflow
