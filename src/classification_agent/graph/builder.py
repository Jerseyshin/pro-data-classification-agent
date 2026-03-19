from typing import Optional
from langgraph.graph import StateGraph, END

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.nodes import (
    FeatureAnalysisNode,
    PreliminaryClassificationNode,
    SelfVerificationNode,
    FinalResultNode,
    RAGRetrievalNode,
    EvaluationNode,
)
from classification_agent.nodes.base_node import BaseNode
from classification_agent.config.settings import Settings
from classification_agent.rag.embeddings import BaseEmbeddings
from classification_agent.rag.vector_store import InMemoryVectorStore
from .edges import after_verification_router, should_retrieve_router


def should_evaluate_router(state: ClassificationState) -> str:
    """Router: 如果有 ground_truth，就走评估节点，否则直接结束"""
    if state.get("ground_truth_data_items") is not None:
        return "evaluation"
    else:
        return "end"


def build_graph(
    llm: BaseLLM,
    settings: Settings,
    vector_store: Optional[InMemoryVectorStore] = None,
    embeddings: Optional[BaseEmbeddings] = None,
) -> StateGraph:
    """构建LangGraph"""

    # 创建节点
    feature_analysis = FeatureAnalysisNode(llm)
    preliminary_classification = PreliminaryClassificationNode(llm)
    self_verification = SelfVerificationNode(llm)
    final_result = FinalResultNode(llm)
    evaluation = EvaluationNode(llm)

    # 构建图
    workflow = StateGraph(ClassificationState)

    # 添加基础节点
    workflow.add_node("feature_analysis", feature_analysis.process)
    workflow.add_node("preliminary_classification", preliminary_classification.process)
    workflow.add_node("self_verification", self_verification.process)
    workflow.add_node("final_result", final_result.process)
    workflow.add_node("evaluation", evaluation.process)

    # Always add RAG node (required for conditional edge validation in LangGraph)
    # When RAG is disabled, router will never select this path, so node is never executed
    if vector_store and embeddings:
        rag_retrieval = RAGRetrievalNode(llm, settings, vector_store, embeddings)
    else:
        # Create a dummy node that just returns None (never executed anyway)
        # We need the node to exist for LangGraph validation
        from typing import Any, Dict
        class DummyRAGNode(BaseNode):
            def process(self, state: ClassificationState) -> Dict[str, Any]:
                return {"retrieved_examples": None}
        rag_retrieval = DummyRAGNode(llm)
    workflow.add_node("rag_retrieval", rag_retrieval.process)

    # 设置入口点
    workflow.set_entry_point("feature_analysis")

    # 条件边：根据是否启用RAG决定走检索还是直接到初步分类
    workflow.add_conditional_edges(
        "feature_analysis",
        should_retrieve_router,
        {
            "rag_retrieval": "rag_retrieval",
            "preliminary_classification": "preliminary_classification",
        }
    )

    # RAG检索完去初步分类
    workflow.add_edge("rag_retrieval", "preliminary_classification")

    # 后续边保持不变
    workflow.add_edge("preliminary_classification", "self_verification")
    workflow.add_conditional_edges(
        "self_verification",
        after_verification_router,
        {
            "preliminary_classification": "preliminary_classification",
            "final_result": "final_result",
        }
    )

    # 条件边：如果提供了 ground truth，走评估；否则直接结束
    workflow.add_conditional_edges(
        "final_result",
        should_evaluate_router,
        {
            "evaluation": "evaluation",
            "end": END,
        }
    )
    # 评估完结束
    workflow.add_edge("evaluation", END)

    return workflow
