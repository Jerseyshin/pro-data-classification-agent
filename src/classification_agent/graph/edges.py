from typing import Literal

from classification_agent.graph.state import ClassificationState


def after_verification_router(state: ClassificationState) -> Literal["final_result", "preliminary_classification"]:
    """验证后的路由逻辑
    如果置信度低于阈值且还没重分类过，回到preliminary_classification重分类一次
    否则去final_result
    """
    verification = state["verification"]
    reclassification_count = state.get("reclassification_count", 0)

    if (
        verification
        and verification.get("suggests_reclassification", False)
        and reclassification_count < 1
    ):
        # 允许重新分类一次
        return "preliminary_classification"

    return "final_result"


def should_retrieve_router(state: ClassificationState) -> Literal["rag_retrieval", "preliminary_classification"]:
    """Router to decide whether to run RAG retrieval or skip it"""
    if state.get("rag_enabled", False):
        return "rag_retrieval"
    return "preliminary_classification"
