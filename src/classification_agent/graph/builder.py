from typing import Optional
from langgraph.graph import StateGraph, END

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.nodes import (
    ContextAnalysisNode,
    FeatureAnalysisNode,
    PreliminaryClassificationNode,
    CombinedFeatureAndClassificationNode,
    SelfVerificationNode,
    FinalResultNode,
    RAGRetrievalNode,
    EvaluationNode,
)
# Import bulk processing nodes
from classification_agent.nodes.bulk_feature_analysis import BulkFeatureAnalysisNode
from classification_agent.nodes.bulk_preliminary_classification import BulkPreliminaryClassificationNode
from classification_agent.nodes.bulk_self_verification import BulkSelfVerificationNode
from classification_agent.nodes.bulk_final_result import BulkFinalResultNode
from classification_agent.nodes.base_node import BaseNode
from classification_agent.config.settings import Settings
from classification_agent.rag.embeddings import BaseEmbeddings
from classification_agent.rag.vector_store import InMemoryVectorStore
from .edges import after_verification_router, should_retrieve_router


def should_evaluate_router(state: ClassificationState) -> str:
    """Router: 如果有 ground_truth，就走评估节点，否则直接结束"""
    # Check if bulk mode - after bulk final result, go directly to evaluate or end
    if state.get("bulk_mode"):
        if state.get("ground_truth_list") is not None:
            return "evaluation_node"
        else:
            return "end"

    # Original single/serial batch mode logic
    remaining = state.get("remaining_inputs")
    if remaining is not None and len(remaining) > 0:
        # Still fields left, go to next field
        return "next_batch_field_node"
    elif state.get("ground_truth_list") is not None or state.get("ground_truth_data_items") is not None:
        # All done, has ground truth -> evaluate
        return "evaluation_node"
    else:
        # All done, no ground truth -> end
        return "end"


# Note: settings is captured from the outer scope in build_graph
def create_next_after_context_router(settings: Settings):
    def next_after_context_router(state: ClassificationState) -> str:
        """After context analysis, decide next step based on bulk mode and RAG enabled"""
        # Check if we're in bulk table-level processing mode
        if state.get("bulk_mode"):
            # Bulk mode: after context, check if we need RAG retrieval
            if state.get("rag_enabled") and state.get("retrieved_examples") is None:
                return "rag_retrieval_node"
            else:
                # Bulk mode: directly go to bulk feature analysis (all fields in one go)
                return "bulk_feature_analysis_node"

        if state.get("inputs") is not None and len(state.get("inputs", [])) > 0:
            # Serial Batch mode: after context, check if we need RAG retrieval
            if state.get("rag_enabled") and state.get("retrieved_examples") is None:
                return "rag_retrieval_node"
            else:
                # Batch mode: after RAG (if any), ready to process first field (serial mode)
                # Note: Parallel mode is now handled at agent.classify_table level
                return "prepare_first_batch_node"
        else:
            # Single field mode: check if we need RAG retrieval
            if state.get("rag_enabled") and state.get("retrieved_examples") is None:
                return "rag_retrieval_node"
            else:
                if settings.fast_mode:
                    return "combined_feature_classification_node"
                else:
                    return "feature_analysis_node"
    return next_after_context_router


def create_after_rag_router(settings: Settings):
    def after_rag_router(state: ClassificationState) -> str:
        """After RAG retrieval, where to go next"""
        # Check if bulk mode
        if state.get("bulk_mode"):
            # Bulk mode: after RAG, directly to bulk feature analysis
            return "bulk_feature_analysis_node"

        if state.get("inputs") is not None and len(state.get("inputs", [])) > 0:
            # Batch mode: after RAG, prepare first field
            return "prepare_first_batch_node"
        else:
            # Single field mode: go to feature/classification
            if settings.fast_mode:
                return "combined_feature_classification_node"
            else:
                return "feature_analysis_node"
    return after_rag_router


def prepare_first_batch(state: ClassificationState) -> dict:
    """Prepare for first batch field processing: take first field from inputs"""
    inputs = state["inputs"]
    ground_truth_list = state.get("ground_truth_list")

    # Take first field
    first_field = inputs[0]
    if len(inputs) > 1:
        remaining_inputs = inputs[1:]
    else:
        remaining_inputs = []

    # Prepare ground truth if available
    remaining_gt = None
    current_gt = None
    if ground_truth_list and len(ground_truth_list) > 0:
        current_gt = ground_truth_list[0]
        if len(ground_truth_list) > 1:
            remaining_gt = ground_truth_list[1:]
        else:
            remaining_gt = []

    return {
        "input": first_field,
        "remaining_inputs": remaining_inputs,
        "remaining_ground_truth": remaining_gt,
        "ground_truth_data_items": current_gt,
        "completed_results": [],
    }


def next_batch_field(state: ClassificationState) -> dict:
    """Save completed result and take next field from remaining_inputs"""
    # Save current result to completed_results
    from classification_agent.types.schemas import ClassificationResult

    # Build classification result for current field
    known_feature_keys = {"table_name_keywords", "field_name_keywords", "description_keywords",
                          "semantic_summary", "consistency_analysis", "dominant_source"}
    known_preliminary_keys = {"predictions", "total_confidence"}
    known_verification_keys = {"verified_predictions", "removed_false_positives", "added_missing",
                               "average_confidence", "cross_validation_note", "suggests_reclassification"}

    feature_analysis = state["feature_analysis"]
    preliminary = state["preliminary_classification"]
    verification = state.get("verification") or {}

    classification_result: ClassificationResult = {
        "final_predictions": state.get("_final_predictions", []),
        "final_labels": state.get("_final_labels", []),
        "final_confidence": state.get("_final_avg_confidence", 0.0),
        "reasoning_chain": state.get("_reasoning_chain", []),
        "feature_analysis": {k: v for k, v in feature_analysis.items() if k in known_feature_keys},
        "preliminary_result": {k: v for k, v in preliminary.items() if k in known_preliminary_keys},
        "verification_result": {k: v for k, v in verification.items() if k in known_verification_keys},
        "evaluation": None,
        "table_context_analysis": state.get("table_context_analysis"),
    }

    # Add to completed results
    completed = state.get("completed_results", [])
    completed.append(classification_result)

    # Take next field
    remaining = state.get("remaining_inputs", [])
    remaining_gt = state.get("remaining_ground_truth")

    if len(remaining) > 0:
        next_field = remaining[0]
        new_remaining = remaining[1:]
        next_gt = None
        if remaining_gt and len(remaining_gt) > 0:
            next_gt = remaining_gt[0]
            new_remaining_gt = remaining_gt[1:] if len(remaining_gt) > 1 else []
        else:
            new_remaining_gt = []

        # Reset for next field
        return {
            "completed_results": completed,
            "input": next_field,
            "remaining_inputs": new_remaining,
            "ground_truth_data_items": next_gt,
            "remaining_ground_truth": new_remaining_gt,
            # Reset per-field state
            "feature_analysis": None,
            "preliminary_classification": None,
            "verification": None,
            "_final_predictions": None,
            "_final_labels": None,
            "_final_avg_confidence": None,
            "_reasoning_chain": None,
            "reclassification_count": 0,
            "hallucinated_data_items": [],
        }
    else:
        # No more fields, keep completed and go to evaluation
        return {
            "completed_results": completed,
        }


def build_graph(
    llm: BaseLLM,
    settings: Settings,
    vector_store: Optional[InMemoryVectorStore] = None,
    embeddings: Optional[BaseEmbeddings] = None,
    enable_table_context: bool = True,
) -> StateGraph:
    """构建LangGraph

    If settings.fast_mode is True: merge feature_analysis + preliminary_classification into one node,
    reduces network round-trips by ~33% for faster processing.

    If enable_table_context is True: add context_analysis node at the beginning
    to analyze whole table context before field-level classification.

    **Bulk Table-Level Processing (new mode - one LLM call per step for all fields):**
    1. context_analysis_node - analyze whole table context (once)
    2. rag_retrieval_node (optional) - retrieve similar examples based on table context
    3. bulk_feature_analysis_node - analyze ALL fields in ONE LLM call
    4. bulk_preliminary_classification_node - classify ALL fields in ONE LLM call
    5. bulk_self_verification_node - verify ALL fields in ONE LLM call
    6. bulk_final_result_node - aggregate all results
    7. evaluation_node (if ground truth provided) -> END

    **Total LLM calls for bulk mode: 3-4 calls total regardless of number of fields**

    **Serial Batch mode flow (multiple fields in one table, original):**
    1. context_analysis_node - analyze whole table context (once, shared by all fields)
    2. rag_retrieval_node (optional) - retrieve similar examples based on table context
    3. prepare_first_batch_node - extract first field to process
    4. feature_analysis_node - analyze current field features
    5. preliminary_classification_node - preliminary classification
    6. self_verification_node - self-verification to remove hallucinations/false positives
    7. final_result_node - aggregate final result for current field
    8. Check if more fields remaining:
       - If yes: next_batch_field_node -> feature_analysis_node (loop)
       - If no: evaluation_node (if ground truth provided) -> END

    **Single field flow:**
    Same as before: context_analysis -> (rag) -> feature -> preliminary -> verification -> final -> evaluation

    **Parallel batch flow (handled at agent level):**
    - context_analysis done once
    - all fields processed in parallel via thread pool
    - results collected and returned
    """

    # 创建节点
    context_analysis = ContextAnalysisNode(llm)
    self_verification = SelfVerificationNode(llm)
    final_result = FinalResultNode(llm)
    evaluation = EvaluationNode(llm)

    # 创建批量处理节点（表级一次性处理）
    bulk_feature_analysis = BulkFeatureAnalysisNode(llm)
    bulk_preliminary_classification = BulkPreliminaryClassificationNode(llm)
    bulk_self_verification = BulkSelfVerificationNode(llm)
    bulk_final_result = BulkFinalResultNode(llm)

    # 构建图
    workflow = StateGraph(ClassificationState)

    # Always add all nodes (required for LangGraph validation - conditional routing requires all targets exist)
    # Add RAG node
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
    workflow.add_node("rag_retrieval_node", rag_retrieval.process)

    # Always add all single mode nodes regardless of fast_mode for langgraph validation
    # Some routes reference them even if they aren't used
    feature_analysis = FeatureAnalysisNode(llm)
    preliminary_classification = PreliminaryClassificationNode(llm)
    combined = CombinedFeatureAndClassificationNode(llm)
    workflow.add_node("feature_analysis_node", feature_analysis.process)
    workflow.add_node("preliminary_classification_node", preliminary_classification.process)
    workflow.add_node("combined_feature_classification_node", combined.process)

    # Add bulk processing nodes (always add for validation)
    workflow.add_node("bulk_feature_analysis_node", bulk_feature_analysis.process)
    workflow.add_node("bulk_preliminary_classification_node", bulk_preliminary_classification.process)
    workflow.add_node("bulk_self_verification_node", bulk_self_verification.process)
    workflow.add_node("bulk_final_result_node", bulk_final_result.process)

    # Add common nodes
    workflow.add_node("context_analysis_node", context_analysis.process)
    workflow.add_node("self_verification_node", self_verification.process)
    workflow.add_node("final_result_node", final_result.process)
    workflow.add_node("evaluation_node", evaluation.process)

    # Add batch preparation node (serial mode - kept for backward compatibility)
    workflow.add_node("prepare_first_batch_node", prepare_first_batch)
    workflow.add_node("next_batch_field_node", next_batch_field)

    # Add bulk flow edges: bulk_feature -> bulk_preliminary -> bulk_verification -> bulk_final
    workflow.add_edge("bulk_feature_analysis_node", "bulk_preliminary_classification_node")
    workflow.add_edge("bulk_preliminary_classification_node", "bulk_self_verification_node")
    workflow.add_edge("bulk_self_verification_node", "bulk_final_result_node")
    # After bulk final -> go to evaluation or end via router
    workflow.add_conditional_edges(
        "bulk_final_result_node",
        should_evaluate_router,
        {
            "evaluation_node": "evaluation_node",
            "end": END,
        }
    )

    if enable_table_context:
        # With table context: start at context analysis
        workflow.set_entry_point("context_analysis_node")

        # After context analysis -> route based on bulk/single/serial and RAG
        next_after_context = create_next_after_context_router(settings)
        workflow.add_conditional_edges(
            "context_analysis_node",
            next_after_context,
            {
                "rag_retrieval_node": "rag_retrieval_node",
                "bulk_feature_analysis_node": "bulk_feature_analysis_node",
                "prepare_first_batch_node": "prepare_first_batch_node",
                "feature_analysis_node": "feature_analysis_node",
                "combined_feature_classification_node": "combined_feature_classification_node",
            }
        )

        # After RAG retrieval -> where to go next
        after_rag = create_after_rag_router(settings)
        workflow.add_conditional_edges(
            "rag_retrieval_node",
            after_rag,
            {
                "bulk_feature_analysis_node": "bulk_feature_analysis_node",
                "prepare_first_batch_node": "prepare_first_batch_node",
                "feature_analysis_node": "feature_analysis_node",
                "combined_feature_classification_node": "combined_feature_classification_node",
            }
        )
    else:
        # Without table context: not implemented for bulk mode yet
        # Entry directly goes to feature/combined for single mode
        if settings.fast_mode:
            workflow.set_entry_point("combined_feature_classification_node")
        else:
            workflow.set_entry_point("feature_analysis_node")

    # After prepare first batch -> go to feature/combined based on fast_mode (serial mode)
    if settings.fast_mode:
        workflow.add_edge("prepare_first_batch_node", "combined_feature_classification_node")
    else:
        workflow.add_edge("prepare_first_batch_node", "feature_analysis_node")

    # Single mode normal: feature -> preliminary -> self verification
    if not settings.fast_mode:
        workflow.add_edge("feature_analysis_node", "preliminary_classification_node")
        workflow.add_edge("preliminary_classification_node", "self_verification_node")

    # Single mode fast mode: combined -> self verification
    if settings.fast_mode:
        if enable_table_context:
            workflow.add_edge("combined_feature_classification_node", "self_verification_node")
        # For batch mode, edge added after prepare_first_batch above
        if not enable_table_context:
            pass  # handled by entry point

    # Self verification -> conditional to reclassify or final
    workflow.add_conditional_edges(
        "self_verification_node",
        after_verification_router,
        {
            "preliminary_classification_node": "combined_feature_classification_node" if settings.fast_mode else "preliminary_classification_node",
            "final_result_node": "final_result_node",
        }
    )

    # Final result (single mode or serial mode) -> check if more fields remaining, then evaluate or end
    workflow.add_conditional_edges(
        "final_result_node",
        should_evaluate_router,
        {
            "next_batch_field_node": "next_batch_field_node",
            "evaluation_node": "evaluation_node",
            "end": END,
        }
    )

    # Next field -> back to feature/combined based on fast_mode (serial mode)
    if settings.fast_mode:
        workflow.add_edge("next_batch_field_node", "combined_feature_classification_node")
    else:
        workflow.add_edge("next_batch_field_node", "feature_analysis_node")

    # Evaluation always ends
    workflow.add_edge("evaluation_node", END)

    return workflow
