from typing import Any, Dict
import numpy as np

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.rag.embeddings import BaseEmbeddings
from classification_agent.rag.vector_store import InMemoryVectorStore
from classification_agent.config.settings import Settings
from .base_node import BaseNode


class RAGRetrievalNode(BaseNode):
    """RAG retrieval node - finds similar labeled examples based on semantic similarity"""

    def __init__(
        self,
        llm: BaseLLM,
        settings: Settings,
        vector_store: InMemoryVectorStore,
        embeddings: BaseEmbeddings,
    ):
        super().__init__(llm)
        self.settings = settings
        self.vector_store = vector_store
        self.embeddings = embeddings

    def _prepare_query_text(self, input: Dict[str, Any], feature_analysis: Dict[str, Any]) -> str:
        """Prepare combined text for embedding from input and feature analysis"""
        parts = [
            f"Table: {input.get('table_name', '')}",
            f"Field: {input.get('field_name', '')}",
        ]
        if input.get('field_description'):
            parts.append(f"Description: {input['field_description']}")
        if feature_analysis:
            if feature_analysis.get('semantic_summary'):
                parts.append(f"Summary: {feature_analysis['semantic_summary']}")
            table_keywords = ','.join(feature_analysis.get('table_name_keywords', []))
            field_keywords = ','.join(feature_analysis.get('field_name_keywords', []))
            desc_keywords = ','.join(feature_analysis.get('description_keywords', []))
            if table_keywords:
                parts.append(f"Table keywords: {table_keywords}")
            if field_keywords:
                parts.append(f"Field keywords: {field_keywords}")
            if desc_keywords:
                parts.append(f"Description keywords: {desc_keywords}")
        return ' '.join(parts)

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        # Only execute if RAG is enabled for this classification
        if not state.get("rag_enabled", False):
            self.logger.info("RAG disabled, skipping retrieval")
            return {"retrieved_examples": None}

        # Check if we have any examples in the store
        if self.vector_store.size == 0:
            self.logger.info("RAG enabled but no examples in store, skipping")
            return {"retrieved_examples": None}

        # Determine input for query: handle both single field and bulk table mode
        if state.get("inputs") and len(state["inputs"]) > 0:
            # Bulk table mode: use the first field as representative for table-level query
            # This is because RAG retrieval happens before feature analysis in bulk mode,
            # so we can only use basic table information
            input_for_query = state["inputs"][0]
            self.logger.info(
                "Bulk table mode: using first field for RAG query - table: %s, field: %s",
                input_for_query.get("table_name"),
                input_for_query.get("field_name")
            )
        else:
            # Single field mode
            input_for_query = state["input"]
            self.logger.info(
                "Single field mode: using input for RAG query - table: %s, field: %s",
                input_for_query.get("table_name"),
                input_for_query.get("field_name")
            )

        # Prepare query text from input and feature analysis
        # Note: In bulk mode, feature_analysis may not be available yet
        query_text = self._prepare_query_text(
            input_for_query,
            state.get("feature_analysis", {})
        )

        # Get embedding
        query_embedding = np.array(self.embeddings.embed_text(query_text))

        # Search for similar examples
        top_k = self.settings.rag_top_k
        threshold = self.settings.rag_similarity_threshold
        results = self.vector_store.search(query_embedding, top_k, threshold)

        self.logger.info(
            "RAG retrieved %d similar examples (top_k=%d, threshold=%.2f)",
            len(results), top_k, threshold
        )

        return {"retrieved_examples": results}
