from typing import List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from classification_agent.types.schemas import TableFieldInput, HierarchicalCategory, RetrievedExample


class InMemoryVectorStore:
    """In-memory vector store for similarity search using cosine similarity"""

    def __init__(self):
        self._embeddings: Optional[np.ndarray] = None
        self._examples: List[Tuple[TableFieldInput, HierarchicalCategory]] = []

    @property
    def size(self) -> int:
        """Number of examples in the store"""
        return len(self._examples)

    def get_all_examples(self) -> List[Tuple[TableFieldInput, HierarchicalCategory]]:
        """Get all stored examples"""
        return self._examples.copy()

    def add_example(
        self,
        embedding: np.ndarray,
        input: TableFieldInput,
        label: HierarchicalCategory,
    ) -> None:
        """Add a single labeled example to the store"""
        if self._embeddings is None:
            self._embeddings = embedding.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])
        self._examples.append((input, label))

    def add_examples(
        self,
        embeddings: List[np.ndarray],
        examples: List[Tuple[TableFieldInput, HierarchicalCategory]],
    ) -> None:
        """Add multiple labeled examples in batch"""
        embeddings_np = np.stack(embeddings)
        if self._embeddings is None:
            self._embeddings = embeddings_np
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings_np])
        self._examples.extend(examples)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> List[RetrievedExample]:
        """Search for similar examples by cosine similarity"""
        if self._embeddings is None or len(self._examples) == 0:
            return []

        # Reshape query for sklearn
        query_reshaped = query_embedding.reshape(1, -1)

        # Compute similarities
        similarities = cosine_similarity(query_reshaped, self._embeddings)[0]

        # Filter by threshold and get top_k
        filtered_indices = [
            i for i, sim in enumerate(similarities)
            if sim >= similarity_threshold
        ]

        # Sort by descending similarity
        filtered_indices.sort(key=lambda i: similarities[i], reverse=True)

        # Take top_k
        top_indices = filtered_indices[:top_k]

        # Build result
        results: List[RetrievedExample] = []
        for idx in top_indices:
            input_example, label = self._examples[idx]
            results.append({
                "input": input_example,
                "label": label,
                "similarity_score": float(similarities[idx]),
            })

        return results

    def clear(self) -> None:
        """Clear all examples from the store"""
        self._embeddings = None
        self._examples = []

    @property
    def size(self) -> int:
        """Return number of examples in store"""
        return len(self._examples)
