"""Vector store adapter interface for embedding-based search."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID


class VectorStore(ABC):
    """Interface for vector-based similarity search on memories."""

    @abstractmethod
    async def add(self, memory_id: UUID, embedding: list[float], metadata: dict[str, Any]) -> None:
        """Add an embedding to the store."""
        ...

    @abstractmethod
    async def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Search for the top-k most similar embeddings."""
        ...

    @abstractmethod
    async def delete(self, memory_id: UUID) -> None:
        """Remove an embedding from the store."""
        ...


class StubVectorStore(VectorStore):
    """In-memory stub vector store for iteration-001.

    Stores embeddings in a dict. Uses naive dot-product similarity.
    Suitable for testing and development only.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def add(self, memory_id: UUID, embedding: list[float], metadata: dict[str, Any]) -> None:
        self._store[str(memory_id)] = {
            "embedding": embedding,
            "metadata": metadata,
        }

    async def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        if not self._store:
            return []

        results = []
        for mid, entry in self._store.items():
            stored = entry["embedding"]
            # naive dot-product similarity
            sim = sum(a * b for a, b in zip(query_embedding, stored))
            results.append({
                "memory_id": mid,
                "score": sim,
                "metadata": entry["metadata"],
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    async def delete(self, memory_id: UUID) -> None:
        self._store.pop(str(memory_id), None)
