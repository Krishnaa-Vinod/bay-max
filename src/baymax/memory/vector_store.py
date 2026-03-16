"""Vector store adapter interface for embedding-based search."""

import logging
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Interface for vector-based similarity search on memories."""

    @abstractmethod
    async def add(self, memory_id: UUID, embedding: list[float], metadata: dict[str, Any]) -> None:
        """Add an embedding to the store."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_user_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Search for the top-k most similar embeddings."""
        ...

    @abstractmethod
    async def delete(self, memory_id: UUID) -> None:
        """Remove an embedding from the store."""
        ...


class StubVectorStore(VectorStore):
    """In-memory stub vector store for testing.

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
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_user_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        if not self._store:
            return []

        results = []
        for mid, entry in self._store.items():
            if filter_user_id and entry["metadata"].get("user_id") != str(filter_user_id):
                continue
            stored = entry["embedding"]
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


class LanceDBVectorStore(VectorStore):
    """LanceDB-backed vector store for production memory retrieval."""

    def __init__(self, db_path: str = "./lancedb_data", table_name: str = "memories") -> None:
        self._db_path = db_path
        self._table_name = table_name
        self._db = None
        self._table = None

    def _ensure_db(self):
        if self._db is None:
            import lancedb

            self._db = lancedb.connect(self._db_path)
            try:
                self._table = self._db.open_table(self._table_name)
                logger.info("Opened existing LanceDB table '%s'", self._table_name)
            except Exception:
                self._table = None
                logger.info("LanceDB table '%s' does not exist yet", self._table_name)

    def _create_table_if_needed(self, dim: int) -> None:
        if self._table is not None:
            return
        import pyarrow as pa

        schema = pa.schema([
            pa.field("memory_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])
        self._table = self._db.create_table(self._table_name, schema=schema)
        logger.info("Created LanceDB table '%s' with dim=%d", self._table_name, dim)

    async def add(self, memory_id: UUID, embedding: list[float], metadata: dict[str, Any]) -> None:
        self._ensure_db()
        self._create_table_if_needed(len(embedding))
        row = {
            "memory_id": str(memory_id),
            "user_id": metadata.get("user_id", ""),
            "memory_type": metadata.get("memory_type", ""),
            "content": metadata.get("content", ""),
            "vector": embedding,
        }
        self._table.add([row])

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_user_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_db()
        if self._table is None:
            return []

        query = self._table.search(query_embedding).limit(top_k)
        if filter_user_id:
            query = query.where(f"user_id = '{filter_user_id}'")

        try:
            rows = query.to_list()
        except Exception:
            logger.warning("LanceDB search failed", exc_info=True)
            return []

        results = []
        for row in rows:
            results.append({
                "memory_id": row["memory_id"],
                "score": 1.0 - row.get("_distance", 0.0),
                "metadata": {
                    "user_id": row.get("user_id", ""),
                    "memory_type": row.get("memory_type", ""),
                    "content": row.get("content", ""),
                },
            })
        return results

    async def delete(self, memory_id: UUID) -> None:
        self._ensure_db()
        if self._table is None:
            return
        try:
            self._table.delete(f"memory_id = '{memory_id}'")
        except Exception:
            logger.warning("LanceDB delete failed for %s", memory_id, exc_info=True)
