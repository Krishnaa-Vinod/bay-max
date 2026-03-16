"""Text embedding adapter for memory retrieval."""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class TextEmbedder(ABC):
    """Interface for text embedding models."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning vectors."""
        ...

    @abstractmethod
    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class SentenceTransformerEmbedder(TextEmbedder):
    """Text embedder using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._dim: int | None = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Loaded SentenceTransformer '%s' (dim=%d)", self._model_name, self._dim
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        self._ensure_model()
        embedding = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dim


class StubTextEmbedder(TextEmbedder):
    """Stub embedder that returns random vectors. For testing only."""

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(texts), self._dim))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        return vecs.tolist()

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dim
