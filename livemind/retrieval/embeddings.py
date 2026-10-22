"""Embedding service supporting sentence-transformers with graceful offline dense fallback."""

import math
import re
import zlib
from typing import List, Optional
import numpy as np
from livemind.core.config import settings


def _stable_hash(text: str) -> int:
    """Process-independent hash; built-in hash() is salted per interpreter run."""
    return zlib.crc32(text.encode("utf-8"))


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class EmbeddingService:
    """Provides dense vector representations for queries and documents."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._dim = 384
        self._load_model()

    def _load_model(self) -> None:
        """Attempts to load sentence-transformers model without blocking on network download."""
        try:
            from sentence_transformers import SentenceTransformer
            # Only use if local cache or offline files exist
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
            if hasattr(self._model, "get_embedding_dimension"):
                self._dim = self._model.get_embedding_dimension()
            else:
                self._dim = self._model.get_sentence_embedding_dimension()
        except Exception:
            self._model = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Computes normalized dense embeddings for a list of passages."""
        if not texts:
            return []

        if self._model is not None:
            try:
                embeddings = self._model.encode(texts, normalize_embeddings=True)
                return [emb.tolist() for emb in embeddings]
            except Exception:
                pass

        return [self._dense_hash_embedding(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Computes dense embedding vector for a search query."""
        results = self.embed_texts([query])
        return results[0] if results else [0.0] * self._dim

    def _dense_hash_embedding(self, text: str) -> List[float]:
        """Fast 384-dim semantic feature vector based on subword n-grams and hashing."""
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = re.findall(r"\b[A-Za-z]+-\d+\b|\b\w+\b", text.lower())
        for token in tokens:
            h = _stable_hash(token) % self._dim
            vec[h] += 1.0
            # Also hash character 3-grams for morphological / synonym capture
            for i in range(max(0, len(token) - 2)):
                ngram = token[i : i + 3]
                nh = (_stable_hash(ngram) * 31) % self._dim
                vec[nh] += 0.5

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.tolist()
