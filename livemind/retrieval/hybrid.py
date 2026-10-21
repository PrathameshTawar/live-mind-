"""Hybrid retriever combining lexical BM25 matching, dense semantic embeddings, and Reciprocal Rank Fusion (RRF)."""

import math
import re
import threading
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
from livemind.core.config import settings
from livemind.core.models import Chunk, SourceType
from livemind.retrieval.embeddings import EmbeddingService, cosine_similarity


class HybridRetriever:
    """Balances lexical precision (ticket IDs, error codes) with dense semantic embeddings via Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: Optional[int] = None):
        self.rrf_k = rrf_k or settings.HYBRID_RRF_K
        self.embedding_service = EmbeddingService()
        self._lock = threading.RLock()

        # Document-to-chunks mapping for incremental upsert and deletion
        self.doc_to_chunk_ids: Dict[str, Set[str]] = defaultdict(set)
        self.chunk_registry: Dict[str, Chunk] = {}

        # Search index structures
        self.chunks: List[Chunk] = []
        self.inverted_index: Dict[str, Set[int]] = defaultdict(set)
        self.doc_term_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 1.0
        self.doc_embeddings: List[List[float]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizes text while preserving ticket IDs (e.g., PAY-1842), error codes, and alphanumeric terms."""
        return re.findall(r"\b[A-Za-z]+-\d+\b|\b\w+\b", text.lower())

    def _rebuild_indices_unlocked(self) -> None:
        """Rebuilds lexical inverted index and vector store across all registered chunks."""
        self.chunks = list(self.chunk_registry.values())
        self.inverted_index.clear()
        self.doc_term_freqs.clear()
        self.doc_lengths.clear()
        self.doc_embeddings.clear()

        if not self.chunks:
            self.avg_doc_len = 1.0
            return

        total_len = 0
        texts_needing_embedding = []
        indices_needing_embedding = []

        for idx, chunk in enumerate(self.chunks):
            tokens = self.tokenize(chunk.text + " " + chunk.source_title)
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)

            for token in tf.keys():
                self.inverted_index[token].add(idx)

            if chunk.embedding is not None:
                self.doc_embeddings.append(chunk.embedding)
            else:
                self.doc_embeddings.append([])
                texts_needing_embedding.append(f"{chunk.source_title}: {chunk.text}")
                indices_needing_embedding.append(idx)

        # Batch embed any chunks that do not yet have dense embeddings
        if texts_needing_embedding:
            computed_vectors = self.embedding_service.embed_texts(texts_needing_embedding)
            for idx, vec in zip(indices_needing_embedding, computed_vectors):
                self.chunks[idx].embedding = vec
                self.doc_embeddings[idx] = vec

        self.avg_doc_len = (total_len / len(self.chunks)) if self.chunks else 1.0

    def upsert_document_chunks(self, document_id: str, new_chunks: List[Chunk]) -> None:
        """Atomically updates chunks for a document: purges old chunks and indexes new ones."""
        with self._lock:
            # 1. Purge previous chunks for this document_id
            old_chunk_ids = self.doc_to_chunk_ids.get(document_id, set())
            for cid in old_chunk_ids:
                self.chunk_registry.pop(cid, None)
            self.doc_to_chunk_ids[document_id].clear()

            # 2. Add new chunks
            for chunk in new_chunks:
                self.chunk_registry[chunk.id] = chunk
                self.doc_to_chunk_ids[document_id].add(chunk.id)

            # 3. Rebuild search structures
            self._rebuild_indices_unlocked()

    def delete_document(self, document_id: str) -> bool:
        """Purges all indexed chunks for a document that has been removed at the source."""
        with self._lock:
            if document_id not in self.doc_to_chunk_ids:
                return False
            old_chunk_ids = self.doc_to_chunk_ids.pop(document_id, set())
            for cid in old_chunk_ids:
                self.chunk_registry.pop(cid, None)
            self._rebuild_indices_unlocked()
            return True

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Full re-indexing of a chunk list (replaces current index)."""
        with self._lock:
            self.chunk_registry.clear()
            self.doc_to_chunk_ids.clear()
            for chunk in chunks:
                self.chunk_registry[chunk.id] = chunk
                self.doc_to_chunk_ids[chunk.document_id].add(chunk.id)
            self._rebuild_indices_unlocked()

    def search_keyword(
        self,
        query: str,
        top_k: int = 20,
        candidate_indices: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Performs BM25 lexical search for exact terms, ticket IDs, and phrases."""
        query_tokens = self.tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        scores: Dict[int, float] = defaultdict(float)
        total_docs = len(self.chunks)

        for token in query_tokens:
            posting_list = self.inverted_index.get(token, set())
            if not posting_list:
                continue

            # Filter candidates if restricted
            active_postings = (
                posting_list.intersection(candidate_indices)
                if candidate_indices is not None
                else posting_list
            )
            if not active_postings:
                continue

            # IDF calculation with smoothing
            df = len(posting_list)
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))

            for doc_idx in active_postings:
                tf = self.doc_term_freqs[doc_idx][token]
                doc_len = self.doc_lengths[doc_idx]
                # Standard BM25 term weighting (k1=1.5, b=0.75)
                numerator = tf * (1.5 + 1.0)
                denominator = tf + 1.5 * (1.0 - 0.75 + 0.75 * (doc_len / self.avg_doc_len))
                scores[doc_idx] += idf * (numerator / denominator)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def search_semantic(
        self,
        query: str,
        top_k: int = 20,
        candidate_indices: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Computes dense cosine similarity between query embedding and chunk vectors."""
        if not self.chunks:
            return []

        query_vec = self.embedding_service.embed_query(query)
        scores: Dict[int, float] = defaultdict(float)

        target_indices = (
            range(len(self.chunks))
            if candidate_indices is None
            else candidate_indices
        )

        for idx in target_indices:
            doc_vec = self.doc_embeddings[idx]
            if doc_vec:
                sim = cosine_similarity(query_vec, doc_vec)
                if sim > 0.0:
                    scores[idx] = sim

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[SourceType] = None,
        keyword_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ) -> List[Chunk]:
        """Combines keyword and dense semantic candidate ranks using Reciprocal Rank Fusion (RRF)."""
        with self._lock:
            if not self.chunks:
                return []

            # Pre-filter candidate indices by source_filter so filtered searches don't lose slots
            candidate_indices: Optional[Set[int]] = None
            if source_filter:
                candidate_indices = {
                    idx for idx, c in enumerate(self.chunks) if c.source_type == source_filter
                }
                if not candidate_indices:
                    return []

            pool_k = max(top_k * 4, 20)
            kw_results = self.search_keyword(query, top_k=pool_k, candidate_indices=candidate_indices)
            sem_results = self.search_semantic(query, top_k=pool_k, candidate_indices=candidate_indices)

            rrf_scores: Dict[int, float] = defaultdict(float)

            # RRF for keyword rankings
            for rank, (doc_idx, _) in enumerate(kw_results, start=1):
                rrf_scores[doc_idx] += keyword_weight / (self.rrf_k + rank)

            # RRF for semantic rankings
            for rank, (doc_idx, _) in enumerate(sem_results, start=1):
                rrf_scores[doc_idx] += semantic_weight / (self.rrf_k + rank)

            ranked_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)

            selected_chunks: List[Chunk] = []
            seen_chunk_ids: Set[str] = set()

            for idx in ranked_indices:
                candidate = self.chunks[idx]
                if candidate.id not in seen_chunk_ids:
                    selected_chunks.append(candidate)
                    seen_chunk_ids.add(candidate.id)
                if len(selected_chunks) >= top_k:
                    break

            return selected_chunks
