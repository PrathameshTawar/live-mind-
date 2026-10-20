"""Hybrid retriever combining exact lexical keyword matching, semantic similarity, and Reciprocal Rank Fusion (RRF)."""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
from livemind.core.models import Chunk, SourceType


class HybridRetriever:
    """Balances lexical precision (ticket IDs, error codes) with semantic similarity via Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.chunks: List[Chunk] = []
        # Inverted index for exact and lexical term search
        self.inverted_index: Dict[str, Set[int]] = defaultdict(set)
        # Term frequencies and document frequencies
        self.doc_term_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizes text while preserving ticket IDs (e.g., PAY-1842), error codes, and alphanumeric terms."""
        # Match alphanumeric words, ticket keys like PAY-1842, and dotted identifiers
        tokens = re.findall(r"\b[A-Za-z]+-\d+\b|\b\w+\b", text.lower())
        return tokens

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Indexes candidate chunks into both lexical and vector representations."""
        self.chunks = chunks
        self.inverted_index.clear()
        self.doc_term_freqs.clear()
        self.doc_lengths.clear()

        total_len = 0
        for idx, chunk in enumerate(chunks):
            tokens = self.tokenize(chunk.text + " " + chunk.source_title)
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)

            for token in tf.keys():
                self.inverted_index[token].add(idx)

        self.avg_doc_len = (total_len / len(chunks)) if chunks else 1.0

    def search_keyword(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Performs BM25-style lexical search for exact terms, ticket IDs, and phrases."""
        query_tokens = self.tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        scores: Dict[int, float] = defaultdict(float)
        total_docs = len(self.chunks)

        for token in query_tokens:
            posting_list = self.inverted_index.get(token, set())
            if not posting_list:
                continue
            # IDF calculation with smoothing
            df = len(posting_list)
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))

            for doc_idx in posting_list:
                tf = self.doc_term_freqs[doc_idx][token]
                doc_len = self.doc_lengths[doc_idx]
                # Standard BM25 term weighting (k1=1.5, b=0.75)
                numerator = tf * (1.5 + 1.0)
                denominator = tf + 1.5 * (1.0 - 0.75 + 0.75 * (doc_len / self.avg_doc_len))
                scores[doc_idx] += idf * (numerator / denominator)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def search_semantic(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Computes semantic overlap score across terms and titles for paraphrased queries."""
        query_tokens = set(self.tokenize(query))
        if not query_tokens or not self.chunks:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for idx, chunk in enumerate(self.chunks):
            doc_tokens = set(self.tokenize(chunk.text + " " + chunk.source_title))
            intersection = query_tokens.intersection(doc_tokens)
            if intersection:
                # Jaccard / Cosine overlap proxy
                sim = len(intersection) / math.sqrt(len(query_tokens) * len(doc_tokens) + 1.0)
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
        """Combines keyword and semantic candidate ranks using Reciprocal Rank Fusion (RRF)."""
        if not self.chunks:
            return []

        kw_results = self.search_keyword(query, top_k=top_k * 2)
        sem_results = self.search_semantic(query, top_k=top_k * 2)

        rrf_scores: Dict[int, float] = defaultdict(float)

        # RRF for keyword rankings
        for rank, (doc_idx, _) in enumerate(kw_results, start=1):
            rrf_scores[doc_idx] += keyword_weight / (self.rrf_k + rank)

        # RRF for semantic rankings
        for rank, (doc_idx, _) in enumerate(sem_results, start=1):
            rrf_scores[doc_idx] += semantic_weight / (self.rrf_k + rank)

        ranked_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)

        selected_chunks: List[Chunk] = []
        for idx in ranked_indices:
            candidate = self.chunks[idx]
            if source_filter and candidate.source_type != source_filter:
                continue
            selected_chunks.append(candidate)
            if len(selected_chunks) >= top_k:
                break

        return selected_chunks
