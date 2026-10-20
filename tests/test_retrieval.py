"""Unit tests for HybridRetriever: BM25 keyword search, semantic similarity, and Reciprocal Rank Fusion."""

import pytest
from livemind.core.models import Chunk, SourceType
from livemind.retrieval.hybrid import HybridRetriever


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            document_id="doc-1",
            text="General payment gateway documentation and API keys for Stripe integration.",
            source_type=SourceType.NOTION,
            source_title="Payment Gateway Guide",
            source_url="https://notion.so/payment-guide",
        ),
        Chunk(
            document_id="doc-2",
            text="Issue PAY-1842: Fix connection pool exhaustion by increasing max_connections to 50.",
            source_type=SourceType.JIRA,
            source_title="[PAY-1842] Connection Pool Bug",
            source_url="https://jira.company.internal/browse/PAY-1842",
        ),
        Chunk(
            document_id="doc-3",
            text="Meeting notes Oct 15: Team agreed to add Prometheus metrics before deploying payment fix.",
            source_type=SourceType.MEETING_NOTES,
            source_title="Architecture Meeting Oct 15",
            source_url="https://notion.so/meeting-oct-15",
        ),
        Chunk(
            document_id="doc-4",
            text="Checkout API incident postmortem: database connection starvation caused 500 errors yesterday at 14:30.",
            source_type=SourceType.NOTION,
            source_title="Incident Postmortem: Checkout Outage",
            source_url="https://notion.so/incident-checkout",
        ),
    ]


def test_tokenizer_preserves_ticket_ids():
    retriever = HybridRetriever()
    tokens = retriever.tokenize("Check ticket PAY-1842 and incident CHECKOUT-99 immediately.")
    assert "pay-1842" in tokens
    assert "checkout-99" in tokens
    assert "immediately" in tokens


def test_keyword_search_matches_exact_ticket_id(sample_chunks):
    retriever = HybridRetriever()
    retriever.index_chunks(sample_chunks)

    results = retriever.search_keyword("PAY-1842")
    assert len(results) > 0
    top_doc_idx = results[0][0]
    assert sample_chunks[top_doc_idx].source_title == "[PAY-1842] Connection Pool Bug"


def test_semantic_search_finds_paraphrased_concepts(sample_chunks):
    retriever = HybridRetriever()
    retriever.index_chunks(sample_chunks)

    results = retriever.search_semantic("database connection starvation outage")
    assert len(results) > 0
    top_doc_idx = results[0][0]
    assert sample_chunks[top_doc_idx].source_title == "Incident Postmortem: Checkout Outage"


def test_hybrid_rrf_fuses_results_and_filters_source(sample_chunks):
    retriever = HybridRetriever()
    retriever.index_chunks(sample_chunks)

    # Search with RRF
    fused_results = retriever.search("checkout connection pool PAY-1842", top_k=3)
    assert len(fused_results) <= 3
    assert any("PAY-1842" in c.text for c in fused_results)

    # Filter strictly by JIRA
    jira_only = retriever.search("connection", top_k=5, source_filter=SourceType.JIRA)
    assert all(c.source_type == SourceType.JIRA for c in jira_only)
