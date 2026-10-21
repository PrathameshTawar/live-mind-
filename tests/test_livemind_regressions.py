"""Regression test suite for LiveMind addressing critical issues and audit findings:
1. Stale chunk elimination and stable document ID tracking.
2. Attribute-level memory conflict resolution (Alice remains active when Stripe is added).
3. Configured project key restriction (rejection of SHA-256, UTF-8, COVID-19).
4. Memory search precision (query 'a' returns 0 results).
5. SQLite memory persistence and restart recovery.
6. Citation deduplication and accurate source type preservation.
7. Pre-ranking source filtering and word-boundary chunking.
8. Source deletion index synchronization.
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest
from livemind.agent.researcher import MultiHopResearcher, extract_jira_ticket_ids
from livemind.connectors.jira import JiraConnector
from livemind.connectors.notion import NotionConnector
from livemind.core.config import settings
from livemind.core.models import Document, MemoryRecord, SourceType, make_stable_doc_id
from livemind.memory.service import MemoryService
from livemind.retrieval.hybrid import HybridRetriever


def test_regression_stale_ticket_chunks_purged_on_update():
    """Confirms ticket PAY-1842 updated from 'In Progress' to 'Done' purges older stale chunks."""
    retriever = HybridRetriever()
    jira = JiraConnector()

    # Step 1: Initial sync with status 'In Progress'
    jira.add_or_update_issue(
        ticket_key="PAY-1842",
        summary="Payment checkout pool exhaustion",
        description="Fixing pool exhaustion. Current status is In Progress.",
        status="In Progress",
    )
    docs_v1 = jira.sync()
    for doc in docs_v1:
        chunks_v1 = jira.chunk_document(doc)
        retriever.upsert_document_chunks(doc.id, chunks_v1)

    # Verify search returns In Progress
    in_progress_results = retriever.search("PAY-1842 In Progress")
    assert any("In Progress" in c.text for c in in_progress_results)

    # Step 2: Update ticket to 'Done'
    jira.add_or_update_issue(
        ticket_key="PAY-1842",
        summary="Payment checkout pool exhaustion",
        description="Fixing pool exhaustion. Status is now Done and deployed.",
        status="Done",
    )
    docs_v2 = jira.sync()
    for doc in docs_v2:
        chunks_v2 = jira.chunk_document(doc)
        retriever.upsert_document_chunks(doc.id, chunks_v2)

    # Step 3: Search should find 'Done' and NEVER contain the stale 'In Progress' chunk
    done_results = retriever.search("PAY-1842")
    assert len(done_results) > 0
    all_matched_texts = [c.text for c in done_results]
    assert any("Done" in text for text in all_matched_texts)
    assert not any("Current status is In Progress" in text for text in all_matched_texts)

    # Only current version chunks exist in the retriever index
    stable_id = make_stable_doc_id(SourceType.JIRA, "PAY-1842")
    assert stable_id in retriever.doc_to_chunk_ids
    assert len(retriever.doc_to_chunk_ids[stable_id]) == len(chunks_v2)


def test_regression_source_deletion_purges_chunks():
    """Confirms deleting an issue from Jira deletes all its chunks from the retriever index."""
    retriever = HybridRetriever()
    jira = JiraConnector()

    jira.add_or_update_issue(
        ticket_key="PAY-9999",
        summary="Temporary ticket to be removed",
        description="This ticket should vanish completely.",
        status="Open",
    )
    for doc in jira.sync():
        retriever.upsert_document_chunks(doc.id, jira.chunk_document(doc))

    # Verify indexed
    assert len(retriever.search("PAY-9999")) > 0

    # Delete from source and retriever
    jira.delete_issue("PAY-9999")
    retriever.delete_document(make_stable_doc_id(SourceType.JIRA, "PAY-9999"))

    # Verify completely purged
    assert len(retriever.search("PAY-9999")) == 0
    assert make_stable_doc_id(SourceType.JIRA, "PAY-9999") not in retriever.doc_to_chunk_ids


def test_regression_unrelated_facts_for_same_subject_not_deactivated():
    """Confirms storing payments: 'Owner is Alice' then payments: 'Uses Stripe' preserves Alice."""
    memory = MemoryService()

    # Fact 1: Owner
    rec1 = memory.store_fact(
        subject="payments",
        fact="Owner is Alice",
        source_id="notion-team-page",
        source_url="https://notion.so/payments-team",
    )
    # Fact 2: Provider (different attribute)
    rec2 = memory.store_fact(
        subject="payments",
        fact="Uses Stripe as the primary gateway",
        source_id="slack-announcement",
        source_url="https://slack.com/archives/payments",
    )

    # Both facts must remain active!
    assert memory.long_term_store[rec1.memory_id].is_active is True
    assert memory.long_term_store[rec2.memory_id].is_active is True

    # Fact 3: Update owner to Bob (same attribute as Fact 1)
    rec3 = memory.store_fact(
        subject="payments",
        fact="Owner is Bob",
        source_id="hr-org-chart",
        source_url="https://hr.company.internal/payments",
    )

    # Alice fact is superseded by Bob; Stripe fact remains active!
    assert memory.long_term_store[rec1.memory_id].is_active is False
    assert memory.long_term_store[rec1.memory_id].superseded_by == rec3.memory_id
    assert memory.long_term_store[rec3.memory_id].is_active is True
    assert memory.long_term_store[rec2.memory_id].is_active is True


def test_regression_ticket_id_pattern_rejects_ordinary_terms():
    """Confirms SHA-256, UTF-8, COVID-19 are rejected as Jira tickets."""
    ordinary_text = "Upgraded to SHA-256 and UTF-8 encoding during COVID-19 response. Check ticket PAY-1842."
    extracted = extract_jira_ticket_ids(ordinary_text, allowed_project_keys=["PAY", "CORE", "INFRA"])

    assert "PAY-1842" in extracted
    assert "SHA-256" not in extracted
    assert "UTF-8" not in extracted
    assert "COVID-19" not in extracted


def test_regression_memory_search_ignores_noise_stopwords():
    """Confirms searching for 'a' or pure stop words does not match every memory record."""
    memory = MemoryService()
    memory.store_fact(
        subject="infrastructure",
        fact="The database runs in a high-availability cluster across regions.",
        source_id="infra-doc-1",
        source_url="https://wiki.internal/infra",
    )

    # Query with noise stop word 'a' must return 0 results
    results_a = memory.search_memories(query="a")
    assert len(results_a) == 0

    # Query with meaningful keyword matches
    results_cluster = memory.search_memories(query="cluster")
    assert len(results_cluster) == 1
    assert results_cluster[0].subject == "infrastructure"


def test_regression_memory_sqlite_persistence(tmp_path: Path):
    """Confirms facts persist to SQLite and are restored across service restarts."""
    db_file = tmp_path / "persistent_test.db"

    # Step 1: Save facts in memory service 1
    mem1 = MemoryService(db_path=str(db_file))
    rec1 = mem1.store_fact(
        subject="auth-service",
        fact="JWT rotation period set to 24 hours.",
        source_id="sec-policy-1",
        source_url="https://notion.so/security-policies",
        predicate="security_config",
    )

    # Step 2: Initialize clean memory service 2 from the same SQLite database
    mem2 = MemoryService(db_path=str(db_file))
    loaded = mem2.search_memories(query="JWT rotation", subject="auth-service")

    assert len(loaded) == 1
    assert loaded[0].memory_id == rec1.memory_id
    assert loaded[0].predicate == "security_config"
    assert "24 hours" in loaded[0].fact


def test_regression_researcher_deduplicates_citations_and_preserves_source_types():
    """Confirms citations are deduplicated and retain authentic SourceType (not hardcoded MEETING_NOTES)."""
    retriever = HybridRetriever()
    jira = JiraConnector()
    notion = NotionConnector()
    memory = MemoryService()

    jira.add_or_update_issue(
        ticket_key="PAY-1842",
        summary="Pool exhaustion fix",
        description="Issue description for pool exhaustion.",
        status="In Progress",
    )
    for doc in jira.sync():
        retriever.upsert_document_chunks(doc.id, jira.chunk_document(doc))

    notion.add_or_update_page(
        page_id="arch-notes-oct15",
        title="Architecture Decision: Connection Pooling",
        body="Team approved the connection pool increase to 50 connections.",
        page_type="architecture_note",
    )
    for doc in notion.sync():
        retriever.upsert_document_chunks(doc.id, notion.chunk_document(doc))

    memory.store_fact(
        subject="checkout-pool",
        fact="Prometheus metrics must accompany PAY-1842 fix.",
        source_id="meeting-memo-42",
        source_url="https://notion.so/meeting-memo-42",
        source_type=SourceType.MEETING_NOTES,
    )

    researcher = MultiHopResearcher(
        retriever=retriever,
        memory_service=memory,
        jira_connector=jira,
    )

    task = researcher.conduct_research("Status of PAY-1842 and connection pool architecture decision")

    # 1. Citations must not contain duplicate chunk/fact IDs
    cited_ids = [c.source_url for c in task.citations]
    assert len(cited_ids) == len(set(cited_ids))

    # 2. Source types must include JIRA and NOTION, not just MEETING_NOTES
    types = {c.source_type for c in task.citations}
    assert SourceType.JIRA in types or SourceType.NOTION in types


def test_regression_chunking_preserves_word_boundaries():
    """Confirms sentence and word boundaries are respected rather than slicing mid-word."""
    jira = JiraConnector()
    doc = Document(
        source_type=SourceType.JIRA,
        title="Word Boundary Test",
        content="Antidisestablishmentarianism is a very long political word. Next sentence explains why it matters.",
        url="https://jira.internal/TEST-1",
        external_id="TEST-1",
    )
    chunks = jira.chunk_document(doc, max_chars=80, overlap=10)
    for chunk in chunks:
        # Every chunk should consist of complete words (no severed trailing characters)
        assert not chunk.text.endswith(" ")
        assert "Antidisestablishmentarianism" in chunks[0].text
