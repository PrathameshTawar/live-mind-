"""Unit tests for MemoryService: short-term session state, long-term facts, provenance, and conflict resolution."""

from datetime import datetime, timedelta, timezone
import pytest
from livemind.memory.service import MemoryService


def test_short_term_memory_session_scoping():
    memory = MemoryService()
    memory.add_short_term_turn("session_1", "user", "Why is checkout failing?")
    memory.add_short_term_turn("session_1", "assistant", "Checking incident reports...")

    turns_s1 = memory.get_short_term_context("session_1")
    assert len(turns_s1) == 2
    assert turns_s1[0]["role"] == "user"
    assert turns_s1[1]["role"] == "assistant"

    # Separate session remains isolated
    turns_s2 = memory.get_short_term_context("session_2")
    assert len(turns_s2) == 0


def test_long_term_fact_storage_and_search():
    memory = MemoryService()
    record = memory.store_fact(
        subject="checkout-api",
        fact="The team agreed to add connection-pool monitoring before deploying the fix.",
        source_id="meeting-note-123",
        source_url="https://notion.so/meeting-note-123",
        confidence=0.96,
    )

    assert record.is_active is True
    assert record.source_id == "meeting-note-123"

    results = memory.search_memories(query="monitoring connection-pool")
    assert len(results) == 1
    assert results[0].subject == "checkout-api"
    assert "monitoring" in results[0].fact


def test_conflict_resolution_supersedes_outdated_fact():
    memory = MemoryService()
    # Initial decision
    rec1 = memory.store_fact(
        subject="checkout-pool-size",
        fact="Agreed to increase pool size to 30.",
        source_id="meeting-1",
        source_url="https://notion.so/meeting-1",
    )
    # New decision updates pool size
    rec2 = memory.store_fact(
        subject="checkout-pool-size",
        fact="Updated decision: increase pool size to 50 instead of 30.",
        source_id="meeting-2",
        source_url="https://notion.so/meeting-2",
    )

    # First record should now be inactive
    assert memory.long_term_store[rec1.memory_id].is_active is False
    assert memory.long_term_store[rec2.memory_id].is_active is True

    # Search should return only the active, updated decision
    active_facts = memory.search_memories(query="pool size", subject="checkout-pool-size")
    assert len(active_facts) == 1
    assert "50 instead of 30" in active_facts[0].fact


def test_expired_memory_ignored_by_default():
    memory = MemoryService()
    past_date = datetime.now(timezone.utc) - timedelta(days=1)
    memory.store_fact(
        subject="temporary-token",
        fact="Temporary staging token: xyz-999",
        source_id="slack-1",
        source_url="https://slack.com/archive",
        expires_at=past_date,
    )

    active = memory.search_memories(query="temporary staging token")
    assert len(active) == 0

    # With include_expired=True, it can be audited
    expired = memory.search_memories(query="temporary staging token", include_expired=True)
    assert len(expired) == 1


def test_delete_memory():
    memory = MemoryService()
    rec = memory.store_fact(
        subject="deprecated-api",
        fact="v1 endpoint deprecated",
        source_id="doc-1",
        source_url="https://docs.internal/v1",
    )
    deleted = memory.delete_memory(rec.memory_id)
    assert deleted is True
    assert len(memory.search_memories(query="v1 endpoint")) == 0
