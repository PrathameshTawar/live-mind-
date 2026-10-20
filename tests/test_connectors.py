"""Unit tests for live connectors: Base chunking, Jira issues, Notion pages, and incremental sync."""

import pytest
from livemind.connectors.jira import JiraConnector
from livemind.connectors.notion import NotionConnector
from livemind.core.models import Document, SourceType


def test_base_connector_chunking():
    jira = JiraConnector()
    doc = Document(
        source_type=SourceType.JIRA,
        title="Checkout Failure",
        content="A" * 700,
        url="https://jira.company.internal/browse/PAY-1",
        external_id="PAY-1",
    )
    chunks = jira.chunk_document(doc, max_chars=300, overlap=50)

    assert len(chunks) >= 3
    assert all(c.document_id == doc.id for c in chunks)
    assert all(c.source_url == doc.url for c in chunks)
    assert all(c.source_type == SourceType.JIRA for c in chunks)


def test_jira_sync_and_incremental_change_detection():
    jira = JiraConnector()
    jira.add_or_update_issue(
        ticket_key="PAY-1842",
        summary="Payment API connection pool exhaustion",
        description="Pool exhausted at peak load yesterday at 14:30.",
        status="In Progress",
    )

    # First sync returns the newly added issue
    first_sync = jira.sync()
    assert len(first_sync) == 1
    assert first_sync[0].metadata["ticket_key"] == "PAY-1842"
    assert first_sync[0].source_type == SourceType.JIRA

    # Second sync without changes returns 0 documents (incremental optimization)
    second_sync = jira.sync()
    assert len(second_sync) == 0

    # Modify ticket -> subsequent sync detects modification
    jira.add_or_update_issue(
        ticket_key="PAY-1842",
        summary="Payment API connection pool exhaustion",
        description="Pool exhausted at peak load yesterday. Patch deployed.",
        status="Resolved",
    )
    third_sync = jira.sync()
    assert len(third_sync) == 1
    assert "Patch deployed" in third_sync[0].content


def test_jira_authoritative_live_status_lookup():
    jira = JiraConnector()
    jira.add_or_update_issue(
        ticket_key="PAY-1842",
        summary="Fix pool limits",
        description="Proposed fix",
        status="In Progress",
    )

    status_data = jira.fetch_live_status("PAY-1842")
    assert status_data is not None
    assert status_data["status"] == "In Progress"
    assert status_data["is_resolved"] is False

    # Update status to Closed
    jira.add_or_update_issue(
        ticket_key="PAY-1842",
        summary="Fix pool limits",
        description="Proposed fix",
        status="Done",
    )
    updated_status = jira.fetch_live_status("PAY-1842")
    assert updated_status["is_resolved"] is True


def test_notion_sync_and_change_detection():
    notion = NotionConnector()
    notion.add_or_update_page(
        page_id="incident-oct-19",
        title="Incident Report: Checkout Outage",
        body="Root cause was database connection pool starvation.",
        page_type="incident_report",
        tags=["checkout", "database"],
    )

    docs = notion.sync()
    assert len(docs) == 1
    assert docs[0].title == "Incident Report: Checkout Outage"
    assert docs[0].source_type == SourceType.NOTION
    assert "checkout" in docs[0].metadata["tags"]

    # Incremental test: no changes -> empty sync
    assert len(notion.sync()) == 0
