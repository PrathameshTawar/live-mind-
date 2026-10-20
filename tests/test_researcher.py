"""End-to-end integration test for MultiHopResearcher on cross-source reasoning with citations and live status verification."""

import pytest
from livemind.agent.researcher import MultiHopResearcher
from livemind.connectors.jira import JiraConnector
from livemind.connectors.notion import NotionConnector
from livemind.core.models import SourceType
from livemind.memory.service import MemoryService
from livemind.retrieval.hybrid import HybridRetriever


@pytest.fixture
def workspace_setup():
    # 1. Setup Jira
    jira = JiraConnector()
    jira.add_or_update_issue(
        ticket_key="PAY-1842",
        summary="Increase connection pool size and add backpressure",
        description="The checkout API suffered pool exhaustion. Proposed fix is to raise pool limit from 10 to 50.",
        status="In Progress",
        assignee="Bob",
    )

    # 2. Setup Notion Incident Report
    notion = NotionConnector()
    notion.add_or_update_page(
        page_id="incident-checkout-oct19",
        title="Incident Post-Mortem: Checkout API Outage",
        body="The checkout API experienced connection pool exhaustion at 14:30 yesterday. Ticket PAY-1842 created to resolve.",
        page_type="incident_report",
    )

    # 3. Setup Long-Term Memory (Past Meeting Decision)
    memory = MemoryService()
    memory.store_fact(
        subject="checkout-fix-agreement",
        fact="The team agreed to add connection-pool monitoring before deploying the PAY-1842 fix.",
        source_id="meeting-note-101",
        source_url="https://notion.so/meeting-note-101",
    )

    # 4. Ingest and index chunks into HybridRetriever
    jira_docs = jira.sync()
    notion_docs = notion.sync()
    all_chunks = []
    for doc in jira_docs:
        all_chunks.extend(jira.chunk_document(doc))
    for doc in notion_docs:
        all_chunks.extend(notion.chunk_document(doc))

    retriever = HybridRetriever()
    retriever.index_chunks(all_chunks)

    researcher = MultiHopResearcher(
        retriever=retriever,
        memory_service=memory,
        jira_connector=jira,
    )

    return researcher, memory


def test_end_to_end_multihop_research_and_citation(workspace_setup):
    researcher, memory = workspace_setup
    question = (
        "Why did our checkout API fail yesterday, what fix did we agree on, "
        "and is the related Jira issue resolved?"
    )

    task = researcher.conduct_research(question, session_id="dev_session_99")

    # 1. Verify task completed and plan created
    assert task.status == "completed"
    assert len(task.plan) >= 2

    # 2. Verify evidence from multiple hops gathered
    assert len(task.evidence) >= 2

    # 3. Verify citations link to authentic documents
    assert len(task.citations) > 0
    citation_titles = [c.source_title for c in task.citations]
    assert any("Incident Post-Mortem" in t for t in citation_titles)

    # 4. Verify authoritative live check identified ticket as still In Progress
    assert "In Progress" in task.final_answer
    assert "Still In Progress" in task.final_answer or "not deployed" in task.final_answer.lower()
    assert "PAY-1842" in task.final_answer

    # 5. Verify short-term conversation memory retained the turn
    session_turns = memory.get_short_term_context("dev_session_99")
    assert len(session_turns) == 2
    assert session_turns[0]["role"] == "user"
    assert session_turns[1]["role"] == "assistant"
