"""FastAPI REST server exposing LiveMind multi-hop research, live connectors, and dual-tier memory."""

from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from livemind import __version__
from livemind.agent.researcher import MultiHopResearcher
from livemind.api.schemas import (
    CitationResponse,
    JiraSyncRequest,
    MemoryFactRequest,
    MemoryFactResponse,
    NotionSyncRequest,
    ResearchRequest,
    ResearchResponse,
    SyncSummaryResponse,
)
from livemind.connectors.jira import JiraConnector
from livemind.connectors.notion import NotionConnector
from livemind.memory.service import MemoryService
from livemind.retrieval.hybrid import HybridRetriever

app = FastAPI(
    title="LiveMind API",
    version=__version__,
    description="Agentic RAG over Live Workspace Data with Dual-Tier Persistent Memory",
)

# Static files for Apple HIG frontend
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", tags=["UI"], include_in_schema=False)
def serve_ui():
    """Serves Apple HIG frontend interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "LiveMind API active. Static UI not built."}

# Shared singleton services
jira_connector = JiraConnector()
notion_connector = NotionConnector()
memory_service = MemoryService()
retriever = HybridRetriever()
researcher = MultiHopResearcher(
    retriever=retriever,
    memory_service=memory_service,
    jira_connector=jira_connector,
)

# Global chunk registry
ALL_CHUNKS = []


def refresh_retriever_index():
    """Helper to re-index all chunks across connectors into the hybrid retriever."""
    global ALL_CHUNKS
    retriever.index_chunks(ALL_CHUNKS)


@app.get("/health", tags=["System"])
def health_check():
    """Returns service operational health and version."""
    return {
        "status": "healthy",
        "service": "LiveMind",
        "version": __version__,
        "indexed_chunks": len(ALL_CHUNKS),
    }


@app.post("/research", response_model=ResearchResponse, tags=["Research"])
def research_question(payload: ResearchRequest):
    """Executes multi-hop reasoning across indexed live workspace data and authoritative APIs."""
    task = researcher.conduct_research(payload.question, session_id=payload.session_id)
    return ResearchResponse(
        task_id=task.task_id,
        question=task.question,
        plan=task.plan,
        final_answer=task.final_answer,
        citations=[
            CitationResponse(
                source_id=c.source_id,
                source_title=c.source_title,
                source_url=c.source_url,
                source_type=c.source_type.value,
                quote_snippet=c.quote_snippet,
                freshness_timestamp=c.freshness_timestamp,
            )
            for c in task.citations
        ],
        status=task.status,
        created_at=task.created_at,
    )


@app.post("/connectors/jira/sync", response_model=SyncSummaryResponse, tags=["Connectors"])
def sync_jira_issue(payload: JiraSyncRequest):
    """Ingests or updates a Jira issue and indexes its chunks incrementally."""
    jira_connector.add_or_update_issue(
        ticket_key=payload.ticket_key,
        summary=payload.summary,
        description=payload.description,
        status=payload.status,
        priority=payload.priority,
        assignee=payload.assignee,
        comments=payload.comments,
    )
    new_docs = jira_connector.sync()
    for doc in new_docs:
        chunks = jira_connector.chunk_document(doc)
        ALL_CHUNKS.extend(chunks)

    refresh_retriever_index()
    return SyncSummaryResponse(
        connector="jira",
        synced_documents_count=len(new_docs),
        total_indexed_chunks=len(ALL_CHUNKS),
    )


@app.post("/connectors/notion/sync", response_model=SyncSummaryResponse, tags=["Connectors"])
def sync_notion_page(payload: NotionSyncRequest):
    """Ingests or updates a Notion document and indexes its chunks incrementally."""
    notion_connector.add_or_update_page(
        page_id=payload.page_id,
        title=payload.title,
        body=payload.body,
        page_type=payload.page_type,
        tags=payload.tags,
        author=payload.author,
    )
    new_docs = notion_connector.sync()
    for doc in new_docs:
        chunks = notion_connector.chunk_document(doc)
        ALL_CHUNKS.extend(chunks)

    refresh_retriever_index()
    return SyncSummaryResponse(
        connector="notion",
        synced_documents_count=len(new_docs),
        total_indexed_chunks=len(ALL_CHUNKS),
    )


@app.post("/memory", response_model=MemoryFactResponse, tags=["Memory"])
def store_memory_fact(payload: MemoryFactRequest):
    """Stores a structured persistent knowledge fact with source provenance and conflict resolution."""
    record = memory_service.store_fact(
        subject=payload.subject,
        fact=payload.fact,
        source_id=payload.source_id,
        source_url=payload.source_url,
        confidence=payload.confidence,
    )
    return MemoryFactResponse(
        memory_id=record.memory_id,
        subject=record.subject,
        fact=record.fact,
        source_id=record.source_id,
        source_url=record.source_url,
        confidence=record.confidence,
        observed_at=record.observed_at,
        is_active=record.is_active,
    )


@app.get("/memory", response_model=List[MemoryFactResponse], tags=["Memory"])
def list_active_memories(query: Optional[str] = "", subject: Optional[str] = None):
    """Retrieves active persistent memory facts matching optional query or subject."""
    records = memory_service.search_memories(query=query or "", subject=subject)
    return [
        MemoryFactResponse(
            memory_id=r.memory_id,
            subject=r.subject,
            fact=r.fact,
            source_id=r.source_id,
            source_url=r.source_url,
            confidence=r.confidence,
            observed_at=r.observed_at,
            is_active=r.is_active,
        )
        for r in records
    ]


@app.delete("/memory/{memory_id}", tags=["Memory"])
def delete_memory_record(memory_id: str):
    """Deactivates a specific memory record."""
    success = memory_service.delete_memory(memory_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory record with id {memory_id} not found",
        )
    return {"status": "deactivated", "memory_id": memory_id}
