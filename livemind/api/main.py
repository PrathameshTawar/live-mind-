"""FastAPI REST server exposing LiveMind multi-hop research, live connectors, and dual-tier memory."""

from pathlib import Path
from typing import List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Security, status
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
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
from livemind.core.config import settings
from livemind.core.models import SourceType, make_stable_doc_id
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
memory_service = MemoryService(db_path=settings.MEMORY_STORAGE_PATH)
retriever = HybridRetriever()
researcher = MultiHopResearcher(
    retriever=retriever,
    memory_service=memory_service,
    jira_connector=jira_connector,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    x_api_key: Optional[str] = Security(api_key_header),
    authorization: Optional[str] = Header(None),
) -> bool:
    """Validates API key if REQUIRE_API_KEY is enabled."""
    if not settings.REQUIRE_API_KEY:
        return True

    key = x_api_key
    if not key and authorization:
        if authorization.startswith("Bearer "):
            key = authorization.split(" ", 1)[1]
        else:
            key = authorization

    if key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return True


@app.get("/health", tags=["System"])
def health_check():
    """Returns service operational health, version, and indexed chunks count."""
    return {
        "status": "healthy",
        "service": "LiveMind",
        "version": __version__,
        "indexed_chunks": len(retriever.chunks),
        "active_memories": sum(1 for m in memory_service.long_term_store.values() if m.is_active),
    }


@app.post("/research", response_model=ResearchResponse, tags=["Research"])
def research_question(
    payload: ResearchRequest,
    _: bool = Depends(verify_api_key),
):
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
        confidence_score=task.confidence_score,
        hop_count=task.hop_count,
        status=task.status,
        created_at=task.created_at,
    )


@app.post("/connectors/jira/sync", response_model=SyncSummaryResponse, tags=["Connectors"])
def sync_jira_issue(
    payload: JiraSyncRequest,
    _: bool = Depends(verify_api_key),
):
    """Ingests or updates a Jira issue and atomically updates its indexed chunks (purging old versions)."""
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
        # Atomically replaces previous chunks of this document to prevent stale knowledge
        retriever.upsert_document_chunks(doc.id, chunks)

    return SyncSummaryResponse(
        connector="jira",
        synced_documents_count=len(new_docs),
        total_indexed_chunks=len(retriever.chunks),
    )


@app.delete("/connectors/jira/{ticket_key}", tags=["Connectors"])
def delete_jira_issue(
    ticket_key: str,
    _: bool = Depends(verify_api_key),
):
    """Removes a Jira issue from the connector and purges all its chunks from the search index."""
    deleted = jira_connector.delete_issue(ticket_key)
    doc_id = make_stable_doc_id(SourceType.JIRA, ticket_key)
    retriever.delete_document(doc_id)
    return {"status": "deleted", "ticket_key": ticket_key, "found_in_source": deleted}


@app.post("/connectors/notion/sync", response_model=SyncSummaryResponse, tags=["Connectors"])
def sync_notion_page(
    payload: NotionSyncRequest,
    _: bool = Depends(verify_api_key),
):
    """Ingests or updates a Notion document and atomically updates its indexed chunks."""
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
        retriever.upsert_document_chunks(doc.id, chunks)

    return SyncSummaryResponse(
        connector="notion",
        synced_documents_count=len(new_docs),
        total_indexed_chunks=len(retriever.chunks),
    )


@app.delete("/connectors/notion/{page_id}", tags=["Connectors"])
def delete_notion_page(
    page_id: str,
    _: bool = Depends(verify_api_key),
):
    """Removes a Notion page from the connector and purges all its chunks from the search index."""
    deleted = notion_connector.delete_page(page_id)
    doc_id = make_stable_doc_id(SourceType.NOTION, page_id)
    retriever.delete_document(doc_id)
    return {"status": "deleted", "page_id": page_id, "found_in_source": deleted}


@app.post("/memory", response_model=MemoryFactResponse, tags=["Memory"])
def store_memory_fact(
    payload: MemoryFactRequest,
    _: bool = Depends(verify_api_key),
):
    """Stores a structured persistent knowledge fact with source provenance and attribute-level conflict resolution."""
    source_type_enum = (
        SourceType(payload.source_type)
        if payload.source_type in SourceType._value2member_map_
        else SourceType.MEETING_NOTES
    )
    record = memory_service.store_fact(
        subject=payload.subject,
        fact=payload.fact,
        source_id=payload.source_id,
        source_url=payload.source_url,
        predicate=payload.predicate,
        source_type=source_type_enum,
        confidence=payload.confidence,
    )
    return MemoryFactResponse(
        memory_id=record.memory_id,
        subject=record.subject,
        predicate=record.predicate,
        fact=record.fact,
        source_id=record.source_id,
        source_url=record.source_url,
        source_type=record.source_type.value,
        confidence=record.confidence,
        observed_at=record.observed_at,
        is_active=record.is_active,
    )


@app.get("/memory", response_model=List[MemoryFactResponse], tags=["Memory"])
def list_active_memories(
    query: Optional[str] = "",
    subject: Optional[str] = None,
    _: bool = Depends(verify_api_key),
):
    """Retrieves active persistent memory facts matching optional query or subject."""
    records = memory_service.search_memories(query=query or "", subject=subject)
    return [
        MemoryFactResponse(
            memory_id=r.memory_id,
            subject=r.subject,
            predicate=r.predicate,
            fact=r.fact,
            source_id=r.source_id,
            source_url=r.source_url,
            source_type=r.source_type.value,
            confidence=r.confidence,
            observed_at=r.observed_at,
            is_active=r.is_active,
        )
        for r in records
    ]


@app.delete("/memory/{memory_id}", tags=["Memory"])
def delete_memory_record(
    memory_id: str,
    _: bool = Depends(verify_api_key),
):
    """Deactivates a specific memory record in memory and SQLite."""
    success = memory_service.delete_memory(memory_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory record with id {memory_id} not found",
        )
    return {"status": "deactivated", "memory_id": memory_id}
