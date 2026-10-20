"""Pydantic schemas for LiveMind research tasks, connector synchronization, and memory endpoints."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Payload to submit a multi-hop research question across workspace data."""

    question: str = Field(..., description="Complex, multi-hop question to research")
    session_id: Optional[str] = Field(default=None, description="Optional conversational session ID for short-term memory")


class CitationResponse(BaseModel):
    source_id: str
    source_title: str
    source_url: str
    source_type: str
    quote_snippet: str
    freshness_timestamp: datetime


class ResearchResponse(BaseModel):
    task_id: str
    question: str
    plan: List[str]
    final_answer: str
    citations: List[CitationResponse]
    status: str
    created_at: datetime


class JiraSyncRequest(BaseModel):
    """Payload to synchronize or ingest a Jira issue."""

    ticket_key: str = Field(..., description="Jira issue key, e.g. PAY-1842")
    summary: str
    description: str
    status: str = "In Progress"
    priority: str = "Medium"
    assignee: Optional[str] = None
    comments: Optional[List[str]] = None


class NotionSyncRequest(BaseModel):
    """Payload to synchronize or ingest a Notion document/page."""

    page_id: str = Field(..., description="Unique Notion page slug/id")
    title: str
    body: str
    page_type: str = "documentation"
    tags: Optional[List[str]] = None
    author: str = "Engineering Team"


class MemoryFactRequest(BaseModel):
    """Payload to record a persistent long-term knowledge fact."""

    subject: str = Field(..., description="Subject or entity of the fact")
    fact: str = Field(..., description="Durable knowledge assertion")
    source_id: str
    source_url: str
    confidence: float = 1.0


class MemoryFactResponse(BaseModel):
    memory_id: str
    subject: str
    fact: str
    source_id: str
    source_url: str
    confidence: float
    observed_at: datetime
    is_active: bool


class SyncSummaryResponse(BaseModel):
    connector: str
    synced_documents_count: int
    total_indexed_chunks: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
