"""Domain models for live documents, chunks, citations, memory records, and multi-hop research tasks."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    JIRA = "jira"
    NOTION = "notion"
    SLACK = "slack"
    MEETING_NOTES = "meeting_notes"
    API = "api"


def make_stable_doc_id(source_type: SourceType, external_id: str) -> str:
    """Generates a stable, canonical document identifier based on source and external key."""
    clean_ext = external_id.strip()
    return f"{source_type.value}:{clean_ext}"


class Document(BaseModel):
    """Normalized external document from a live source connector with stable ID tracking."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: SourceType
    title: str
    content: str
    url: str
    external_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Ensure stable canonical document ID if not explicitly provided as custom
        if "id" not in data and self.external_id:
            self.id = make_stable_doc_id(self.source_type, self.external_id)


class Chunk(BaseModel):
    """Searchable passage generated from a document with full provenance tracking and stable ID."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    source_type: SourceType
    source_title: str
    source_url: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: Optional[List[float]] = Field(default=None, description="Dense vector embedding representation")


class Citation(BaseModel):
    """Verified reference linking claims directly to authentic source documents."""

    source_id: str
    source_title: str
    source_url: str
    source_type: SourceType
    quote_snippet: str
    freshness_timestamp: datetime


class MemoryRecord(BaseModel):
    """Structured, durable knowledge unit preserved across agent conversations with attribute conflict resolution."""

    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    subject: str
    predicate: str = Field(default="general", description="Attribute or aspect key (e.g. 'owner', 'infrastructure')")
    fact: str
    source_id: str
    source_url: str
    source_type: SourceType = Field(default=SourceType.MEETING_NOTES)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0
    expires_at: Optional[datetime] = None
    is_active: bool = True
    superseded_by: Optional[str] = Field(default=None, description="ID of the newer record that superseded this fact")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchEvidence(BaseModel):
    """Intermediate evidence gathered for a specific subquestion during multi-hop reasoning."""

    subquestion: str
    retrieved_chunks: List[Chunk] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    findings: str = ""
    confidence: float = 1.0


class ResearchTask(BaseModel):
    """End-to-end multi-hop research request, execution plan, and synthesized answer."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    plan: List[str] = Field(default_factory=list)
    evidence: List[ResearchEvidence] = Field(default_factory=list)
    final_answer: str = ""
    citations: List[Citation] = Field(default_factory=list)
    confidence_score: float = 1.0
    hop_count: int = 1
    status: str = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
