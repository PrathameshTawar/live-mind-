"""Base live data connector supporting incremental synchronization and provenance-aware chunking."""

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional
from livemind.core.models import Chunk, Document


class BaseConnector(ABC):
    """Abstract base class for live data connectors with incremental sync and change tracking."""

    def __init__(self, name: str):
        self.name = name
        self.document_hashes: Dict[str, str] = {}
        self.last_sync_time: Optional[datetime] = None

    @staticmethod
    def compute_hash(content: str) -> str:
        """Computes SHA-256 fingerprint of normalized document content."""
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

    def is_updated(self, doc_id: str, content: str) -> bool:
        """Returns True if the document has been modified since the last sync."""
        new_hash = self.compute_hash(content)
        return self.document_hashes.get(doc_id) != new_hash

    def chunk_document(self, document: Document, max_chars: int = 400, overlap: int = 60) -> List[Chunk]:
        """Splits document content into overlapping passages preserving full citation metadata."""
        text = document.content.strip()
        if not text:
            return []

        chunks: List[Chunk] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + max_chars, text_len)
            passage = text[start:end].strip()

            if passage:
                chunk = Chunk(
                    document_id=document.id,
                    text=passage,
                    source_type=document.source_type,
                    source_title=document.title,
                    source_url=document.url,
                    metadata=document.metadata,
                    timestamp=document.updated_at,
                )
                chunks.append(chunk)

            if end >= text_len:
                break
            start += max_chars - overlap

        return chunks

    @abstractmethod
    def sync(self) -> List[Document]:
        """Synchronizes and returns current normalized documents from the live service."""
        pass
