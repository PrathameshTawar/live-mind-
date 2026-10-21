"""Base live data connector supporting incremental synchronization and sentence-aware chunking."""

import hashlib
import re
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
        """Splits document content into natural sentence/paragraph passages preserving word boundaries and metadata."""
        text = document.content.strip()
        if not text:
            return []

        if len(text) <= max_chars:
            return [
                Chunk(
                    id=f"{document.id}:chunk:0",
                    document_id=document.id,
                    text=text,
                    source_type=document.source_type,
                    source_title=document.title,
                    source_url=document.url,
                    metadata=document.metadata,
                    timestamp=document.updated_at,
                )
            ]

        # Break into atomic units (sentences or lines or words)
        raw_units = re.split(r"(?<=\n)\s*(?=\n)|(?<=[.!?])\s+(?=[A-Z0-9])", text)
        units: List[str] = []
        for u in raw_units:
            u_clean = u.strip()
            if not u_clean:
                continue
            if len(u_clean) > max_chars:
                # Split large unit by spaces or words
                words = u_clean.split()
                if len(words) > 1:
                    cur_w: List[str] = []
                    cur_wl = 0
                    for w in words:
                        if cur_wl + len(w) + 1 > max_chars and cur_w:
                            units.append(" ".join(cur_w))
                            cur_w = [w]
                            cur_wl = len(w)
                        else:
                            cur_w.append(w)
                            cur_wl += len(w) + 1
                    if cur_w:
                        units.append(" ".join(cur_w))
                else:
                    # Single unspaced monolithic string
                    step = max(1, max_chars - overlap)
                    for start in range(0, len(u_clean), step):
                        slice_end = min(start + max_chars, len(u_clean))
                        units.append(u_clean[start:slice_end])
            else:
                units.append(u_clean)

        chunks: List[Chunk] = []
        current_chunk: List[str] = []
        current_length = 0
        chunk_idx = 0

        i = 0
        while i < len(units):
            unit = units[i]
            if not current_chunk:
                current_chunk.append(unit)
                current_length = len(unit)
                i += 1
            elif current_length + 1 + len(unit) <= max_chars:
                current_chunk.append(unit)
                current_length += 1 + len(unit)
                i += 1
            else:
                passage = " ".join(current_chunk).strip()
                chunks.append(
                    Chunk(
                        id=f"{document.id}:chunk:{chunk_idx}",
                        document_id=document.id,
                        text=passage,
                        source_type=document.source_type,
                        source_title=document.title,
                        source_url=document.url,
                        metadata=document.metadata,
                        timestamp=document.updated_at,
                    )
                )
                chunk_idx += 1
                overlap_chars = 0
                overlap_units: List[str] = []
                for prev_u in reversed(current_chunk):
                    if overlap_chars + len(prev_u) <= overlap and len(overlap_units) < len(current_chunk):
                        overlap_units.insert(0, prev_u)
                        overlap_chars += len(prev_u) + 1
                    else:
                        break
                current_chunk = overlap_units
                current_length = sum(len(u) for u in current_chunk) + max(0, len(current_chunk) - 1)

        if current_chunk:
            passage = " ".join(current_chunk).strip()
            chunks.append(
                Chunk(
                    id=f"{document.id}:chunk:{chunk_idx}",
                    document_id=document.id,
                    text=passage,
                    source_type=document.source_type,
                    source_title=document.title,
                    source_url=document.url,
                    metadata=document.metadata,
                    timestamp=document.updated_at,
                )
            )

        return chunks

    @abstractmethod
    def sync(self) -> List[Document]:
        """Synchronizes and returns current normalized documents from the live service."""
        pass
