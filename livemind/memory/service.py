"""Dual-tier memory service managing ephemeral session context and persistent durable facts with provenance."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from livemind.core.models import MemoryRecord


class MemoryService:
    """Provides short-term conversational context and durable, provenance-backed long-term facts."""

    def __init__(self):
        # Short-term memory: session_id -> list of turn messages / findings
        self.short_term_store: Dict[str, List[Dict[str, str]]] = {}
        # Long-term memory: memory_id -> MemoryRecord
        self.long_term_store: Dict[str, MemoryRecord] = {}

    # --- Short-Term Memory Operations ---

    def add_short_term_turn(self, session_id: str, role: str, content: str) -> None:
        """Appends a conversation turn to active session memory."""
        if session_id not in self.short_term_store:
            self.short_term_store[session_id] = []
        self.short_term_store[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_short_term_context(self, session_id: str, last_n: int = 6) -> List[Dict[str, str]]:
        """Retrieves recent conversation turns for the active session."""
        return self.short_term_store.get(session_id, [])[-last_n:]

    def clear_short_term(self, session_id: str) -> None:
        """Clears short-term context for a completed session."""
        if session_id in self.short_term_store:
            del self.short_term_store[session_id]

    # --- Long-Term Memory Operations ---

    def store_fact(
        self,
        subject: str,
        fact: str,
        source_id: str,
        source_url: str,
        confidence: float = 1.0,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
    ) -> MemoryRecord:
        """Stores or updates a durable fact with explicit provenance and conflict resolution."""
        # Conflict resolution: if a record for the same subject with similar fact exists, update it
        clean_subj = subject.strip().lower()
        for existing in self.long_term_store.values():
            if existing.is_active and existing.subject.strip().lower() == clean_subj:
                # If existing is older and new fact updates the subject, supersede old record
                existing.is_active = False

        record = MemoryRecord(
            subject=subject,
            fact=fact,
            source_id=source_id,
            source_url=source_url,
            confidence=confidence,
            expires_at=expires_at,
            is_active=True,
            metadata=metadata or {},
        )
        self.long_term_store[record.memory_id] = record
        return record

    def search_memories(
        self,
        query: str,
        subject: Optional[str] = None,
        include_expired: bool = False,
    ) -> List[MemoryRecord]:
        """Searches active persistent memories by subject or query keyword."""
        now = datetime.now(timezone.utc)
        results: List[MemoryRecord] = []
        tokens = query.lower().split()

        for record in self.long_term_store.values():
            if not record.is_active:
                continue

            # Check expiration
            if record.expires_at and record.expires_at < now and not include_expired:
                continue

            # Subject match
            if subject and record.subject.lower() != subject.lower():
                continue

            # Query relevance match
            rec_text = f"{record.subject} {record.fact}".lower()
            if any(t in rec_text for t in tokens) or not query:
                results.append(record)

        return results

    def delete_memory(self, memory_id: str) -> bool:
        """Deletes or deactivates a specific memory record."""
        if memory_id in self.long_term_store:
            self.long_term_store[memory_id].is_active = False
            return True
        return False
