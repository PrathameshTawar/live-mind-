"""Dual-tier memory service managing ephemeral session context and persistent SQLite facts with attribute-level conflict resolution."""

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from livemind.core.config import settings
from livemind.core.models import MemoryRecord, SourceType

# Stop words ignored during keyword memory relevance checking
STOP_WORDS: Set[str] = {
    "a", "an", "the", "in", "on", "of", "to", "for", "is", "are", "was",
    "were", "at", "by", "with", "from", "and", "or", "what", "which",
    "who", "how", "where", "why", "can", "could", "should", "does",
}


def extract_predicate(fact_text: str, subject: str = "") -> str:
    """Infers the attribute/predicate being described if not explicitly declared."""
    lower = f"{subject} {fact_text}".lower()

    # Attribute concept categories
    ATTRIBUTE_PATTERNS = [
        ("owner", ["owner", "lead", "maintainer", "assignee", "author", "responsible"]),
        ("provider", ["provider", "vendor", "gateway", "stripe", "paypal", "adyen", "braintree"]),
        ("database", ["database", "db", "postgres", "mysql", "mongo", "redis", "sqlite"]),
        ("infrastructure", ["cloud", "aws", "gcp", "azure", "hosting", "k8s", "kubernetes", "docker"]),
        ("pool_size", ["pool size", "pool_size", "pool", "max_connections", "pool-size"]),
        ("size_capacity", ["limit", "capacity", "quota", "timeout", "size"]),
        ("status", ["status", "state", "phase", "condition"]),
        ("version", ["version", "release", "v1", "v2"]),
        ("architecture", ["architecture", "framework", "language", "stack"]),
        ("endpoint", ["url", "endpoint", "port", "host", "api"]),
    ]

    for attr_name, keywords in ATTRIBUTE_PATTERNS:
        if any(kw in lower for kw in keywords):
            return attr_name

    # Significant content words (ignoring stopwords)
    words = [w for w in re.findall(r"\b\w+\b", fact_text.lower()) if w not in STOP_WORDS]
    return words[0] if words else "general"


class MemoryService:
    """Provides short-term conversational context and durable, SQLite-backed long-term facts with attribute-level conflict resolution."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._lock = threading.RLock()

        # Short-term memory: session_id -> list of turn messages / findings
        self.short_term_store: Dict[str, List[Dict[str, str]]] = {}
        # Long-term memory: memory_id -> MemoryRecord
        self.long_term_store: Dict[str, MemoryRecord] = {}

        if self.db_path:
            self._init_sqlite_db()
            self._load_from_sqlite()

    def _init_sqlite_db(self) -> None:
        """Initializes SQLite schema for persistent episodic facts."""
        if not self.db_path:
            return
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        memory_id TEXT PRIMARY KEY,
                        subject TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        fact TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        source_url TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        observed_at TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        expires_at TEXT,
                        is_active INTEGER NOT NULL,
                        superseded_by TEXT,
                        metadata_json TEXT
                    )
                    """
                )
                conn.commit()

    def _load_from_sqlite(self) -> None:
        """Loads active memories into in-memory store from SQLite on startup."""
        if not self.db_path:
            return
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT memory_id, subject, predicate, fact, source_id, source_url, source_type, observed_at, confidence, expires_at, is_active, superseded_by, metadata_json FROM memories"
                    )
                    for row in cursor.fetchall():
                        rec = MemoryRecord(
                            memory_id=row[0],
                            subject=row[1],
                            predicate=row[2],
                            fact=row[3],
                            source_id=row[4],
                            source_url=row[5],
                            source_type=SourceType(row[6]) if row[6] in SourceType._value2member_map_ else SourceType.MEETING_NOTES,
                            observed_at=datetime.fromisoformat(row[7]),
                            confidence=row[8],
                            expires_at=datetime.fromisoformat(row[9]) if row[9] else None,
                            is_active=bool(row[10]),
                            superseded_by=row[11],
                            metadata=json.loads(row[12]) if row[12] else {},
                        )
                        self.long_term_store[rec.memory_id] = rec
            except Exception:
                pass

    def _persist_record(self, record: MemoryRecord) -> None:
        """Saves or updates a MemoryRecord in SQLite if db_path is configured."""
        if not self.db_path:
            return
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO memories (
                            memory_id, subject, predicate, fact, source_id, source_url,
                            source_type, observed_at, confidence, expires_at, is_active,
                            superseded_by, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.memory_id,
                            record.subject,
                            record.predicate,
                            record.fact,
                            record.source_id,
                            record.source_url,
                            record.source_type.value,
                            record.observed_at.isoformat(),
                            record.confidence,
                            record.expires_at.isoformat() if record.expires_at else None,
                            1 if record.is_active else 0,
                            record.superseded_by,
                            json.dumps(record.metadata),
                        ),
                    )
                    conn.commit()
            except Exception:
                pass

    # --- Short-Term Memory Operations ---

    def add_short_term_turn(self, session_id: str, role: str, content: str) -> None:
        """Appends a conversation turn to active session memory."""
        with self._lock:
            if session_id not in self.short_term_store:
                self.short_term_store[session_id] = []
            self.short_term_store[session_id].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def get_short_term_context(self, session_id: str, last_n: int = 6) -> List[Dict[str, str]]:
        """Retrieves recent conversation turns for the active session."""
        with self._lock:
            return self.short_term_store.get(session_id, [])[-last_n:]

    def clear_short_term(self, session_id: str) -> None:
        """Clears short-term context for a completed session."""
        with self._lock:
            if session_id in self.short_term_store:
                del self.short_term_store[session_id]

    # --- Long-Term Memory Operations ---

    def store_fact(
        self,
        subject: str,
        fact: str,
        source_id: str,
        source_url: str,
        predicate: Optional[str] = None,
        source_type: SourceType = SourceType.MEETING_NOTES,
        confidence: float = 1.0,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
    ) -> MemoryRecord:
        """Stores or updates a durable fact with attribute-level conflict resolution.

        Only supersedes existing facts when BOTH the subject and predicate/attribute match!
        """
        with self._lock:
            clean_subj = subject.strip().lower()
            clean_pred = (predicate or extract_predicate(fact, subject)).strip().lower()

            record = MemoryRecord(
                subject=subject,
                predicate=clean_pred,
                fact=fact,
                source_id=source_id,
                source_url=source_url,
                source_type=source_type,
                confidence=confidence,
                expires_at=expires_at,
                is_active=True,
                metadata=metadata or {},
            )

            # Attribute-level conflict resolution:
            # Only supersede an older fact if BOTH subject AND predicate match!
            for existing in self.long_term_store.values():
                if (
                    existing.is_active
                    and existing.subject.strip().lower() == clean_subj
                    and existing.predicate.strip().lower() == clean_pred
                ):
                    existing.is_active = False
                    existing.superseded_by = record.memory_id
                    self._persist_record(existing)

            self.long_term_store[record.memory_id] = record
            self._persist_record(record)
            return record

    def search_memories(
        self,
        query: str,
        subject: Optional[str] = None,
        include_expired: bool = False,
    ) -> List[MemoryRecord]:
        """Searches active persistent memories by subject or query keyword with stop-word filtering."""
        with self._lock:
            now = datetime.now(timezone.utc)
            results: List[MemoryRecord] = []

            # Extract query tokens, excluding noise stop-words like 'a', 'the', 'is'
            active_tokens: List[str] = []
            if query:
                raw_tokens = re.findall(r"\b\w+\b", query.lower())
                meaningful_tokens = [t for t in raw_tokens if t not in STOP_WORDS]
                if not meaningful_tokens:
                    # Query contains only stop words (e.g. 'a') - reject broad matching
                    return []
                active_tokens = meaningful_tokens

            for record in self.long_term_store.values():
                if not record.is_active:
                    continue

                # Check expiration
                if record.expires_at and record.expires_at < now and not include_expired:
                    continue

                # Subject filter
                if subject and record.subject.strip().lower() != subject.strip().lower():
                    continue

                if not query:
                    results.append(record)
                    continue

                # Tokenize record text into distinct whole words (avoiding substring matching)
                rec_words = set(re.findall(r"\b\w+\b", f"{record.subject} {record.predicate} {record.fact}".lower()))
                matching_count = sum(1 for t in active_tokens if t in rec_words)

                # Match if any meaningful whole-word token matches
                if matching_count > 0:
                    results.append(record)

            return results

    def delete_memory(self, memory_id: str) -> bool:
        """Deletes or deactivates a specific memory record in memory and SQLite."""
        with self._lock:
            if memory_id in self.long_term_store:
                rec = self.long_term_store[memory_id]
                rec.is_active = False
                self._persist_record(rec)
                return True
            return False
