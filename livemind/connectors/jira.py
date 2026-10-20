"""Jira live connector for issue synchronization, metadata normalization, and live status inspection."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from livemind.connectors.base import BaseConnector
from livemind.core.models import Document, SourceType


class JiraConnector(BaseConnector):
    """Synchronizes and queries live issues, transitions, and comments from Jira."""

    def __init__(self, base_url: str = "https://jira.company.internal"):
        super().__init__(name="jira")
        self.base_url = base_url.rstrip("/")
        # In-memory store representing live Jira service
        self._mock_issues_db: Dict[str, Dict[str, Any]] = {}

    def add_or_update_issue(
        self,
        ticket_key: str,
        summary: str,
        description: str,
        status: str = "Open",
        priority: str = "Medium",
        assignee: Optional[str] = None,
        comments: Optional[List[str]] = None,
    ) -> None:
        """Helper to simulate or register a Jira issue payload."""
        self._mock_issues_db[ticket_key] = {
            "key": ticket_key,
            "summary": summary,
            "description": description,
            "status": status,
            "priority": priority,
            "assignee": assignee or "Unassigned",
            "comments": comments or [],
            "updated_at": datetime.now(timezone.utc),
        }

    def fetch_live_status(self, ticket_key: str) -> Optional[Dict[str, Any]]:
        """Direct authoritative API lookup for real-time ticket status, bypassing cached indexes."""
        issue = self._mock_issues_db.get(ticket_key)
        if not issue:
            return None
        return {
            "ticket_key": issue["key"],
            "status": issue["status"],
            "summary": issue["summary"],
            "assignee": issue["assignee"],
            "updated_at": issue["updated_at"],
            "is_resolved": issue["status"].lower() in ["done", "resolved", "closed"],
        }

    def sync(self) -> List[Document]:
        """Synchronizes issues incrementally, emitting only new or modified documents."""
        synced_docs: List[Document] = []

        for key, issue in self._mock_issues_db.items():
            comments_text = "\n".join(f"- Comment: {c}" for c in issue["comments"])
            full_content = (
                f"Issue Key: {issue['key']}\n"
                f"Summary: {issue['summary']}\n"
                f"Status: {issue['status']}\n"
                f"Priority: {issue['priority']}\n"
                f"Assignee: {issue['assignee']}\n\n"
                f"Description:\n{issue['description']}\n\n"
                f"Activity & Comments:\n{comments_text}"
            )

            # Check if updated since last sync
            if self.is_updated(key, full_content):
                new_hash = self.compute_hash(full_content)
                self.document_hashes[key] = new_hash

                doc = Document(
                    source_type=SourceType.JIRA,
                    title=f"[{issue['key']}] {issue['summary']}",
                    content=full_content,
                    url=f"{self.base_url}/browse/{issue['key']}",
                    external_id=key,
                    metadata={
                        "ticket_key": issue["key"],
                        "status": issue["status"],
                        "priority": issue["priority"],
                        "assignee": issue["assignee"],
                    },
                    updated_at=issue["updated_at"],
                    content_hash=new_hash,
                )
                synced_docs.append(doc)

        self.last_sync_time = datetime.now(timezone.utc)
        return synced_docs
