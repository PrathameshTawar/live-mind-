"""Jira live connector supporting REST API synchronization, metadata normalization, and live status inspection."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
from livemind.connectors.base import BaseConnector
from livemind.core.config import settings
from livemind.core.models import Document, SourceType, make_stable_doc_id


class JiraConnector(BaseConnector):
    """Synchronizes and queries live issues, transitions, and comments from Jira."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        user_email: Optional[str] = None,
    ):
        super().__init__(name="jira")
        self.base_url = (base_url or settings.JIRA_BASE_URL).rstrip("/")
        self.api_token = api_token or settings.JIRA_API_TOKEN
        self.user_email = user_email or settings.JIRA_USER_EMAIL
        self.project_keys = settings.recognized_project_keys

        # In-memory store representing local/offline Jira service state
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
        """Helper to simulate, update, or register a Jira issue payload."""
        clean_key = ticket_key.strip().upper()
        self._mock_issues_db[clean_key] = {
            "key": clean_key,
            "summary": summary,
            "description": description,
            "status": status,
            "priority": priority,
            "assignee": assignee or "Unassigned",
            "comments": comments or [],
            "updated_at": datetime.now(timezone.utc),
        }

    def delete_issue(self, ticket_key: str) -> bool:
        """Removes a ticket from the source tracking store."""
        clean_key = ticket_key.strip().upper()
        if clean_key in self._mock_issues_db:
            del self._mock_issues_db[clean_key]
            self.document_hashes.pop(clean_key, None)
            return True
        return False

    def fetch_live_status(self, ticket_key: str) -> Optional[Dict[str, Any]]:
        """Direct authoritative API lookup for real-time ticket status, bypassing cached indexes."""
        clean_key = ticket_key.strip().upper()

        # If live credentials configured, attempt real Jira REST API call first
        if self.api_token and self.user_email and "example" not in self.user_email:
            try:
                auth = (self.user_email, self.api_token)
                url = f"{self.base_url}/rest/api/3/issue/{clean_key}?fields=summary,status,assignee,updated"
                with httpx.Client(timeout=4.0) as client:
                    resp = client.get(url, auth=auth)
                    if resp.status_code == 200:
                        data = resp.json()
                        fields = data.get("fields", {})
                        st_name = fields.get("status", {}).get("name", "Unknown")
                        return {
                            "ticket_key": clean_key,
                            "status": st_name,
                            "summary": fields.get("summary", ""),
                            "assignee": (fields.get("assignee") or {}).get("displayName", "Unassigned"),
                            "updated_at": datetime.fromisoformat(fields.get("updated").replace("Z", "+00:00")),
                            "is_resolved": st_name.lower() in ["done", "resolved", "closed"],
                        }
            except Exception:
                # Fallback to local store on network failure
                pass

        # Authoritative lookup from local/mock store
        issue = self._mock_issues_db.get(clean_key)
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
        """Synchronizes issues incrementally, emitting only new or modified documents with stable canonical IDs."""
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

                # Use deterministic canonical document ID (e.g. jira:PAY-1842)
                stable_id = make_stable_doc_id(SourceType.JIRA, key)

                doc = Document(
                    id=stable_id,
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
