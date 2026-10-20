"""Notion connector for workspace pages, meeting notes, and incident post-mortems."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from livemind.connectors.base import BaseConnector
from livemind.core.models import Document, SourceType


class NotionConnector(BaseConnector):
    """Synchronizes pages, architectural records, and meeting notes from Notion."""

    def __init__(self, workspace_domain: str = "notion.so/workspace"):
        super().__init__(name="notion")
        self.workspace_domain = workspace_domain.rstrip("/")
        self._mock_pages_db: Dict[str, Dict[str, Any]] = {}

    def add_or_update_page(
        self,
        page_id: str,
        title: str,
        body: str,
        page_type: str = "documentation",
        tags: Optional[List[str]] = None,
        author: str = "Engineering Team",
    ) -> None:
        """Helper to simulate or register a Notion workspace page."""
        self._mock_pages_db[page_id] = {
            "page_id": page_id,
            "title": title,
            "body": body,
            "page_type": page_type,
            "tags": tags or [],
            "author": author,
            "updated_at": datetime.now(timezone.utc),
        }

    def sync(self) -> List[Document]:
        """Performs incremental synchronization of workspace pages."""
        synced_docs: List[Document] = []

        for page_id, page in self._mock_pages_db.items():
            tags_str = ", ".join(page["tags"])
            full_content = (
                f"Page Title: {page['title']}\n"
                f"Type: {page['page_type']}\n"
                f"Author: {page['author']}\n"
                f"Tags: {tags_str}\n\n"
                f"{page['body']}"
            )

            if self.is_updated(page_id, full_content):
                new_hash = self.compute_hash(full_content)
                self.document_hashes[page_id] = new_hash

                doc = Document(
                    source_type=SourceType.NOTION,
                    title=page["title"],
                    content=full_content,
                    url=f"https://{self.workspace_domain}/{page_id}",
                    external_id=page_id,
                    metadata={
                        "page_id": page_id,
                        "page_type": page["page_type"],
                        "tags": page["tags"],
                        "author": page["author"],
                    },
                    updated_at=page["updated_at"],
                    content_hash=new_hash,
                )
                synced_docs.append(doc)

        self.last_sync_time = datetime.now(timezone.utc)
        return synced_docs
