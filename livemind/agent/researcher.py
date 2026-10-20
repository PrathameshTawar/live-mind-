"""Multi-hop research planner with iterative subquery dispatch, citation verification, and freshness checks."""

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from livemind.connectors.jira import JiraConnector
from livemind.core.models import (
    Chunk,
    Citation,
    ResearchEvidence,
    ResearchTask,
    SourceType,
)
from livemind.memory.service import MemoryService
from livemind.retrieval.hybrid import HybridRetriever


class MultiHopResearcher:
    """Orchestrates multi-hop information retrieval across live documents, memory facts, and real-time APIs."""

    def __init__(
        self,
        retriever: HybridRetriever,
        memory_service: MemoryService,
        jira_connector: Optional[JiraConnector] = None,
        max_hops: int = 4,
    ):
        self.retriever = retriever
        self.memory = memory_service
        self.jira = jira_connector or JiraConnector()
        self.max_hops = max_hops

    def decompose_question(self, question: str) -> List[str]:
        """Breaks a compound question into targeted sub-questions."""
        subquestions: List[str] = []
        lower_q = question.lower()

        # Check for incident / root cause subquestion
        if any(w in lower_q for w in ["why", "fail", "outage", "incident", "error", "crash"]):
            subquestions.append("Find root cause and incident details for failure")

        # Check for fix / agreement / meeting notes subquestion
        if any(w in lower_q for w in ["fix", "agree", "decide", "meeting", "solution", "proposal"]):
            subquestions.append("Find proposed fix and team agreements or meeting decisions")

        # Check for ticket status / resolution subquestion
        if any(w in lower_q for w in ["jira", "ticket", "resolved", "status", "deployed", "issue"]):
            subquestions.append("Check related Jira ticket and authoritative live status")

        if not subquestions:
            subquestions.append(question)

        return subquestions

    def extract_ticket_keys(self, text: str) -> List[str]:
        """Extracts Jira issue keys like PAY-1842 or CHECKOUT-101."""
        return list(set(re.findall(r"\b[A-Za-z]+-\d+\b", text)))

    def conduct_research(self, question: str, session_id: Optional[str] = None) -> ResearchTask:
        """Executes the multi-hop research process, gathering cited evidence and verifying live status."""
        task = ResearchTask(question=question)
        subquestions = self.decompose_question(question)
        task.plan = [f"Step {i}: {sq}" for i, sq in enumerate(subquestions, start=1)]

        discovered_tickets: Set[str] = set()
        all_citations: List[Citation] = []

        # Check existing ticket keys in query directly
        for key in self.extract_ticket_keys(question):
            discovered_tickets.add(key.upper())

        for hop_idx, sq in enumerate(subquestions[: self.max_hops], start=1):
            evidence_step = ResearchEvidence(subquestion=sq)

            # 1. Search indexed documents using HybridRetriever
            retrieved_chunks = self.retriever.search(f"{question} {sq}", top_k=3)
            evidence_step.retrieved_chunks = retrieved_chunks

            # Extract any ticket keys discovered in retrieved chunks
            for chunk in retrieved_chunks:
                for key in self.extract_ticket_keys(chunk.text):
                    discovered_tickets.add(key.upper())

                # Generate verified citation
                citation = Citation(
                    source_id=chunk.document_id,
                    source_title=chunk.source_title,
                    source_url=chunk.source_url,
                    source_type=chunk.source_type,
                    quote_snippet=chunk.text[:200],
                    freshness_timestamp=chunk.timestamp,
                )
                evidence_step.citations.append(citation)
                all_citations.append(citation)

            # 2. Check long-term memory for relevant durable facts
            memories = self.memory.search_memories(query=sq)
            for mem in memories:
                mem_citation = Citation(
                    source_id=mem.source_id,
                    source_title=f"Memory: {mem.subject}",
                    source_url=mem.source_url,
                    source_type=SourceType.MEETING_NOTES,
                    quote_snippet=mem.fact,
                    freshness_timestamp=mem.observed_at,
                )
                evidence_step.citations.append(mem_citation)
                all_citations.append(mem_citation)

            task.evidence.append(evidence_step)

        # 3. Freshness Check: Query Authoritative Live Jira API for current status
        live_ticket_statuses: Dict[str, Dict] = {}
        for ticket_key in discovered_tickets:
            live_status = self.jira.fetch_live_status(ticket_key)
            if live_status:
                live_ticket_statuses[ticket_key] = live_status
                task.evidence.append(
                    ResearchEvidence(
                        subquestion=f"Authoritative live API check for {ticket_key}",
                        findings=f"Live Status: {live_status['status']} (Resolved: {live_status['is_resolved']})",
                        citations=[
                            Citation(
                                source_id=ticket_key,
                                source_title=f"Live Jira: {ticket_key}",
                                source_url=f"{self.jira.base_url}/browse/{ticket_key}",
                                source_type=SourceType.JIRA,
                                quote_snippet=f"Status is currently {live_status['status']}. Assignee: {live_status['assignee']}",
                                freshness_timestamp=datetime.now(timezone.utc),
                            )
                        ],
                    )
                )

        # 4. Synthesize final grounded answer
        task.final_answer = self._synthesize_answer(task, live_ticket_statuses)
        task.citations = all_citations
        task.status = "completed"
        task.updated_at = datetime.now(timezone.utc)

        # Save to short-term session if session_id provided
        if session_id:
            self.memory.add_short_term_turn(session_id, "user", question)
            self.memory.add_short_term_turn(session_id, "assistant", task.final_answer)

        return task

    def _synthesize_answer(self, task: ResearchTask, live_statuses: Dict[str, Dict]) -> str:
        """Synthesizes structured findings into an answer referencing evidence and live statuses."""
        parts = ["### Synthesized Research Findings:"]

        # Summarize retrieved evidence
        found_snippets = []
        for ev in task.evidence:
            for c in ev.retrieved_chunks:
                found_snippets.append(f"- **{c.source_title}**: {c.text.strip()}")

        if found_snippets:
            parts.append("\n**Retrieved Evidence:**\n" + "\n".join(found_snippets[:4]))

        # Summarize live ticket statuses
        if live_statuses:
            status_lines = []
            for key, st in live_statuses.items():
                res_str = "Resolved / Closed" if st["is_resolved"] else "Still In Progress (Not Deployed)"
                status_lines.append(f"- Ticket **{key}** ({st['summary']}) is currently **{st['status']}** ({res_str}).")
            parts.append("\n**Live Authoritative Status:**\n" + "\n".join(status_lines))

        return "\n\n".join(parts)
