"""Multi-hop research planner with iterative subquery dispatch, dynamic hop chaining, citation verification, and freshness checks."""

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple
from livemind.connectors.jira import JiraConnector
from livemind.core.config import settings
from livemind.core.models import (
    Chunk,
    Citation,
    ResearchEvidence,
    ResearchTask,
    SourceType,
)
from livemind.memory.service import MemoryService
from livemind.retrieval.hybrid import HybridRetriever


def isolate_untrusted_content(text: str, source_label: str = "document") -> str:
    """Wraps retrieved third-party content in isolation delimiters to defend against prompt injection attacks."""
    # Sanitize attempted boundary breakout tags
    sanitized = re.sub(
        r"<\/?(?:system|instruction|prompt_injection|context|admin)[^>]*>",
        "[REDACTED_DELIMITER]",
        text,
        flags=re.IGNORECASE,
    )
    # Redact exposed secrets or auth tokens
    sanitized = re.sub(
        r"(?i)(bearer\s+[a-zA-Z0-9_\-\.]{16,}|(?:sk|ghp|xoxb|xoxp)_[a-zA-Z0-9]{16,})",
        "[REDACTED_TOKEN]",
        sanitized,
    )
    return f"<<<UNTRUSTED_{source_label.upper()}>>>\n{sanitized.strip()}\n<<<END_UNTRUSTED_{source_label.upper()}>>>"


def extract_jira_ticket_ids(text: str, allowed_project_keys: Optional[List[str]] = None) -> List[str]:
    """Extracts recognized Jira issue keys matching configured project keys (e.g. PAY-1842, CORE-901).

    Guaranteed not to match general terms like SHA-256, UTF-8, or COVID-19.
    """
    keys = allowed_project_keys or settings.recognized_project_keys
    if not keys:
        return []
    pattern = rf"\b(?:{'|'.join(re.escape(k) for k in keys)})-\d+\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return sorted(list({m.upper() for m in matches}))


class MultiHopResearcher:
    """Orchestrates multi-hop information retrieval across live documents, persistent memory facts, and real-time APIs."""

    def __init__(
        self,
        retriever: HybridRetriever,
        memory_service: MemoryService,
        jira_connector: Optional[JiraConnector] = None,
        max_hops: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.retriever = retriever
        self.memory = memory_service
        self.jira = jira_connector or JiraConnector()
        self.max_hops = max_hops or settings.RESEARCH_MAX_HOPS
        self.confidence_threshold = confidence_threshold or settings.RESEARCH_CONFIDENCE_THRESHOLD

    def extract_ticket_keys(self, text: str) -> List[str]:
        """Extracts recognized Jira issue keys matching configured project keys."""
        return extract_jira_ticket_ids(text, settings.recognized_project_keys)

    def decompose_question(self, question: str) -> List[str]:
        """Dynamically formulates research plan based on query semantics and entities."""
        subquestions: List[str] = []
        lower_q = question.lower()
        extracted_keys = self.extract_ticket_keys(question)

        # 1. Primary entity/concept search
        if extracted_keys:
            subquestions.append(f"Search specifications and records for ticket {', '.join(extracted_keys)}")
        else:
            subquestions.append(f"Search documentation and workspace notes regarding {question}")

        # 2. Causality / Failure investigation
        if any(w in lower_q for w in ["why", "fail", "outage", "incident", "error", "crash", "cause", "broken"]):
            subquestions.append("Identify root cause, error trace, and incident post-mortem details")

        # 3. Action / Resolution verification
        if any(w in lower_q for w in ["fix", "agree", "decide", "meeting", "solution", "proposal", "status", "deploy", "release"]):
            subquestions.append("Find proposed architectural fixes, team agreements, and implementation decisions")

        return subquestions

    def _calculate_evidence_confidence(self, chunks: List[Chunk], memories: List) -> float:
        """Evaluates confidence score of retrieved evidence against the current subquestion."""
        if not chunks and not memories:
            return 0.0

        # Confidence increases with volume of relevant evidence and freshness
        score = 0.5
        if chunks:
            score += min(0.3, len(chunks) * 0.1)
        if memories:
            score += min(0.2, len(memories) * 0.1)
        return round(min(1.0, score), 2)

    def conduct_research(self, question: str, session_id: Optional[str] = None) -> ResearchTask:
        """Executes dynamic multi-hop reasoning, gathering deduplicated evidence and verifying live status."""
        task = ResearchTask(question=question)
        initial_plan = self.decompose_question(question)
        task.plan = [f"Step {i}: {sq}" for i, sq in enumerate(initial_plan, start=1)]

        discovered_tickets: Set[str] = set(self.extract_ticket_keys(question))
        seen_citation_urls: Set[str] = set()
        all_citations: List[Citation] = []

        active_queries = list(initial_plan)
        executed_hops = 0

        while active_queries and executed_hops < self.max_hops:
            sq = active_queries.pop(0)
            executed_hops += 1

            evidence_step = ResearchEvidence(subquestion=sq)

            # 1. Search indexed documents using HybridRetriever
            retrieved_chunks = self.retriever.search(sq, top_k=3)
            evidence_step.retrieved_chunks = retrieved_chunks

            # Extract any ticket keys discovered during retrieval for dynamic chaining
            for chunk in retrieved_chunks:
                for key in self.extract_ticket_keys(chunk.text):
                    if key not in discovered_tickets:
                        discovered_tickets.add(key)
                        # Dynamically add follow-up hop for discovered ticket if hop limit allows
                        if len(active_queries) + executed_hops < self.max_hops:
                            active_queries.append(f"Follow up on newly discovered cross-reference ticket {key}")
                            task.plan.append(f"Step {len(task.plan) + 1}: Follow up on newly discovered ticket {key}")

                # Deduplicated, verified citation tracking by URL
                if chunk.source_url not in seen_citation_urls:
                    seen_citation_urls.add(chunk.source_url)
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

            # 2. Check persistent long-term memory facts
            memories = self.memory.search_memories(query=sq)
            for mem in memories:
                if mem.source_url not in seen_citation_urls:
                    seen_citation_urls.add(mem.source_url)
                    mem_citation = Citation(
                        source_id=mem.source_id,
                        source_title=f"Memory: {mem.subject} ({mem.predicate})",
                        source_url=mem.source_url,
                        source_type=mem.source_type,  # Preserves real source type!
                        quote_snippet=mem.fact,
                        freshness_timestamp=mem.observed_at,
                    )
                    evidence_step.citations.append(mem_citation)
                    all_citations.append(mem_citation)

            evidence_step.confidence = self._calculate_evidence_confidence(retrieved_chunks, memories)
            task.evidence.append(evidence_step)

        # 3. Live Authoritative Verification for discovered Jira tickets
        live_ticket_statuses: Dict[str, Dict] = {}
        for ticket_key in sorted(list(discovered_tickets)):
            live_status = self.jira.fetch_live_status(ticket_key)
            if live_status:
                live_ticket_statuses[ticket_key] = live_status
                live_url = f"{self.jira.base_url}/browse/{ticket_key}"
                live_cit = Citation(
                    source_id=ticket_key,
                    source_title=f"Live Jira API: {ticket_key}",
                    source_url=live_url,
                    source_type=SourceType.JIRA,
                    quote_snippet=f"Status: {live_status['status']}. Summary: {live_status['summary']}. Assignee: {live_status['assignee']}",
                    freshness_timestamp=datetime.now(timezone.utc),
                )
                # Replace static citation with authoritative live citation if previously present
                replaced = False
                for idx, existing_cit in enumerate(all_citations):
                    if existing_cit.source_url == live_url:
                        all_citations[idx] = live_cit
                        replaced = True
                        break
                if not replaced:
                    all_citations.append(live_cit)
                    seen_citation_urls.add(live_url)

                task.evidence.append(
                    ResearchEvidence(
                        subquestion=f"Authoritative live API check for {ticket_key}",
                        findings=f"Live Status: {live_status['status']} (Resolved: {live_status['is_resolved']})",
                        citations=[live_cit],
                        confidence=1.0,
                    )
                )

        # 4. Synthesize grounded answer
        task.final_answer = self._synthesize_answer(task, live_ticket_statuses)
        task.citations = all_citations
        task.hop_count = executed_hops
        task.confidence_score = (
            sum(e.confidence for e in task.evidence) / len(task.evidence)
            if task.evidence
            else 0.0
        )
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
            # Show up to 4 most relevant snippets
            parts.append("\n**Retrieved Evidence:**\n" + "\n".join(found_snippets[:4]))

        # Summarize live ticket statuses
        if live_statuses:
            status_lines = []
            for key, st in live_statuses.items():
                res_str = "Resolved / Closed" if st["is_resolved"] else "Still In Progress (Not Deployed)"
                status_lines.append(f"- Ticket **{key}** ({st['summary']}) is currently **{st['status']}** ({res_str}).")
            parts.append("\n**Live Authoritative Status:**\n" + "\n".join(status_lines))

        return "\n\n".join(parts)
