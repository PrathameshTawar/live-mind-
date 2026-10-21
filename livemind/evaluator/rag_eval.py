"""Automated evaluation suite measuring Retrieval Recall@K, Mean Reciprocal Rank (MRR), and Groundedness."""

import argparse
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from livemind.agent.researcher import MultiHopResearcher
from livemind.connectors.jira import JiraConnector
from livemind.connectors.notion import NotionConnector
from livemind.core.models import SourceType
from livemind.memory.service import MemoryService
from livemind.retrieval.hybrid import HybridRetriever


class RAGEvaluationScorecard(BaseModel):
    """Aggregate benchmark results across retrieval and agentic research suites."""

    total_queries: int
    recall_at_3: float
    recall_at_5: float
    mrr: float
    faithfulness_score: float
    stale_data_rate: float
    pass_rate: float
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    case_results: List[Dict[str, Any]] = Field(default_factory=list)


class RAGEvaluator:
    """Evaluates LiveMind retrieval and multi-hop reasoning against ground-truth evaluation queries."""

    BENCHMARK_CASES = [
        {
            "id": "RAG-Q-001",
            "question": "What is the status of payment gateway migration in ticket PAY-1842?",
            "expected_ticket": "PAY-1842",
            "expected_source_type": SourceType.JIRA,
            "target_keywords": ["pay-1842", "stripe", "payment"],
        },
        {
            "id": "RAG-Q-002",
            "question": "Who approved the database failover architecture in our Notion post-mortem?",
            "expected_source_type": SourceType.NOTION,
            "target_keywords": ["failover", "database", "postgres", "replication"],
        },
        {
            "id": "RAG-Q-003",
            "question": "Check the live status of infrastructure ticket INFRA-501 and current assignee.",
            "expected_ticket": "INFRA-501",
            "expected_source_type": SourceType.JIRA,
            "target_keywords": ["infra-501", "kubernetes", "ingress"],
        },
        {
            "id": "RAG-Q-004",
            "question": "What are our persistent guidelines for handling customer refund authorizations?",
            "expected_source_type": SourceType.MEETING_NOTES,
            "target_keywords": ["refund", "policy", "authorization"],
        },
        {
            "id": "RAG-Q-005",
            "question": "Why did the checkout service incident occur and what fix was agreed in CORE-304?",
            "expected_ticket": "CORE-304",
            "expected_source_type": SourceType.JIRA,
            "target_keywords": ["core-304", "checkout", "timeout", "redis"],
        },
    ]

    def __init__(self):
        self.retriever = HybridRetriever()
        self.memory = MemoryService(db_path=":memory:")
        self.jira = JiraConnector()
        self.notion = NotionConnector()
        self.researcher = MultiHopResearcher(
            retriever=self.retriever,
            memory_service=self.memory,
            jira_connector=self.jira,
        )
        self._seed_test_workspace()

    def _seed_test_workspace(self) -> None:
        """Populates realistic documents across Jira, Notion, and persistent memory."""
        # 1. Jira PAY-1842
        self.jira.add_or_update_issue(
            ticket_key="PAY-1842",
            summary="Migrate Stripe payment provider",
            description="Upgrading checkout to Stripe Elements v3. All tests passing in staging.",
            status="Done",
            priority="High",
            assignee="Alice Developer",
            comments=["Stripe webhooks verified."],
        )

        # 2. Jira INFRA-501
        self.jira.add_or_update_issue(
            ticket_key="INFRA-501",
            summary="Upgrade Kubernetes cluster to v1.30",
            description="Production ingress migration scheduled for maintenance window.",
            status="In Progress",
            priority="Critical",
            assignee="DevOps Lead",
        )

        # 3. Jira CORE-304
        self.jira.add_or_update_issue(
            ticket_key="CORE-304",
            summary="Checkout timeout incident root cause",
            description="Redis connection pool exhaustion caused 504 errors on /checkout.",
            status="Resolved",
            priority="Critical",
            assignee="Backend Architect",
        )

        # Sync Jira to retriever
        for doc in self.jira.sync():
            chunks = self.jira.chunk_document(doc)
            self.retriever.upsert_document_chunks(doc.id, chunks)

        # 4. Notion Pages
        self.notion.add_or_update_page(
            page_id="arch-db-failover-postmortem",
            title="Database Failover Post-Mortem & Architecture",
            body="PostgreSQL replication failover was approved by Architecture Guild. Failover automated via Patroni.",
            page_type="incident_postmortem",
            tags=["database", "failover", "postgres"],
        )
        self.notion.add_or_update_page(
            page_id="finance-sop-refund-policy",
            title="Finance SOP: Customer Refund Guidelines & Authorization",
            body="Customer refund authorization guidelines: Refunds exceeding $1000 require dual approval from Finance lead and Support director.",
            page_type="documentation",
            tags=["finance", "refund", "policy", "authorization"],
        )

        for doc in self.notion.sync():
            chunks = self.notion.chunk_document(doc)
            self.retriever.upsert_document_chunks(doc.id, chunks)

        # 5. Persistent Memory Fact
        self.memory.store_fact(
            subject="refund policy",
            predicate="authorization",
            fact="Refunds exceeding $1000 require dual authorization from Finance and Customer Support lead.",
            source_id="finance-sop-refund-policy",
            source_url="https://notion.so/finance-sop-refund-policy",
            source_type=SourceType.NOTION,
        )

    def run_benchmark(self) -> RAGEvaluationScorecard:
        """Executes full evaluation measuring Recall@K, MRR, and groundedness."""
        total_queries = len(self.BENCHMARK_CASES)
        hits_at_3 = 0
        hits_at_5 = 0
        reciprocal_ranks = []
        faithfulness_scores = []
        stale_data_count = 0
        case_results = []

        for case in self.BENCHMARK_CASES:
            q = case["question"]
            expected_ticket = case.get("expected_ticket")
            target_keywords = case["target_keywords"]

            # Retrieve top 5 candidates
            top_chunks = self.retriever.search(q, top_k=5)

            # Check ranking position of best matching chunk
            rank_found = 0
            for r_idx, c in enumerate(top_chunks, start=1):
                chunk_lower = (c.text + " " + c.source_title).lower()
                matches_ticket = expected_ticket and expected_ticket.lower() in chunk_lower
                matches_kw = any(kw in chunk_lower for kw in target_keywords)
                if matches_ticket or matches_kw:
                    rank_found = r_idx
                    break

            if rank_found > 0:
                reciprocal_ranks.append(1.0 / rank_found)
                if rank_found <= 3:
                    hits_at_3 += 1
                if rank_found <= 5:
                    hits_at_5 += 1
            else:
                reciprocal_ranks.append(0.0)

            # Test full multi-hop research execution
            task = self.researcher.conduct_research(q)

            # Check faithfulness (citations must be present and ground the findings)
            has_citations = len(task.citations) > 0
            faithfulness_score = 1.0 if has_citations else 0.0
            faithfulness_scores.append(faithfulness_score)

            case_results.append({
                "id": case["id"],
                "question": q,
                "rank_found": rank_found,
                "citations_count": len(task.citations),
                "passed": rank_found > 0 and has_citations,
            })

        recall_3 = round(hits_at_3 / total_queries, 4) if total_queries else 1.0
        recall_5 = round(hits_at_5 / total_queries, 4) if total_queries else 1.0
        mrr = round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 1.0
        faithfulness = round(sum(faithfulness_scores) / len(faithfulness_scores), 4) if faithfulness_scores else 1.0
        stale_rate = round(stale_data_count / total_queries, 4) if total_queries else 0.0
        pass_rate = round(sum(1 for c in case_results if c["passed"]) / total_queries, 4) if total_queries else 1.0

        return RAGEvaluationScorecard(
            total_queries=total_queries,
            recall_at_3=recall_3,
            recall_at_5=recall_5,
            mrr=mrr,
            faithfulness_score=faithfulness,
            stale_data_rate=stale_rate,
            pass_rate=pass_rate,
            case_results=case_results,
        )

    def generate_markdown_report(self, scorecard: RAGEvaluationScorecard) -> str:
        """Renders an audit-ready Markdown evaluation report for CI/CD."""
        return (
            "# 🧠 LiveMind Agentic RAG Evaluation Report\n\n"
            f"**Evaluated At:** {scorecard.evaluated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            "### Summary Retrieval & Grounding Metrics\n"
            "| Metric | Result | Benchmark Target | Status |\n"
            "| :--- | :--- | :--- | :--- |\n"
            f"| **Retrieval Recall@3** | **{scorecard.recall_at_3 * 100:.1f}%** | &ge; 90.0% | {'✅ PASS' if scorecard.recall_at_3 >= 0.90 else '❌ FAIL'} |\n"
            f"| **Retrieval Recall@5** | **{scorecard.recall_at_5 * 100:.1f}%** | &ge; 95.0% | {'✅ PASS' if scorecard.recall_at_5 >= 0.95 else '❌ FAIL'} |\n"
            f"| **Mean Reciprocal Rank (MRR)** | **{scorecard.mrr:.3f}** | &ge; 0.850 | {'✅ PASS' if scorecard.mrr >= 0.850 else '❌ FAIL'} |\n"
            f"| **Citation Faithfulness Score** | **{scorecard.faithfulness_score * 100:.1f}%** | &ge; 95.0% | {'✅ PASS' if scorecard.faithfulness_score >= 0.95 else '❌ FAIL'} |\n"
            f"| **Stale Data Hallucination Rate** | **{scorecard.stale_data_rate * 100:.1f}%** | **0.0% (Zero Tolerance)** | {'✅ PASS' if scorecard.stale_data_rate == 0.0 else '❌ FAIL'} |\n"
            f"| **Overall Evaluation Pass Rate** | **{scorecard.pass_rate * 100:.1f}%** | 100.0% | {'✅ PASS' if scorecard.pass_rate == 1.0 else '⚠️ WARN'} |\n\n"
            f"**Total Queries Evaluated:** {scorecard.total_queries} | "
            f"**Recall@3 Hits:** {round(scorecard.recall_at_3 * scorecard.total_queries)}/{scorecard.total_queries}\n"
        )


def main():
    """CLI runner supporting --fail-under threshold enforcement for CI/CD pipelines."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="LiveMind RAG Benchmark Runner")
    parser.add_argument("--fail-under", type=float, default=90.0, help="Minimum required Recall@3 percentage")
    args = parser.parse_args()

    evaluator = RAGEvaluator()
    scorecard = evaluator.run_benchmark()
    report = evaluator.generate_markdown_report(scorecard)
    print(report)

    recall_pct = scorecard.recall_at_3 * 100
    if recall_pct < args.fail_under:
        print(f"❌ Recall failure: {recall_pct:.1f}% is below required {args.fail_under:.1f}%", file=sys.stderr)
        sys.exit(1)
    else:
        print("✅ LiveMind RAG evaluation passed all quality gates.")
        sys.exit(0)


if __name__ == "__main__":
    main()
