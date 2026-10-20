"""Integration tests for LiveMind FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient
from livemind.api.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "LiveMind"


def test_jira_sync_endpoint():
    payload = {
        "ticket_key": "PAY-1842",
        "summary": "Connection pool fix",
        "description": "Increase pool size to 50",
        "status": "In Progress",
    }
    res = client.post("/connectors/jira/sync", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["connector"] == "jira"
    assert data["synced_documents_count"] >= 1
    assert data["total_indexed_chunks"] >= 1


def test_notion_sync_endpoint():
    payload = {
        "page_id": "checkout-postmortem",
        "title": "Incident Postmortem: Checkout Outage",
        "body": "Connection starvation caused the incident.",
        "page_type": "incident_report",
        "tags": ["checkout", "infrastructure"],
    }
    res = client.post("/connectors/notion/sync", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["connector"] == "notion"
    assert data["synced_documents_count"] >= 1


def test_memory_crud_endpoints():
    # 1. Store memory
    payload = {
        "subject": "checkout-monitoring",
        "fact": "Prometheus metric agreed for connection pool.",
        "source_id": "meeting-99",
        "source_url": "https://notion.so/meeting-99",
        "confidence": 0.95,
    }
    create_res = client.post("/memory", json=payload)
    assert create_res.status_code == 200
    mem_data = create_res.json()
    assert mem_data["subject"] == "checkout-monitoring"
    mem_id = mem_data["memory_id"]

    # 2. List memories
    list_res = client.get("/memory?query=Prometheus")
    assert list_res.status_code == 200
    assert any(m["memory_id"] == mem_id for m in list_res.json())

    # 3. Delete memory
    del_res = client.delete(f"/memory/{mem_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deactivated"


def test_research_endpoint():
    res = client.post(
        "/research",
        json={
            "question": "What caused the checkout incident and what is the status of PAY-1842?",
            "session_id": "api_test_session_1",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert len(data["plan"]) >= 1
    assert "findings" in data["final_answer"].lower()
    assert len(data["citations"]) > 0
