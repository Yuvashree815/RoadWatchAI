"""
RoadWatch AI — Automated Tests for Enhancements
Covers:
- LangSmith configuration & safe observability setup
- Mock LLM mode (zero Gemini quota usage)
- Mock Email mode & Real SMTP service
- Complaint ID generation and end-to-end propagation
- LangGraph workflow submission routing (approved vs rejected)
- PDF attachment to complaint email
- FastAPI endpoints (/api/config, /api/analyze, SSE stream)
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock

from backend.config import settings
from backend.observability import configure_langsmith, get_sanitized_config
from backend.services.email_service import EmailSubmissionService
from backend.graph.state import GraphState
from backend.graph.workflow import build_roadwatch_graph
from backend.api.service import AnalysisService, default_analysis_service
from backend.llm import get_llm, MockChatMultimodal
from backend.database.repository import DatabaseRepository
from backend.rag.hybrid_search import HybridSearcher


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_evidence_state() -> GraphState:
    """Fixture providing a complete, verified analysis state ready for submission."""
    return {
        "run_id": "run-test-1234",
        "complaint_id": "DEMO-COMPLAINT-RW26-TEST01",
        "image_url": "test_pothole.jpg",
        "user_location_hint": "Synthetic Road 1",
        "exif_gps": None,
        "vision_result": {
            "pothole_detected": True,
            "severity": "moderate",
            "confidence": 0.95,
            "visual_evidence": "Moderate pothole with surface asphalt disintegration.",
        },
        "location_result": {
            "resolution_method": "demo_mapping",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "estimated_road_name": "Synthetic Road 1",
            "location_hint_used": "Synthetic Road 1",
            "confidence": 0.95,
            "demo_case_id": "demo-1",
            "notes": "Resolved via synthetic mapping.",
        },
        "road_data": {
            "road": {
                "road_id": "RD-001",
                "road_name": "Synthetic Road 1",
                "district": "North District",
                "area": "Sector A",
            },
            "project": {
                "project_id": "PRJ-001",
                "contract_id": "CNT-001",
                "maintenance_type": "Resurfacing",
                "status": "Active",
            },
            "match_method": "direct",
            "confidence": 1.0,
            "notes": "Direct match",
        },
        "contract_data": {
            "retrieved_chunks": [],
            "best_contract_id": "CNT-001",
            "best_tender_reference": "TND-001",
            "contract_record": {
                "contract_id": "CNT-001",
                "tender_reference": "TND-001",
                "title": "North Corridor Resurfacing",
                "start_date": "2025-01-01",
                "end_date": "2026-12-31",
                "contract_value": 500000.0,
            },
            "contractor_record": {
                "contractor_id": "CTR-001",
                "contractor_name": "Acme Roadworks Synthetic Ltd",
                "contact_email": "contact@acme.local",
                "contact_phone": "+1-555-0100",
                "rating": 4.5,
            },
            "rag_confidence": 0.9,
            "structured_match": True,
            "notes": "Matched contract",
        },
        "officer_data": {
            "officer": {
                "officer_id": "OFF-002",
                "officer_name": "Jane Doe 2",
                "department": "Department of Synthetic Works",
                "role": "Chief Infrastructure Inspector",
                "jurisdiction": "North District",
            },
            "match_method": "project_link",
            "confidence": 1.0,
            "notes": "Direct project officer match",
        },
        "evidence_conflicts": [],
        "verification_confidence": 0.95,
        "requires_human_review": False,
        "human_feedback": None,
        "complaint_record": {
            "complaint_id": "DEMO-COMPLAINT-RW26-TEST01",
            "run_id": "run-test-1234",
            "generated_at": "2026-08-24T12:00:00Z",
            "disclaimer": "SYNTHETIC DEMO RECORD",
            "issue_description": "Pothole detected on Synthetic Road 1.",
            "severity": "moderate",
            "pothole_detected": True,
            "vision_confidence": 0.95,
            "location_summary": "Synthetic Road 1 (37.7749, -122.4194)",
            "road": {"road_id": "RD-001", "road_name": "Synthetic Road 1", "district": "North District"},
            "maintenance_project": {"project_id": "PRJ-001", "status": "Active"},
            "contract": {"contract_id": "CNT-001"},
            "contractor": {"contractor_name": "Acme Roadworks Synthetic Ltd"},
            "responsible_officer": {"officer_name": "Jane Doe 2", "officer_id": "OFF-002"},
            "verification_status": "VERIFIED",
            "verification_confidence": 0.95,
        },
        "final_quality_score": 94.0,
        "quality_explanation": "Deterministic score: 94/100",
        "submission_status": "QUALITY_APPROVED",
        "submission_result": None,
    }


# ── 1. LangSmith Tests ────────────────────────────────────────────────────────

def test_langsmith_configuration_disabled(monkeypatch):
    """Verifies that when LangSmith tracing is disabled, environment variables reflect it."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    configured = configure_langsmith()
    assert configured is False
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"


def test_langsmith_configuration_enabled(monkeypatch):
    """Verifies that when LangSmith tracing is enabled with key, variables are correctly configured."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__mock_test_key_12345")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__mock_test_key_12345")
    monkeypatch.setenv("LANGSMITH_PROJECT", "TestProject")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "TestProject")
    configured = configure_langsmith()
    assert configured is True
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGCHAIN_PROJECT") == "TestProject"


def test_sanitized_config_does_not_leak_secrets():
    """Verifies get_sanitized_config does not return raw API keys or passwords."""
    cfg = get_sanitized_config()
    assert "smtp_password" not in cfg
    assert "gemini_api_key" not in cfg
    assert "langsmith_api_key" not in cfg
    assert isinstance(cfg["langsmith_configured"], bool)
    assert isinstance(cfg["gemini_configured"], bool)


# ── 2. Mock LLM & Complaint ID Tests ──────────────────────────────────────────

def test_mock_llm_mode():
    """Verifies that MockChatMultimodal returns realistic structured vision output."""
    mock_llm = get_llm(force_mock=True)
    assert isinstance(mock_llm, MockChatMultimodal)
    resp = mock_llm.invoke([])
    parsed = json.loads(resp.content)
    assert parsed["pothole_detected"] is True
    assert parsed["severity"] == "moderate"
    assert parsed["confidence"] == 0.95


def test_complaint_id_propagation_in_state():
    """Verifies that a complaint ID initialized in GraphState is preserved across workflow execution."""
    custom_cid = "DEMO-COMPLAINT-RW26-CUSTOM99"
    initial_state = {
        "run_id": "run-prop-01",
        "complaint_id": custom_cid,
        "image_url": "test.jpg",
        "user_location_hint": "Synthetic Road 1",
        "vision_result": {
            "pothole_detected": True,
            "severity": "moderate",
            "confidence": 0.95,
            "visual_evidence": "Pothole detected",
        },
    }

    db = DatabaseRepository()
    mock_email_svc = EmailSubmissionService(email_enabled=True, mock_email=True)
    app = build_roadwatch_graph(db=db, email_service=mock_email_svc)
    result = app.invoke(initial_state)

    assert result["complaint_id"] == custom_cid
    assert result["complaint_record"]["complaint_id"] == custom_cid
    assert result["submission_status"] == "SUBMITTED"
    assert result["submission_result"]["complaint_id"] == custom_cid


# ── 3. Email Submission Service Tests ─────────────────────────────────────────

def test_email_submission_mock_mode(mock_evidence_state):
    """Verifies mock email submission formats message, attaches PDF, and returns SUBMITTED without SMTP."""
    svc = EmailSubmissionService(
        email_enabled=True,
        mock_email=True,
        demo_email="test-demo@roadwatch.local",
    )
    result = svc.submit_complaint(mock_evidence_state)

    assert result["status"] == "SUBMITTED"
    assert result["is_mock"] is True
    assert result["recipient"] == "test-demo@roadwatch.local"
    assert result["pdf_attached"] is True
    assert "DEMO-COMPLAINT-RW26-TEST01" in result["subject"]


def test_email_submission_disabled(mock_evidence_state):
    """Verifies email submission returns SUBMISSION_SKIPPED when EMAIL_ENABLED=false."""
    svc = EmailSubmissionService(email_enabled=False)
    result = svc.submit_complaint(mock_evidence_state)

    assert result["status"] == "SUBMISSION_SKIPPED"
    assert result["pdf_attached"] is False
    assert "disabled" in result["reason"]


def test_email_submission_invalid_recipient(mock_evidence_state):
    """Verifies email submission returns SUBMISSION_FAILED when recipient is invalid."""
    svc = EmailSubmissionService(email_enabled=True, mock_email=True, demo_email="invalid-address")
    result = svc.submit_complaint(mock_evidence_state)

    assert result["status"] == "SUBMISSION_FAILED"
    assert "Invalid recipient" in result["error"]


def test_email_submission_real_smtp_failure(mock_evidence_state):
    """Verifies real SMTP errors are caught and recorded as SUBMISSION_FAILED."""
    svc = EmailSubmissionService(
        email_enabled=True,
        mock_email=False,
        smtp_host="invalid.smtp.host.local",
        smtp_port=587,
        demo_email="test@example.com",
    )
    result = svc.submit_complaint(mock_evidence_state)

    assert result["status"] == "SUBMISSION_FAILED"
    assert result["is_mock"] is False
    assert "SMTP transmission error" in result["error"]


# ── 4. Workflow Quality & Submission Routing Tests ────────────────────────────

def test_workflow_submission_rejected_when_no_pothole():
    """Verifies that if vision reports no pothole detected, the submission node is rejected."""
    state = {
        "run_id": "run-no-pothole",
        "image_url": "clear_road.jpg",
        "user_location_hint": "Synthetic Road 1",
        "vision_result": {
            "pothole_detected": False,
            "severity": "none",
            "confidence": 0.99,
            "visual_evidence": "Clear road, no damage.",
        },
    }
    db = DatabaseRepository()
    app = build_roadwatch_graph(db=db)
    result = app.invoke(state)

    assert result["submission_status"] == "QUALITY_REJECTED"
    assert result["submission_result"]["status"] == "REJECTED"
    assert "No road damage" in result["submission_result"]["reason"]


def test_workflow_submission_rejected_on_low_quality():
    """Verifies that if quality score is below threshold, submission is rejected."""
    state = {
        "run_id": "run-low-qual",
        "image_url": "test.jpg",
        "user_location_hint": "Nonexistent Road Nowhere",
        "vision_result": {
            "pothole_detected": True,
            "severity": "low",
            "confidence": 0.3,
            "visual_evidence": "Blurry image",
        },
    }
    db = DatabaseRepository()
    app = build_roadwatch_graph(db=db)
    result = app.invoke(state)

    assert result["final_quality_score"] < 50.0
    assert result["submission_status"] == "QUALITY_REJECTED"
    assert result["submission_result"]["status"] == "REJECTED"


# ── 5. FastAPI Endpoints & Config Diagnostics ─────────────────────────────────

def test_api_config_endpoint():
    """Verifies GET /api/config returns sanitized configuration with expected keys."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "demo_mode" in data
    assert "mock_email" in data
    assert "gemini_model" in data
    assert "langsmith_tracing" in data
