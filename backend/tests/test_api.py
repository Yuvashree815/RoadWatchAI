"""
Comprehensive tests for Milestone 6: FastAPI Backend API, Image Upload, and SSE Streaming.

Uses FastAPI TestClient with mocked LLM for offline testing without real API keys.
"""
import io
import os
import sys
import json
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.main import app
from backend.api.service import default_analysis_service, AnalysisService
from backend.database.repository import DatabaseRepository
from backend.rag.vector_store import VectorStoreManager
from backend.rag.keyword_search import KeywordSearchManager
from backend.rag.hybrid_search import HybridSearcher


# ── Shared Test Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db_repo():
    return DatabaseRepository()


@pytest.fixture(scope="module")
def hybrid_searcher():
    vm = VectorStoreManager(persist_directory="./chroma_test_db")
    vm.ingest_documents()
    km = KeywordSearchManager()
    return HybridSearcher(vm, km)


@pytest.fixture(autouse=True)
def inject_mock_service(db_repo, hybrid_searcher):
    """
    Injects a mock LLM and test hybrid searcher into default_analysis_service for API tests.
    """
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "pothole_detected": True,
        "severity": "moderate",
        "confidence": 0.88,
        "visual_evidence": "Visible asphalt pothole in right lane.",
    })
    mock_llm.invoke.return_value = mock_response

    # Configure default_analysis_service
    default_analysis_service.llm = mock_llm
    default_analysis_service.db = db_repo
    default_analysis_service._hybrid_searcher = hybrid_searcher
    default_analysis_service._app = None  # Force rebuild with mock


@pytest.fixture
def client():
    return TestClient(app)


def make_dummy_image(filename="test_pothole.jpg", content_type="image/jpeg", content=b"\xFF\xD8\xFF\xE0\x00\x10JFIF"):
    """Helper to create dummy in-memory image files for upload tests."""
    return (filename, io.BytesIO(content), content_type)


# ══════════════════════════════════════════════════════════════════════════════
# 1. HEALTH CHECK & BASIC ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def test_health_check(client):
    """GET /health returns 200 with operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "RoadWatch AI backend is running." in data["message"]
    assert data["version"] == "0.1.0"


def test_cors_headers(client):
    """Verify CORS middleware headers on preflight requests."""
    response = client.options(
        "/api/analyze",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ANALYSIS ENDPOINT (POST /api/analyze)
# ══════════════════════════════════════════════════════════════════════════════

def test_analyze_with_location_hint(client):
    """POST /api/analyze with valid image and location hint succeeds end-to-end."""
    file_tuple = make_dummy_image("pothole_007.jpg", "image/jpeg")
    response = client.post(
        "/api/analyze",
        files={"file": file_tuple},
        data={"location_hint": "Synthetic Road 7"},
    )
    assert response.status_code == 200
    data = response.json()

    # Core state validations
    assert data["run_id"].startswith("run-")
    assert data["vision_result"]["pothole_detected"] is True
    assert data["location_result"]["resolution_method"] == "user_hint"
    assert data["location_result"]["estimated_road_name"] == "Synthetic Road 7"
    assert data["road_data"]["road"]["road_id"] == "RD-007"
    assert data["contract_data"]["best_contract_id"] == "CNT-007"
    assert data["officer_data"]["officer"]["officer_id"] == "OFF-008"

    # Complaint and Quality validations
    assert data["complaint_record"]["complaint_id"].startswith("DEMO-COMPLAINT-")
    assert data["complaint_record"]["verification_status"] == "VERIFIED"
    assert data["final_quality_score"] > 80.0
    assert "SYNTHETIC DEMO RECORD" in data["disclaimer"]


def test_analyze_unresolved_location(client):
    """POST /api/analyze without location hint flags human review."""
    file_tuple = make_dummy_image("unknown_road.jpg", "image/jpeg")
    response = client.post(
        "/api/analyze",
        files={"file": file_tuple},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["location_result"]["resolution_method"] == "unknown"
    assert data["requires_human_review"] is True
    assert data["complaint_record"]["verification_status"] == "REQUIRES_HUMAN_REVIEW"
    assert len(data["evidence_conflicts"]) > 0


def test_analyze_missing_file(client):
    """POST /api/analyze without a file returns 422 validation error."""
    response = client.post(
        "/api/analyze",
        data={"location_hint": "Synthetic Road 1"},
    )
    assert response.status_code == 422


def test_analyze_unsupported_file_type(client):
    """POST /api/analyze with non-image file returns 400 Bad Request."""
    dummy_pdf = ("document.pdf", io.BytesIO(b"%PDF-1.4 dummy content"), "application/pdf")
    response = client.post(
        "/api/analyze",
        files={"file": dummy_pdf},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_analyze_empty_file(client):
    """POST /api/analyze with 0-byte file returns 400 Bad Request."""
    empty_file = ("empty.jpg", io.BytesIO(b""), "image/jpeg")
    response = client.post(
        "/api/analyze",
        files={"file": empty_file},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# 3. SERVER-SENT EVENTS STREAMING (POST /api/analyze/stream)
# ══════════════════════════════════════════════════════════════════════════════

def test_stream_analysis_events(client):
    """POST /api/analyze/stream streams structured SSE events in correct sequence."""
    file_tuple = make_dummy_image("stream_test.jpg", "image/jpeg")
    response = client.post(
        "/api/analyze/stream",
        files={"file": file_tuple},
        data={"location_hint": "Synthetic Road 4"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    # Parse SSE events from response text
    events = []
    for line in body.split("\n"):
        if line.startswith("event: "):
            events.append(line.replace("event: ", "").strip())

    # Verify expected event flow
    assert "workflow_started" in events
    assert "vision_completed" in events
    assert "location_completed" in events
    assert "evidence_found" in events
    assert "verification_completed" in events
    assert "complaint_generated" in events
    assert "quality_evaluated" in events
    assert "workflow_completed" in events


def test_stream_unresolved_location_includes_human_review_event(client):
    """POST /api/analyze/stream emits human_review_required event when location is unknown."""
    file_tuple = make_dummy_image("unresolved_stream.jpg", "image/jpeg")
    response = client.post(
        "/api/analyze/stream",
        files={"file": file_tuple},
    )
    assert response.status_code == 200

    body = response.text
    events = [line.replace("event: ", "").strip() for line in body.split("\n") if line.startswith("event: ")]

    assert "workflow_started" in events
    assert "human_review_required" in events
    assert "workflow_completed" in events


def test_stream_unsupported_file_emits_error_event(client):
    """POST /api/analyze/stream with invalid file yields workflow_error event."""
    bad_file = ("bad.txt", io.BytesIO(b"plain text"), "text/plain")
    response = client.post(
        "/api/analyze/stream",
        files={"file": bad_file},
    )
    assert response.status_code == 200
    body = response.text
    assert "event: workflow_error" in body
    assert "Unsupported file format" in body
