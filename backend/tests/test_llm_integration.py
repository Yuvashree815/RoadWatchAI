"""
Unit and integration tests for Google Gemini LLM configuration, local image base64 conversion,
offline execution, and error handling.
"""
import io
import os
import sys
import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.llm import get_llm, is_llm_configured, get_gemini_model
from backend.agents.vision_agent import ensure_image_data_url, run_vision_agent
from backend.api.service import AnalysisService, default_analysis_service
from backend.database.repository import DatabaseRepository
from backend.graph.workflow import build_roadwatch_graph
from backend.graph.state import GraphState
from backend.main import app


# ── 1. Gemini LLM Initialization & Configuration Tests ───────────────────────

def test_get_llm_returns_none_without_api_key(monkeypatch):
    """When GEMINI_API_KEY and GOOGLE_API_KEY are not set, get_llm() returns None."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    llm = get_llm()
    assert llm is None
    assert is_llm_configured() is False


def test_get_llm_initializes_with_gemini_api_key(monkeypatch):
    """When GEMINI_API_KEY is set, get_llm() returns a ChatGoogleGenerativeAI instance."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key-for-testing")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")

    llm = get_llm()
    assert llm is not None
    assert llm.model == "gemini-3.6-flash"
    assert is_llm_configured() is True


def test_get_llm_default_model_fallback(monkeypatch):
    """Default model is gemini-3.6-flash if GEMINI_MODEL is unset."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key-for-testing")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    llm = get_llm()
    assert llm is not None
    assert llm.model == "gemini-3.6-flash"


# ── 2. Local Image Base64 Data URL Tests ──────────────────────────────────────

def test_ensure_image_data_url_from_local_file():
    """Converts a local file to a valid data:image/jpeg;base64,... URL."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
        temp_path = f.name

    try:
        data_url = ensure_image_data_url(temp_path)
        assert data_url.startswith("data:image/jpeg;base64,")
        # Ensure base64 payload is non-empty
        assert len(data_url.split("base64,")[1]) > 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_ensure_image_data_url_preserves_remote_and_data_urls():
    """Preserves data URLs and http/https URLs as-is."""
    existing_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    assert ensure_image_data_url(existing_data_url) == existing_data_url

    http_url = "https://example.com/pothole.jpg"
    assert ensure_image_data_url(http_url) == http_url


def test_ensure_image_data_url_missing_file_raises():
    """Raises FileNotFoundError for a missing local file path."""
    with pytest.raises(FileNotFoundError):
        ensure_image_data_url("C:\\nonexistent\\path\\to\\image.jpg")


# ── 3. Vision Agent with Local Image & Mock LLM ───────────────────────────────

def test_vision_agent_raises_clear_error_without_llm():
    """Raises a clear RuntimeError when llm is None."""
    with pytest.raises(RuntimeError) as exc_info:
        run_vision_agent("https://example.com/test.jpg", llm=None)
    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_vision_agent_passes_base64_url_to_llm():
    """Vision Agent passes the converted base64 data URL to the multimodal model."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01")
        temp_path = f.name

    try:
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "pothole_detected": True,
            "severity": "moderate",
            "confidence": 0.85,
            "visual_evidence": "Visible pothole in center lane.",
        })
        mock_llm.invoke.return_value = mock_response

        result = run_vision_agent(temp_path, llm=mock_llm)

        assert result["pothole_detected"] is True
        assert result["severity"] == "moderate"

        # Verify the message passed to mock_llm contains the base64 data URL
        call_args = mock_llm.invoke.call_args[0][0]
        human_msg = call_args[1]
        image_content = human_msg.content[1]
        assert image_content["type"] == "image_url"
        assert image_content["image_url"]["url"].startswith("data:image/png;base64,")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── 4. Offline Workflow Preservation Tests ────────────────────────────────────

def test_workflow_offline_with_prepopulated_vision_does_not_call_llm():
    """If vision_result is pre-populated, graph never invokes LLM or requires API key."""
    # Graph built with llm=None
    app = build_roadwatch_graph(llm=None, db=DatabaseRepository())

    state: GraphState = {
        "run_id": "test-offline-123",
        "image_url": "dummy_path.jpg",
        "user_location_hint": "Synthetic Road 7",
        "exif_gps": None,
        "vision_result": {
            "pothole_detected": True,
            "severity": "low",
            "confidence": 0.90,
            "visual_evidence": "Pre-computed inspection result.",
        },
        "location_result": None,
        "road_data": None,
        "contract_data": None,
        "officer_data": None,
        "evidence_conflicts": None,
        "verification_confidence": None,
        "requires_human_review": None,
        "human_feedback": None,
        "complaint_record": None,
        "final_quality_score": None,
        "quality_explanation": None,
    }

    final_state = app.invoke(state)
    assert final_state["vision_result"]["severity"] == "low"
    assert final_state["road_data"]["road"]["road_id"] == "RD-007"
    assert final_state["complaint_record"]["complaint_id"].startswith("DEMO-COMPLAINT-")


# ── 5. API Error Handling for Missing API Key ─────────────────────────────────

def test_api_analyze_missing_api_key_returns_clear_error(monkeypatch):
    """POST /api/analyze returns HTTP 400 when GEMINI_API_KEY is not set."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    service = AnalysisService(llm=None, db=DatabaseRepository())
    client = TestClient(app)

    # Patch default_analysis_service._injected_llm to None
    with patch.object(default_analysis_service, "_injected_llm", None):
        with patch("backend.api.service.get_llm", return_value=None):
            file_tuple = ("test.jpg", io.BytesIO(b"\xFF\xD8\xFF\xE0JFIF"), "image/jpeg")
            response = client.post(
                "/api/analyze",
                files={"file": file_tuple},
                data={"location_hint": "Synthetic Road 1"},
            )
            assert response.status_code == 400
            assert "GEMINI_API_KEY is not configured" in response.json()["detail"]


def test_api_stream_missing_api_key_yields_workflow_error(monkeypatch):
    """POST /api/analyze/stream emits workflow_error when GEMINI_API_KEY is not set."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    client = TestClient(app)

    with patch.object(default_analysis_service, "_injected_llm", None):
        with patch("backend.api.service.get_llm", return_value=None):
            file_tuple = ("test.jpg", io.BytesIO(b"\xFF\xD8\xFF\xE0JFIF"), "image/jpeg")
            response = client.post(
                "/api/analyze/stream",
                files={"file": file_tuple},
                data={"location_hint": "Synthetic Road 1"},
            )
            assert response.status_code == 200
            body = response.text
            assert "event: workflow_error" in body
            assert "GEMINI_API_KEY is not configured" in body
