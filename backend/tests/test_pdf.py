"""
Unit and integration tests for ReportLab PDF complaint generator and /api/complaints/pdf endpoint.
"""
import io
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.main import app
from backend.utils.complaint_pdf import generate_complaint_pdf


@pytest.fixture
def sample_analysis_state():
    return {
        "run_id": "RUN-TEST-007",
        "vision_result": {
            "pothole_detected": True,
            "severity": "moderate",
            "confidence": 0.95,
            "visual_evidence": "Visible pothole in center lane with asphalt degradation.",
        },
        "location_result": {
            "estimated_road_name": "Synthetic Road 7",
            "road_id": "RD-007",
            "district": "Synthetic North District",
            "coordinates": {"latitude": 12.9716, "longitude": 77.5946},
            "resolution_method": "user_hint",
            "confidence": 0.95,
        },
        "road_data": {
            "road": {
                "road_id": "RD-007",
                "road_name": "Synthetic Road 7",
                "district": "Synthetic North District",
            },
            "project": {
                "project_id": "PRJ-007",
                "project_status": "Completed",
                "notes": "Surfacing works completed under tender PRJ-007.",
            },
        },
        "contract_data": {
            "best_contract_id": "CNT-007",
            "best_tender_reference": "TND-007",
            "contractor_name": "Synthetic Infrastructure Corp",
            "match_status": "exact_match",
            "rag_confidence": 0.92,
            "evidence_summary": "Retrieved tender documentation for road surfacing.",
        },
        "officer_data": {
            "officer_name": "Jane Doe 8",
            "officer_id": "OFF-008",
            "department": "Public Works Division",
            "role": "Superintending Engineer",
            "jurisdiction": "Synthetic North District",
        },
        "evidence_conflicts": [],
        "verification_confidence": 0.94,
        "requires_human_review": False,
        "complaint_record": {
            "complaint_id": "DEMO-COMPLAINT-A0202DBA",
            "generated_at": "2026-08-23T10:00:00.000Z",
            "issue_description": "Pothole detected on Synthetic Road 7 with moderate severity.",
            "location_summary": "Synthetic Road 7, Synthetic North District",
            "contractor": {
                "contractor_name": "Synthetic Infrastructure Corp",
                "contract_id": "CNT-007",
            },
            "responsible_officer": {
                "officer_name": "Jane Doe 8",
                "officer_id": "OFF-008",
                "department": "Public Works Division",
                "role": "Superintending Engineer",
                "jurisdiction": "Synthetic North District",
            },
            "verification_status": "VERIFIED",
            "verification_confidence": 0.94,
            "evidence_conflicts": [],
            "requires_human_review": False,
            "disclaimer": "SYNTHETIC DEMO RECORD — All data is fictional.",
        },
        "final_quality_score": 94.0,
        "quality_explanation": "All 8 deterministic evidence components verified successfully.",
    }


def test_generate_pdf_produces_valid_pdf_bytes(sample_analysis_state):
    """Verifies that generate_complaint_pdf produces valid non-empty PDF bytes starting with %PDF."""
    pdf_bytes = generate_complaint_pdf(sample_analysis_state)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_pdf_with_minimal_or_empty_state():
    """Verifies that missing optional fields do not crash PDF generation."""
    minimal_state = {"run_id": "RUN-EMPTY"}
    pdf_bytes = generate_complaint_pdf(minimal_state)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_pdf_with_long_descriptions():
    """Verifies that long text descriptions wrap properly without errors."""
    long_state = {
        "complaint_record": {
            "complaint_id": "DEMO-LONG-TEXT",
            "issue_description": "A " * 500,  # 1000 character description
            "generated_at": "2026-08-23T12:00:00Z",
            "verification_status": "REQUIRES_REVIEW",
            "evidence_conflicts": ["Conflict 1: " + "X" * 100, "Conflict 2: " + "Y" * 100],
            "requires_human_review": True,
        },
        "final_quality_score": 45.0,
        "quality_explanation": "Quality score lowered due to multiple evidence gaps. " * 10,
    }
    pdf_bytes = generate_complaint_pdf(long_state)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")


def test_api_download_complaint_pdf_endpoint(sample_analysis_state):
    """Verifies POST /api/complaints/pdf returns 200 with application/pdf and proper header."""
    client = TestClient(app)
    response = client.post("/api/complaints/pdf", json=sample_analysis_state)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment; filename=\"RoadWatch_Complaint_DEMO-COMPLAINT-A0202DBA.pdf\"" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF-")
