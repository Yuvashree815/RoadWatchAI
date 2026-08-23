"""
Tests for Milestone 5 agents:
  - Verification Agent
  - Complaint Agent
  - Quality Evaluation Agent
  - Updated LangGraph workflow (end-to-end)

All tests run offline using mock LLM and synthetic CSV data.
No external API keys required.
"""
import json
import os
import sys
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.agents.verification_agent import run_verification_agent
from backend.agents.complaint_agent import run_complaint_agent
from backend.agents.quality_agent import run_quality_evaluation_agent
from backend.database.repository import DatabaseRepository
from backend.rag.vector_store import VectorStoreManager
from backend.rag.keyword_search import KeywordSearchManager
from backend.rag.hybrid_search import HybridSearcher
from backend.graph.workflow import build_roadwatch_graph
from backend.graph.state import GraphState


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db_repo():
    return DatabaseRepository()


@pytest.fixture(scope="module")
def hybrid_searcher():
    vm = VectorStoreManager(persist_directory="./chroma_test_db")
    vm.ingest_documents()
    km = KeywordSearchManager()
    return HybridSearcher(vm, km)


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.invoke.return_value.content = json.dumps({
        "pothole_detected": True,
        "severity": "moderate",
        "confidence": 0.85,
        "visual_evidence": "Moderate pothole in right lane.",
    })
    return mock


# ── Complete evidence fixture ─────────────────────────────────────────────────

@pytest.fixture
def full_evidence():
    return {
        "vision_result": {
            "pothole_detected": True,
            "severity": "moderate",
            "confidence": 0.85,
            "visual_evidence": "Moderate pothole in right lane.",
        },
        "location_result": {
            "resolution_method": "demo_mapping",
            "latitude": 34.0401,
            "longitude": -117.9599,
            "estimated_road_name": "Synthetic Road 4",
            "location_hint_used": None,
            "confidence": 0.9,
            "demo_case_id": "DEMO-004",
            "notes": "[DEMO] Demo mapping used.",
        },
        "road_data": {
            "road": {"road_id": "RD-004", "road_name": "Synthetic Road 4",
                     "district": "North District", "area": "Sector A"},
            "project": {"project_id": "PRJ-004", "contract_id": "CNT-004",
                        "contractor_id": "CON-005", "officer_id": "OFF-005",
                        "maintenance_type": "Pothole Repair", "status": "Active"},
            "match_method": "demo_case_mapping",
            "confidence": 0.9,
        },
        "contract_data": {
            "best_contract_id": "CNT-004",
            "best_tender_reference": "TN-2026-004",
            "contract_record": {
                "contract_id": "CNT-004",
                "tender_reference": "TN-2026-004",
                "title": "Maintenance of Synthetic Road 4",
                "start_date": "2026-01-01",
                "end_date": "2026-12-31",
                "contract_value": 500000.0,
                "contractor_id": "CON-005",
            },
            "contractor_record": {
                "contractor_id": "CON-005",
                "contractor_name": "Demo Contractor 5",
                "contact_email": "con5@demo.com",
                "contact_phone": "555-0005",
                "rating": 3.8,
            },
            "rag_confidence": 0.9,
            "structured_match": True,
            "retrieved_chunks": [],
            "notes": "[DEMO] Contract found.",
        },
        "officer_data": {
            "officer": {
                "officer_id": "OFF-005",
                "officer_name": "Demo Officer 5",
                "department": "Department of Synthetic Works",
                "role": "Road Inspector",
                "jurisdiction": "North District",
            },
            "match_method": "project_direct_link",
            "confidence": 1.0,
            "notes": "[DEMO] Officer found via project link.",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestVerificationAgent:

    def test_full_evidence_verified(self, full_evidence):
        result = run_verification_agent(
            vision_result=full_evidence["vision_result"],
            location_result=full_evidence["location_result"],
            road_data=full_evidence["road_data"],
            contract_data=full_evidence["contract_data"],
            officer_data=full_evidence["officer_data"],
        )
        assert result["verified"] is True
        assert result["conflicts"] == []
        assert result["missing_evidence"] == []
        assert result["requires_human_review"] is False
        assert result["verification_confidence"] > 0.8
        assert "[DEMO]" in result["notes"]

    def test_missing_vision_flagged(self, full_evidence):
        result = run_verification_agent(
            vision_result=None,
            location_result=full_evidence["location_result"],
            road_data=full_evidence["road_data"],
            contract_data=full_evidence["contract_data"],
            officer_data=full_evidence["officer_data"],
        )
        assert result["verified"] is False
        assert any("vision_result" in m for m in result["missing_evidence"])

    def test_no_pothole_creates_conflict(self, full_evidence):
        no_pothole_vision = {**full_evidence["vision_result"], "pothole_detected": False}
        result = run_verification_agent(
            vision_result=no_pothole_vision,
            location_result=full_evidence["location_result"],
            road_data=full_evidence["road_data"],
            contract_data=full_evidence["contract_data"],
            officer_data=full_evidence["officer_data"],
        )
        assert any("did not detect a pothole" in c for c in result["conflicts"])
        assert result["requires_human_review"] is True

    def test_unresolved_location_conflict(self, full_evidence):
        bad_loc = {**full_evidence["location_result"],
                   "resolution_method": "unknown", "confidence": 0.0}
        result = run_verification_agent(
            vision_result=full_evidence["vision_result"],
            location_result=bad_loc,
            road_data=full_evidence["road_data"],
            contract_data=full_evidence["contract_data"],
            officer_data=full_evidence["officer_data"],
        )
        assert any("location" in c.lower() for c in result["conflicts"])
        assert result["requires_human_review"] is True

    def test_contract_mismatch_conflict(self, full_evidence):
        mismatched_contract = {**full_evidence["contract_data"],
                               "best_contract_id": "CNT-999"}
        result = run_verification_agent(
            vision_result=full_evidence["vision_result"],
            location_result=full_evidence["location_result"],
            road_data=full_evidence["road_data"],
            contract_data=mismatched_contract,
            officer_data=full_evidence["officer_data"],
        )
        assert any("mismatch" in c.lower() for c in result["conflicts"])

    def test_low_vision_confidence_warns(self, full_evidence):
        low_conf_vision = {**full_evidence["vision_result"], "confidence": 0.3}
        result = run_verification_agent(
            vision_result=low_conf_vision,
            location_result=full_evidence["location_result"],
            road_data=full_evidence["road_data"],
            contract_data=full_evidence["contract_data"],
            officer_data=full_evidence["officer_data"],
        )
        assert any("low" in w.lower() or "confidence" in w.lower()
                   for w in result["warnings"])

    def test_existing_conflicts_carried_forward(self, full_evidence):
        pre_conflict = ["Pre-existing conflict from aggregation step."]
        result = run_verification_agent(
            vision_result=full_evidence["vision_result"],
            location_result=full_evidence["location_result"],
            road_data=full_evidence["road_data"],
            contract_data=full_evidence["contract_data"],
            officer_data=full_evidence["officer_data"],
            existing_conflicts=pre_conflict,
        )
        assert "Pre-existing conflict from aggregation step." in result["conflicts"]
        assert result["requires_human_review"] is True

    def test_all_result_keys_present(self, full_evidence):
        result = run_verification_agent(
            vision_result=full_evidence["vision_result"],
            location_result=full_evidence["location_result"],
            road_data=full_evidence["road_data"],
            contract_data=full_evidence["contract_data"],
            officer_data=full_evidence["officer_data"],
        )
        for key in ("verified", "conflicts", "warnings", "missing_evidence",
                    "requires_human_review", "verification_confidence", "notes"):
            assert key in result


# ══════════════════════════════════════════════════════════════════════════════
# COMPLAINT AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestComplaintAgent:

    def _make_verification(self, verified=True, conflicts=None, warnings=None, missing=None):
        return {
            "verified": verified,
            "conflicts": conflicts or [],
            "warnings": warnings or [],
            "missing_evidence": missing or [],
            "requires_human_review": not verified,
            "verification_confidence": 0.95 if verified else 0.4,
            "notes": "[DEMO] Test verification.",
        }

    def test_verified_complaint_has_all_keys(self, full_evidence):
        verification = self._make_verification(verified=True)
        result = run_complaint_agent(
            run_id="test-run-001",
            **full_evidence,
            verification_result=verification,
        )
        required = {
            "complaint_id", "run_id", "generated_at", "disclaimer",
            "issue_description", "severity", "pothole_detected", "vision_confidence",
            "location_summary", "location_detail", "road", "maintenance_project",
            "contract", "contractor", "responsible_officer",
            "verification_status", "verification_confidence",
            "evidence_conflicts", "evidence_warnings", "missing_evidence",
        }
        assert required.issubset(result.keys())

    def test_complaint_id_prefixed_correctly(self, full_evidence):
        verification = self._make_verification()
        result = run_complaint_agent(
            run_id="test-run-002",
            **full_evidence,
            verification_result=verification,
        )
        assert result["complaint_id"].startswith("DEMO-COMPLAINT-")

    def test_verified_status_set_correctly(self, full_evidence):
        result = run_complaint_agent(
            run_id="test-run-003",
            **full_evidence,
            verification_result=self._make_verification(verified=True),
        )
        assert result["verification_status"] == "VERIFIED"

    def test_unverified_status_requires_review(self, full_evidence):
        result = run_complaint_agent(
            run_id="test-run-004",
            **full_evidence,
            verification_result=self._make_verification(
                verified=False, conflicts=["Some conflict."]),
        )
        assert result["verification_status"] == "REQUIRES_HUMAN_REVIEW"

    def test_disclaimer_present(self, full_evidence):
        result = run_complaint_agent(
            run_id="test-run-005",
            **full_evidence,
            verification_result=self._make_verification(),
        )
        assert "SYNTHETIC DEMO RECORD" in result["disclaimer"]

    def test_complaint_with_no_officer(self, full_evidence):
        no_officer_data = {**full_evidence["officer_data"], "officer": None}
        result = run_complaint_agent(
            run_id="test-run-006",
            vision_result=full_evidence["vision_result"],
            location_result=full_evidence["location_result"],
            road_data=full_evidence["road_data"],
            contract_data=full_evidence["contract_data"],
            officer_data=no_officer_data,
            verification_result=self._make_verification(),
        )
        assert result["responsible_officer"] is None

    def test_complaint_no_invented_data(self, full_evidence):
        """Verify no 'UNKNOWN' or placeholder data is silently injected."""
        result = run_complaint_agent(
            run_id="test-run-007",
            **full_evidence,
            verification_result=self._make_verification(),
        )
        # Road from state should exactly match
        assert result["road"]["road_id"] == "RD-004"
        assert result["contract"]["contract_id"] == "CNT-004"
        assert result["responsible_officer"]["officer_id"] == "OFF-005"


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY EVALUATION AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestQualityEvaluationAgent:

    def test_full_evidence_scores_high(self, full_evidence):
        vr = {"conflicts": [], "verification_confidence": 0.95}
        result = run_quality_evaluation_agent(
            **full_evidence,
            verification_result=vr,
        )
        assert result["final_quality_score"] >= 80.0
        assert result["quality_level"] in ("EXCELLENT", "GOOD")
        assert result["quality_flags"] == []

    def test_missing_all_evidence_scores_zero(self):
        vr = {"conflicts": ["No evidence at all."], "verification_confidence": 0.0}
        result = run_quality_evaluation_agent(
            vision_result=None,
            location_result=None,
            road_data=None,
            contract_data=None,
            officer_data=None,
            verification_result=vr,
        )
        assert result["final_quality_score"] == 0.0
        assert result["quality_level"] == "POOR"
        assert len(result["quality_flags"]) > 0

    def test_component_scores_sum_to_final(self, full_evidence):
        vr = {"conflicts": [], "verification_confidence": 0.95}
        result = run_quality_evaluation_agent(**full_evidence, verification_result=vr)
        computed_sum = round(sum(result["component_scores"].values()), 1)
        assert computed_sum == result["final_quality_score"]

    def test_all_result_keys_present(self, full_evidence):
        vr = {"conflicts": [], "verification_confidence": 0.9}
        result = run_quality_evaluation_agent(**full_evidence, verification_result=vr)
        for key in ("final_quality_score", "quality_explanation",
                    "component_scores", "quality_flags", "quality_level"):
            assert key in result

    def test_conflict_reduces_score(self, full_evidence):
        vr_clean = {"conflicts": [], "verification_confidence": 0.95}
        vr_dirty = {"conflicts": ["Conflict A."], "verification_confidence": 0.6}
        score_clean = run_quality_evaluation_agent(
            **full_evidence, verification_result=vr_clean)["final_quality_score"]
        score_dirty = run_quality_evaluation_agent(
            **full_evidence, verification_result=vr_dirty)["final_quality_score"]
        assert score_dirty < score_clean

    def test_quality_explanation_contains_demo(self, full_evidence):
        vr = {"conflicts": [], "verification_confidence": 0.95}
        result = run_quality_evaluation_agent(**full_evidence, verification_result=vr)
        assert "[DEMO]" in result["quality_explanation"]


# ══════════════════════════════════════════════════════════════════════════════
# END-TO-END WORKFLOW TESTS (Milestone 5 Extension)
# ══════════════════════════════════════════════════════════════════════════════

class TestMilestone5Workflow:

    def _base_state(self, run_id, hint=None, demo_case_id=None,
                    exif_gps=None, vision_result=None, location_result=None) -> GraphState:
        return {
            "run_id": run_id,
            "image_url": "https://example.com/test.jpg",
            "user_location_hint": hint,
            "exif_gps": exif_gps,
            "vision_result": vision_result,
            "location_result": location_result,
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

    def test_complete_pipeline_with_demo_case(self, mock_llm, db_repo, hybrid_searcher):
        """Full end-to-end with demo_case_id — expects verified, high-quality record."""
        app = build_roadwatch_graph(llm=mock_llm, db=db_repo, hybrid_searcher=hybrid_searcher)
        state = self._base_state(
            "e2e-001",
            location_result={"demo_case_id": "DEMO-004", "resolution_method": "demo_mapping"},
        )
        result = app.invoke(state)

        # Complaint record generated
        assert result["complaint_record"] is not None
        assert result["complaint_record"]["complaint_id"].startswith("DEMO-COMPLAINT-")
        assert result["complaint_record"]["road"]["road_id"] == "RD-004"
        assert result["complaint_record"]["verification_status"] == "VERIFIED"

        # Quality score computed
        assert result["final_quality_score"] is not None
        assert result["final_quality_score"] > 0.0
        assert "[DEMO]" in result["quality_explanation"]

    def test_workflow_compilation_includes_new_nodes(self, db_repo, hybrid_searcher):
        """Verify graph now contains verification, complaint, quality_evaluation nodes."""
        app = build_roadwatch_graph(llm=None, db=db_repo, hybrid_searcher=hybrid_searcher)
        nodes = app.get_graph().nodes
        assert "verification" in nodes
        assert "complaint" in nodes
        assert "quality_evaluation" in nodes

    def test_unresolved_location_still_generates_complaint(self, mock_llm, db_repo, hybrid_searcher):
        """Unresolved location should still produce a complaint record flagged for review."""
        app = build_roadwatch_graph(llm=mock_llm, db=db_repo, hybrid_searcher=hybrid_searcher)
        state = self._base_state("e2e-unresolved")
        result = app.invoke(state)

        assert result["complaint_record"] is not None
        assert result["complaint_record"]["verification_status"] == "REQUIRES_HUMAN_REVIEW"
        assert result["requires_human_review"] is True
        assert result["final_quality_score"] < 50.0

    def test_ground_truth_full_pipeline(self, mock_llm, db_repo, hybrid_searcher):
        """Run all ground truth cases through the complete Milestone 5 pipeline."""
        app = build_roadwatch_graph(llm=mock_llm, db=db_repo, hybrid_searcher=hybrid_searcher)
        gt_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/ground_truth.json"))
        with open(gt_path) as f:
            ground_truth = json.load(f)

        for case in ground_truth:
            state = self._base_state(
                f"gt-m5-{case['demo_case_id']}",
                location_result={
                    "demo_case_id": case["demo_case_id"],
                    "resolution_method": "demo_mapping",
                },
            )
            result = app.invoke(state)

            # Road, contract, officer correct
            assert result["road_data"]["road"]["road_id"] == case["expected_road_id"]
            assert result["contract_data"]["best_contract_id"] == case["expected_contract_id"]
            assert result["officer_data"]["officer"]["officer_id"] == case["expected_officer_id"]

            # Complaint generated
            assert result["complaint_record"]["complaint_id"].startswith("DEMO-COMPLAINT-")
            assert result["complaint_record"]["verification_status"] == "VERIFIED"

            # Quality score positive
            assert result["final_quality_score"] > 0.0
