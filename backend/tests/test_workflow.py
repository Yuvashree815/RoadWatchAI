"""
Unit and integration tests for Milestone 4: LangGraph Workflow Orchestration.

Tests verify:
  - Graph compilation and structure
  - Full execution pipeline with mock LLM and real local RAG / DB
  - Parallel road research and contract research execution
  - Evidence aggregation and conflict detection
  - Conditional routing on unresolved location
  - Ground truth multi-case validation across the workflow
"""
import os
import sys
import json
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.database.repository import DatabaseRepository
from backend.rag.vector_store import VectorStoreManager
from backend.rag.keyword_search import KeywordSearchManager
from backend.rag.hybrid_search import HybridSearcher
from backend.graph.workflow import build_roadwatch_graph
from backend.graph.state import GraphState


# ── Shared Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db_repo():
    return DatabaseRepository()


@pytest.fixture(scope="module")
def vector_manager():
    manager = VectorStoreManager(persist_directory="./chroma_test_db")
    manager.ingest_documents()
    return manager


@pytest.fixture(scope="module")
def keyword_manager():
    return KeywordSearchManager()


@pytest.fixture(scope="module")
def hybrid_searcher(vector_manager, keyword_manager):
    return HybridSearcher(vector_manager, keyword_manager)


@pytest.fixture
def mock_vision_llm():
    """Returns a mock LLM that responds with valid vision detection output."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "pothole_detected": True,
        "severity": "moderate",
        "confidence": 0.88,
        "visual_evidence": "Visible asphalt pothole in right lane, approx 40cm across.",
    })
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ══════════════════════════════════════════════════════════════════════════════
# WORKFLOW TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_workflow_compilation(db_repo, hybrid_searcher):
    """Verify that build_roadwatch_graph compiles cleanly."""
    app = build_roadwatch_graph(llm=None, db=db_repo, hybrid_searcher=hybrid_searcher)
    assert app is not None
    # Check that nodes are registered in the graph
    nodes = app.get_graph().nodes
    assert "vision" in nodes
    assert "location" in nodes
    assert "road_research" in nodes
    assert "contract_research" in nodes
    assert "aggregate_evidence" in nodes
    assert "unresolved_evidence" in nodes
    assert "officer_research" in nodes


def test_workflow_execution_with_user_hint(mock_vision_llm, db_repo, hybrid_searcher):
    """Test full workflow execution when user provides a location hint."""
    app = build_roadwatch_graph(llm=mock_vision_llm, db=db_repo, hybrid_searcher=hybrid_searcher)

    initial_state: GraphState = {
        "run_id": "test-run-001",
        "image_url": "https://example.com/pothole_007.jpg",
        "user_location_hint": "Synthetic Road 7",
        "exif_gps": None,
        "vision_result": None,
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

    final_state = app.invoke(initial_state)

    # 1. Vision Agent output
    assert final_state["vision_result"] is not None
    assert final_state["vision_result"]["pothole_detected"] is True
    assert final_state["vision_result"]["severity"] == "moderate"

    # 2. Location Agent output
    assert final_state["location_result"] is not None
    assert final_state["location_result"]["resolution_method"] == "user_hint"
    assert final_state["location_result"]["estimated_road_name"] == "Synthetic Road 7"

    # 3. Road Research output
    assert final_state["road_data"] is not None
    assert final_state["road_data"]["road"]["road_id"] == "RD-007"
    assert final_state["road_data"]["project"]["contract_id"] == "CNT-007"

    # 4. Contract Research output
    assert final_state["contract_data"] is not None
    assert final_state["contract_data"]["best_contract_id"] == "CNT-007"
    assert final_state["contract_data"]["structured_match"] is True

    # 5. Officer Research output
    assert final_state["officer_data"] is not None
    assert final_state["officer_data"]["officer"]["officer_id"] == "OFF-008"
    assert final_state["officer_data"]["match_method"] == "project_direct_link"

    # 6. Evidence Aggregation output
    assert final_state["evidence_conflicts"] == []
    assert final_state["requires_human_review"] is False


def test_workflow_execution_with_exif_gps(mock_vision_llm, db_repo, hybrid_searcher):
    """Test full workflow execution when EXIF GPS is available."""
    app = build_roadwatch_graph(llm=mock_vision_llm, db=db_repo, hybrid_searcher=hybrid_searcher)

    # Coordinates near RD-001 (34.01, -117.99)
    initial_state: GraphState = {
        "run_id": "test-run-002",
        "image_url": "https://example.com/pothole_001.jpg",
        "user_location_hint": None,
        "exif_gps": {"latitude": 34.0102, "longitude": -117.9898},
        "vision_result": None,
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

    final_state = app.invoke(initial_state)

    assert final_state["location_result"]["resolution_method"] == "exif_gps"
    assert final_state["road_data"]["road"]["road_id"] == "RD-001"
    assert final_state["contract_data"]["best_contract_id"] == "CNT-001"
    assert final_state["officer_data"]["officer"]["officer_id"] == "OFF-002"


def test_workflow_conditional_routing_unresolved_location(mock_vision_llm, db_repo, hybrid_searcher):
    """Test conditional routing to unresolved_evidence when location cannot be determined."""
    app = build_roadwatch_graph(llm=mock_vision_llm, db=db_repo, hybrid_searcher=hybrid_searcher)

    initial_state: GraphState = {
        "run_id": "test-run-003",
        "image_url": "https://example.com/pothole_unknown.jpg",
        "user_location_hint": None,
        "exif_gps": None,
        "vision_result": None,
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

    final_state = app.invoke(initial_state)

    # 1. Location was unknown
    assert final_state["location_result"]["resolution_method"] == "unknown"
    assert final_state["location_result"]["confidence"] == 0.0

    # 2. Routed to unresolved_evidence_node
    assert final_state["road_data"]["road"] is None
    assert final_state["contract_data"]["best_contract_id"] is None
    assert final_state["requires_human_review"] is True
    assert len(final_state["evidence_conflicts"]) > 0
    assert "Unresolved road location" in final_state["evidence_conflicts"][0]

    # 3. Officer could not be identified
    assert final_state["officer_data"]["officer"] is None


def test_workflow_offline_with_prepopulated_vision(db_repo, hybrid_searcher):
    """Verify workflow works completely offline without LLM when vision_result is pre-populated."""
    # Note: llm=None passed to graph
    app = build_roadwatch_graph(llm=None, db=db_repo, hybrid_searcher=hybrid_searcher)

    initial_state: GraphState = {
        "run_id": "test-run-offline",
        "image_url": "https://example.com/offline.jpg",
        "user_location_hint": "RD-003",
        "exif_gps": None,
        "vision_result": {
            "pothole_detected": True,
            "severity": "low",
            "confidence": 0.95,
            "visual_evidence": "Pre-computed offline inspection data.",
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

    final_state = app.invoke(initial_state)
    assert final_state["vision_result"]["severity"] == "low"
    assert final_state["road_data"]["road"]["road_id"] == "RD-003"
    assert final_state["contract_data"]["best_contract_id"] == "CNT-003"


def test_workflow_on_ground_truth_cases(mock_vision_llm, db_repo, hybrid_searcher):
    """Run all ground truth cases through the LangGraph workflow."""
    app = build_roadwatch_graph(llm=mock_vision_llm, db=db_repo, hybrid_searcher=hybrid_searcher)

    gt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/ground_truth.json"))
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    for case in ground_truth:
        demo_case_id = case["demo_case_id"]
        expected_road_id = case["expected_road_id"]
        expected_contract_id = case["expected_contract_id"]
        expected_officer_id = case["expected_officer_id"]

        state: GraphState = {
            "run_id": f"gt-run-{demo_case_id}",
            "image_url": f"https://example.com/images/{demo_case_id}.jpg",
            "user_location_hint": None,
            "exif_gps": None,
            "vision_result": None,
            "location_result": {"demo_case_id": demo_case_id, "resolution_method": "demo_mapping"},
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

        result = app.invoke(state)

        assert result["road_data"]["road"]["road_id"] == expected_road_id
        assert result["contract_data"]["best_contract_id"] == expected_contract_id
        assert result["officer_data"]["officer"]["officer_id"] == expected_officer_id
        assert result["evidence_conflicts"] == []
