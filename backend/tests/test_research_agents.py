"""
Unit and integration tests for Milestone 3C Research Agents:
  - Road Research Agent (backend.agents.road_research_agent)
  - Contract / Tender Research Agent (backend.agents.contract_research_agent)
  - Officer Research Agent (backend.agents.officer_research_agent)

Uses local synthetic CSV data and mock / local RAG fixtures.
No external API keys required.
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

from backend.agents.road_research_agent import run_road_research_agent
from backend.agents.contract_research_agent import run_contract_research_agent
from backend.agents.officer_research_agent import run_officer_research_agent
from backend.graph.state import GraphState


# ── Shared fixtures ──────────────────────────────────────────────────────────

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
def real_hybrid_searcher(vector_manager, keyword_manager):
    return HybridSearcher(vector_manager, keyword_manager)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ROAD RESEARCH AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRoadResearchAgent:

    def test_direct_road_id_lookup(self, db_repo):
        res = run_road_research_agent(
            estimated_road_name="RD-001",
            location_result=None,
            db=db_repo,
        )
        assert res["road"] is not None
        assert res["road"]["road_id"] == "RD-001"
        assert res["road"]["road_name"] == "Synthetic Road 1"
        assert res["project"] is not None
        assert res["project"]["road_id"] == "RD-001"
        assert res["match_method"] == "road_id_direct"
        assert res["confidence"] == 1.0
        assert "[DEMO]" in res["notes"]

    def test_name_substring_lookup(self, db_repo):
        res = run_road_research_agent(
            estimated_road_name="Synthetic Road 7",
            location_result=None,
            db=db_repo,
        )
        assert res["road"] is not None
        assert res["road"]["road_id"] == "RD-007"
        assert res["project"] is not None
        assert res["match_method"] == "name_substring"
        assert res["confidence"] >= 0.85

    def test_demo_case_fallback_lookup(self, db_repo):
        location_result = {
            "resolution_method": "demo_mapping",
            "demo_case_id": "DEMO-004",
            "estimated_road_name": None,
        }
        res = run_road_research_agent(
            estimated_road_name=None,
            location_result=location_result,
            db=db_repo,
        )
        assert res["road"] is not None
        assert res["road"]["road_id"] == "RD-004"
        assert res["match_method"] == "demo_case_mapping"
        assert res["project"] is not None

    def test_nonexistent_road_lookup(self, db_repo):
        res = run_road_research_agent(
            estimated_road_name="NonExistent Road 999",
            location_result=None,
            db=db_repo,
        )
        assert res["road"] is None
        assert res["project"] is None
        assert res["confidence"] == 0.0
        assert "Could not identify road" in res["notes"]

    def test_all_expected_keys_present(self, db_repo):
        res = run_road_research_agent(
            estimated_road_name="RD-002",
            location_result=None,
            db=db_repo,
        )
        expected = {"road", "project", "match_method", "confidence", "notes"}
        assert expected.issubset(res.keys())


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONTRACT / TENDER RESEARCH AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestContractResearchAgent:

    def test_contract_research_with_mock_hybrid_searcher(self, db_repo):
        mock_searcher = MagicMock()
        mock_searcher.hybrid_search.return_value = [
            {
                "page_content": "Tender Reference: TN-2026-003 Contract ID: CNT-003 Maintenance of Synthetic Road 3",
                "metadata": {
                    "contract_id": "CNT-003",
                    "tender_reference": "TN-2026-003",
                    "road_name": "Synthetic Road 3",
                },
                "rrf_score": 0.032,
            }
        ]

        road_data = {
            "road": {"road_id": "RD-003", "road_name": "Synthetic Road 3"},
            "project": {"project_id": "PRJ-003", "contract_id": "CNT-003", "contractor_id": "CON-003"},
        }

        res = run_contract_research_agent(
            road_data=road_data,
            hybrid_searcher=mock_searcher,
            db=db_repo,
        )

        assert res["best_contract_id"] == "CNT-003"
        assert res["best_tender_reference"] == "TN-2026-003"
        assert res["structured_match"] is True
        assert res["contract_record"] is not None
        assert res["contract_record"]["contract_id"] == "CNT-003"
        assert res["contractor_record"] is not None
        assert res["rag_confidence"] > 0.0
        assert len(res["retrieved_chunks"]) == 1

    def test_contract_research_with_real_hybrid_searcher(self, db_repo, real_hybrid_searcher):
        road_data = {
            "road": {"road_id": "RD-007", "road_name": "Synthetic Road 7"},
            "project": {"project_id": "PRJ-007", "contract_id": "CNT-007", "contractor_id": "CON-007"},
        }

        res = run_contract_research_agent(
            road_data=road_data,
            hybrid_searcher=real_hybrid_searcher,
            db=db_repo,
            top_k=3,
        )

        assert res["best_contract_id"] == "CNT-007"
        assert res["structured_match"] is True
        assert res["contract_record"]["contract_id"] == "CNT-007"
        assert res["contractor_record"] is not None
        assert len(res["retrieved_chunks"]) > 0

    def test_contract_research_empty_road_data(self, db_repo):
        mock_searcher = MagicMock()
        mock_searcher.hybrid_search.return_value = []

        res = run_contract_research_agent(
            road_data=None,
            hybrid_searcher=mock_searcher,
            db=db_repo,
        )

        assert res["best_contract_id"] is None
        assert res["contract_record"] is None
        assert res["structured_match"] is False
        assert res["rag_confidence"] == 0.0
        assert "missing" in res["notes"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# 3. OFFICER RESEARCH AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestOfficerResearchAgent:

    def test_officer_resolved_from_project_direct_link(self, db_repo):
        road_data = {
            "road": {"road_id": "RD-001", "road_name": "Synthetic Road 1", "district": "North District"},
            "project": {"project_id": "PRJ-001", "contract_id": "CNT-001", "officer_id": "OFF-001"},
        }
        res = run_officer_research_agent(road_data=road_data, db=db_repo)

        assert res["officer"] is not None
        assert res["officer"]["officer_id"] == "OFF-001"
        assert res["match_method"] == "project_direct_link"
        assert res["confidence"] == 1.0
        assert "Department of Synthetic Works" in res["officer"]["department"]
        assert "[DEMO]" in res["notes"]

    def test_officer_resolved_via_jurisdiction_fallback(self, db_repo):
        road_data = {
            "road": {"road_id": "RD-999", "road_name": "Unknown Road", "district": "North District"},
            "project": None,
        }
        res = run_officer_research_agent(road_data=road_data, db=db_repo)

        assert res["officer"] is not None
        assert res["match_method"] == "jurisdiction_fallback"
        assert res["confidence"] == 0.65

    def test_officer_not_found(self, db_repo):
        road_data = {
            "road": {"road_id": "RD-999", "road_name": "Unknown Road", "district": "Atlantis Unknown District"},
            "project": None,
        }
        res = run_officer_research_agent(road_data=road_data, db=db_repo)

        assert res["officer"] is None
        assert res["match_method"] == "none"
        assert res["confidence"] == 0.0
        assert "could not be identified" in res["notes"]

    def test_all_expected_keys_present(self, db_repo):
        road_data = {
            "road": {"road_id": "RD-002", "road_name": "Synthetic Road 2", "district": "North District"},
            "project": {"project_id": "PRJ-002", "officer_id": "OFF-002"},
        }
        res = run_officer_research_agent(road_data=road_data, db=db_repo)
        expected = {"officer", "match_method", "confidence", "notes"}
        assert expected.issubset(res.keys())


# ══════════════════════════════════════════════════════════════════════════════
# 4. CHAIN INTEGRATION WITH GROUND TRUTH & GRAPHSTATE
# ══════════════════════════════════════════════════════════════════════════════

class TestResearchAgentsChain:

    def test_full_chain_on_ground_truth(self, db_repo, real_hybrid_searcher):
        gt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/ground_truth.json"))
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        for case in ground_truth[:4]:
            demo_case_id = case["demo_case_id"]
            expected_road_id = case["expected_road_id"]
            expected_contract_id = case["expected_contract_id"]
            expected_officer_id = case["expected_officer_id"]

            # Step 1: Road Research
            location_result = {"demo_case_id": demo_case_id, "estimated_road_name": None}
            road_data = run_road_research_agent(
                estimated_road_name=None,
                location_result=location_result,
                db=db_repo,
            )
            assert road_data["road"]["road_id"] == expected_road_id

            # Step 2: Contract Research
            contract_data = run_contract_research_agent(
                road_data=road_data,
                hybrid_searcher=real_hybrid_searcher,
                db=db_repo,
            )
            assert contract_data["best_contract_id"] == expected_contract_id
            assert contract_data["structured_match"] is True

            # Step 3: Officer Research
            officer_data = run_officer_research_agent(
                road_data=road_data,
                contract_data=contract_data,
                db=db_repo,
            )
            assert officer_data["officer"]["officer_id"] == expected_officer_id

            # Step 4: Validate compatibility with GraphState
            state: GraphState = {
                "run_id": "test-run-123",
                "image_url": "https://example.com/test.jpg",
                "user_location_hint": None,
                "exif_gps": None,
                "vision_result": None,
                "location_result": location_result,
                "road_data": road_data,
                "contract_data": contract_data,
                "officer_data": officer_data,
                "evidence_conflicts": None,
                "verification_confidence": None,
                "requires_human_review": None,
                "human_feedback": None,
                "complaint_record": None,
                "final_quality_score": None,
                "quality_explanation": None,
            }
            assert state["road_data"]["road"]["road_id"] == expected_road_id
            assert state["contract_data"]["best_contract_id"] == expected_contract_id
            assert state["officer_data"]["officer"]["officer_id"] == expected_officer_id
