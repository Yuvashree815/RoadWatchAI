"""
Unit tests for the Vision Agent and Location Agent.

Vision Agent tests use unittest.mock to avoid real LLM calls.
Location Agent tests run fully offline using the existing synthetic CSV data.
No API keys are required.
"""
import json
import math
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.agents.vision_agent import (
    VisionAgentOutput,
    run_vision_agent,
)
from backend.agents.location_agent import (
    _haversine_km,
    _hint_match,
    _load_demo_locations,
    _load_roads,
    _nearest_road,
    run_location_agent,
)


# ══════════════════════════════════════════════════════════════════════════════
# VISION AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestVisionAgentOutput:
    """Tests for the VisionAgentOutput Pydantic model."""

    def test_valid_output_parses(self):
        o = VisionAgentOutput(
            pothole_detected=True,
            severity="moderate",
            confidence=0.85,
            visual_evidence="A visible pothole with raised edges in the left lane.",
        )
        assert o.pothole_detected is True
        assert o.severity == "moderate"
        assert 0.0 <= o.confidence <= 1.0

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            VisionAgentOutput(
                pothole_detected=False,
                severity="none",
                confidence=1.5,  # out of range
                visual_evidence="clear road",
            )

    def test_invalid_severity_raises(self):
        o = VisionAgentOutput(
            pothole_detected=False,
            severity="catastrophic",  # not a valid value
            confidence=0.5,
            visual_evidence="some damage",
        )
        with pytest.raises(ValueError):
            o.validate_severity()

    def test_valid_severities(self):
        for sev in ("none", "low", "moderate", "severe"):
            o = VisionAgentOutput(
                pothole_detected=(sev != "none"),
                severity=sev,
                confidence=0.7,
                visual_evidence="test",
            )
            o.validate_severity()  # must not raise


class TestRunVisionAgent:
    """Tests for the run_vision_agent() function using a mocked LLM."""

    def _make_llm(self, json_payload: dict) -> MagicMock:
        """Return a mock LLM whose invoke() returns the given JSON as message content."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(json_payload)
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    def test_pothole_detected(self):
        llm = self._make_llm({
            "pothole_detected": True,
            "severity": "severe",
            "confidence": 0.92,
            "visual_evidence": "Deep pothole approximately 30 cm wide in the centre lane.",
        })
        result = run_vision_agent("https://example.com/road.jpg", llm)
        assert result["pothole_detected"] is True
        assert result["severity"] == "severe"
        assert result["confidence"] == 0.92
        assert "pothole" in result["visual_evidence"].lower()

    def test_no_pothole(self):
        llm = self._make_llm({
            "pothole_detected": False,
            "severity": "none",
            "confidence": 0.88,
            "visual_evidence": "Road surface appears smooth and undamaged.",
        })
        result = run_vision_agent("https://example.com/clear.jpg", llm)
        assert result["pothole_detected"] is False
        assert result["severity"] == "none"

    def test_llm_wraps_json_in_code_fence(self):
        """Some models wrap JSON in ```json ... ``` — the agent must strip it."""
        payload = {
            "pothole_detected": True,
            "severity": "low",
            "confidence": 0.6,
            "visual_evidence": "Minor surface cracking near the kerb.",
        }
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = f"```json\n{json.dumps(payload)}\n```"
        mock_llm.invoke.return_value = mock_response

        result = run_vision_agent("https://example.com/crack.jpg", mock_llm)
        assert result["severity"] == "low"

    def test_no_llm_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Vision Agent requires an LLM"):
            run_vision_agent("https://example.com/road.jpg", llm=None)

    def test_invalid_json_raises_value_error(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is not JSON at all."
        mock_llm.invoke.return_value = mock_response
        with pytest.raises(ValueError, match="non-JSON"):
            run_vision_agent("https://example.com/road.jpg", mock_llm)

    def test_result_has_all_required_keys(self):
        llm = self._make_llm({
            "pothole_detected": True,
            "severity": "moderate",
            "confidence": 0.75,
            "visual_evidence": "Moderate pothole observed.",
        })
        result = run_vision_agent("https://example.com/road.jpg", llm)
        for key in ("pothole_detected", "severity", "confidence", "visual_evidence"):
            assert key in result


# ══════════════════════════════════════════════════════════════════════════════
# LOCATION AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestLocationAgentUtilities:
    """Tests for the internal helper functions."""

    def test_haversine_same_point_is_zero(self):
        assert _haversine_km(34.0, -118.0, 34.0, -118.0) == pytest.approx(0.0)

    def test_haversine_known_distance(self):
        # ~111 km per degree of latitude
        d = _haversine_km(34.0, -118.0, 35.0, -118.0)
        assert 110.0 < d < 112.0

    def test_load_roads_returns_records(self):
        roads = _load_roads()
        assert len(roads) >= 12
        assert "road_id" in roads[0]
        assert "latitude" in roads[0]

    def test_load_demo_locations_returns_records(self):
        demos = _load_demo_locations()
        assert len(demos) >= 8
        assert "demo_case_id" in demos[0]

    def test_nearest_road_finds_correct_road(self):
        roads = _load_roads()
        # RD-007 is at lat=34.07, lon=-117.93
        best, dist = _nearest_road(34.07, -117.93, roads)
        assert best["road_id"] == "RD-007"
        assert dist < 0.1  # very close

    def test_hint_match_road_name(self):
        roads = _load_roads()
        match = _hint_match("Synthetic Road 3", roads)
        assert match is not None
        assert match["road_id"] == "RD-003"

    def test_hint_match_road_id(self):
        roads = _load_roads()
        match = _hint_match("RD-005", roads)
        assert match is not None
        assert match["road_id"] == "RD-005"

    def test_hint_match_no_result(self):
        roads = _load_roads()
        match = _hint_match("Nonexistent Street XYZ", roads)
        assert match is None


class TestRunLocationAgent:
    """Integration tests for run_location_agent()."""

    def test_exif_gps_resolution(self):
        # Coordinates very close to RD-007 (34.07, -117.93)
        result = run_location_agent(exif_gps={"latitude": 34.0705, "longitude": -117.9305})
        assert result["resolution_method"] == "exif_gps"
        assert result["estimated_road_name"] == "Synthetic Road 7"
        assert result["confidence"] > 0.8
        assert "[DEMO]" in result["notes"]
        assert result["latitude"] is not None

    def test_user_hint_resolution(self):
        result = run_location_agent(user_location_hint="Synthetic Road 4")
        assert result["resolution_method"] == "user_hint"
        assert result["estimated_road_name"] == "Synthetic Road 4"
        assert result["confidence"] == 0.75
        assert "[DEMO]" in result["notes"]

    def test_user_hint_by_road_id(self):
        result = run_location_agent(user_location_hint="RD-002")
        assert result["resolution_method"] == "user_hint"
        assert "Synthetic Road 2" in result["estimated_road_name"]

    def test_demo_case_mapping_resolution(self):
        result = run_location_agent(demo_case_id="DEMO-005")
        assert result["resolution_method"] == "demo_mapping"
        assert result["estimated_road_name"] == "Synthetic Road 5"
        assert result["demo_case_id"] == "DEMO-005"
        assert result["confidence"] == 0.90
        assert "pre-defined demonstration mapping" in result["notes"]

    def test_unknown_resolution_when_no_input(self):
        result = run_location_agent()
        assert result["resolution_method"] == "unknown"
        assert result["confidence"] == 0.0
        assert result["latitude"] is None
        assert result["estimated_road_name"] is None

    def test_exif_takes_priority_over_hint(self):
        # Both EXIF and hint provided — EXIF should win
        result = run_location_agent(
            exif_gps={"latitude": 34.01, "longitude": -117.99},
            user_location_hint="Synthetic Road 12",
        )
        assert result["resolution_method"] == "exif_gps"
        assert result["estimated_road_name"] == "Synthetic Road 1"  # nearest to (34.01, -117.99)

    def test_hint_takes_priority_over_demo_case(self):
        # Hint provided alongside demo_case_id — hint should win (strategy 2 > strategy 3)
        result = run_location_agent(
            user_location_hint="Synthetic Road 6",
            demo_case_id="DEMO-001",
        )
        assert result["resolution_method"] == "user_hint"
        assert result["estimated_road_name"] == "Synthetic Road 6"

    def test_all_result_keys_present(self):
        result = run_location_agent(demo_case_id="DEMO-003")
        expected_keys = {
            "resolution_method", "latitude", "longitude",
            "estimated_road_name", "location_hint_used",
            "confidence", "demo_case_id", "notes",
        }
        assert expected_keys.issubset(result.keys())

    def test_invalid_demo_case_falls_through_to_unknown(self):
        result = run_location_agent(demo_case_id="DEMO-999")
        assert result["resolution_method"] == "unknown"

    def test_graph_state_integration(self):
        """Verify the result shape is compatible with GraphState["location_result"]."""
        from backend.graph.state import GraphState
        result = run_location_agent(demo_case_id="DEMO-002")
        # Build a minimal GraphState and confirm no KeyError / TypeError
        state: GraphState = {
            "run_id": "test-run-001",
            "image_url": "https://example.com/img.jpg",
            "user_location_hint": None,
            "exif_gps": None,
            "vision_result": None,
            "location_result": result,
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
        assert state["location_result"]["estimated_road_name"] == "Synthetic Road 2"
