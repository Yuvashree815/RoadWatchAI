"""
LangGraph Workflow Orchestration for RoadWatch AI

Orchestrates the multi-agent workflow:
  START
    ↓
  Vision Agent
    ↓
  Location Agent
    ↓ (Conditional Route)
    ├─→ [Location Known] ──→ Parallel: [Road Research] + [Contract Research] ──→ Evidence Aggregation ──→ Officer Research ──→ END
    └─→ [Location Unknown] ──→ Unresolved Evidence Handler ──→ Officer Research ──→ END

Design notes:
  - Multi-agent state is passed explicitly via GraphState.
  - Dependency injection for LLM, DatabaseRepository, and HybridSearcher ensures
    the graph can be tested offline without external API keys.
  - Strict preservation of [DEMO] synthetic-data annotations.
"""
from typing import Optional, Any, Dict, List
from langgraph.graph import StateGraph, START, END

from backend.graph.state import GraphState, VisionResult, LocationResult
from backend.agents.vision_agent import run_vision_agent
from backend.agents.location_agent import run_location_agent
from backend.agents.road_research_agent import run_road_research_agent
from backend.agents.contract_research_agent import run_contract_research_agent
from backend.agents.officer_research_agent import run_officer_research_agent
from backend.agents.verification_agent import run_verification_agent
from backend.agents.complaint_agent import run_complaint_agent
from backend.agents.quality_agent import run_quality_evaluation_agent
from backend.database.repository import DatabaseRepository
from backend.rag.hybrid_search import HybridSearcher


# ── Workflow Factory ─────────────────────────────────────────────────────────

def build_roadwatch_graph(
    llm: Optional[Any] = None,
    db: Optional[DatabaseRepository] = None,
    hybrid_searcher: Optional[Any] = None,
):
    """
    Build and compile the LangGraph workflow for RoadWatch AI.

    Parameters
    ----------
    llm : Any | None
        Injected LangChain chat model for Vision Agent. If None, vision node
        will use pre-existing vision_result in state (for offline testing)
        or raise a clear RuntimeError.
    db : DatabaseRepository | None
        Injected database repository for structured data lookups.
    hybrid_searcher : HybridSearcher | None
        Injected HybridSearcher for ChromaDB + BM25 retrieval.

    Returns
    -------
    CompiledStateGraph
        The executable LangGraph app.
    """
    # ── 1. Node Implementations ──────────────────────────────────────────────

    def vision_node(state: GraphState) -> Dict[str, Any]:
        """Runs the Vision Agent to detect potholes and estimate severity."""
        if state.get("vision_result") is not None:
            # Pre-populated for testing or prior step
            return {"vision_result": state["vision_result"]}

        image_url = state.get("image_url", "")
        if not image_url:
            return {
                "vision_result": {
                    "pothole_detected": False,
                    "severity": "none",
                    "confidence": 0.0,
                    "visual_evidence": "[DEMO] No image URL provided.",
                }
            }

        if llm is None:
            raise RuntimeError(
                "Vision Agent in workflow requires an LLM instance. "
                "Pass an LLM to build_roadwatch_graph(llm=...) or supply "
                "a pre-populated vision_result in GraphState for offline testing."
            )

        result = run_vision_agent(image_url=image_url, llm=llm)
        return {"vision_result": result}

    def location_node(state: GraphState) -> Dict[str, Any]:
        """Runs the Location Agent to resolve the road using DEMO mechanisms."""
        exif_gps = state.get("exif_gps")
        user_hint = state.get("user_location_hint")

        # Allow passing demo_case_id through existing location_result if present
        existing_loc = state.get("location_result")
        demo_case_id = existing_loc.get("demo_case_id") if existing_loc else None

        result = run_location_agent(
            exif_gps=exif_gps,
            user_location_hint=user_hint,
            demo_case_id=demo_case_id,
        )
        return {"location_result": result}

    def road_research_node(state: GraphState) -> Dict[str, Any]:
        """Queries structured database for road and maintenance project records."""
        loc_res = state.get("location_result", {})
        est_road_name = loc_res.get("estimated_road_name") if loc_res else None

        result = run_road_research_agent(
            estimated_road_name=est_road_name,
            location_result=loc_res,
            db=db,
        )
        return {"road_data": result}

    def contract_research_node(state: GraphState) -> Dict[str, Any]:
        """Queries ChromaDB via hybrid search for tender and contract documents."""
        road_data = state.get("road_data")
        loc_res = state.get("location_result")

        # If road_data is not yet in state (due to parallel branch execution),
        # resolve structured road and project information from location_result
        if not road_data and loc_res:
            est_road_name = loc_res.get("estimated_road_name")
            road_data = run_road_research_agent(
                estimated_road_name=est_road_name,
                location_result=loc_res,
                db=db,
            )

        if hybrid_searcher is None:
            return {
                "contract_data": {
                    "retrieved_chunks": [],
                    "best_contract_id": None,
                    "best_tender_reference": None,
                    "contract_record": None,
                    "contractor_record": None,
                    "rag_confidence": 0.0,
                    "structured_match": False,
                    "notes": "[DEMO] HybridSearcher not configured for workflow.",
                }
            }

        result = run_contract_research_agent(
            road_data=road_data,
            hybrid_searcher=hybrid_searcher,
            db=db,
            top_k=5,
        )
        return {"contract_data": result}

    def aggregate_evidence_node(state: GraphState) -> Dict[str, Any]:
        """Consolidates road research and contract research, detecting any conflicts."""
        conflicts: List[str] = []
        road_data = state.get("road_data") or {}
        contract_data = state.get("contract_data") or {}

        road = road_data.get("road")
        project = road_data.get("project")
        contract = contract_data.get("contract_record")

        # Conflict check 1: Road research found a project with a contract ID,
        # but contract research retrieved a different best contract ID
        if project and contract_data.get("best_contract_id"):
            proj_contract_id = project.get("contract_id")
            retrieved_contract_id = contract_data.get("best_contract_id")
            if proj_contract_id and retrieved_contract_id and proj_contract_id != retrieved_contract_id:
                conflicts.append(
                    f"Contract mismatch: Project links to '{proj_contract_id}', "
                    f"but hybrid RAG retrieved '{retrieved_contract_id}'."
                )

        # Conflict check 2: Road identified but no maintenance project active
        if road and not project:
            conflicts.append(f"No active maintenance project record found for road '{road.get('road_name')}'.")

        return {
            "evidence_conflicts": conflicts,
            "requires_human_review": len(conflicts) > 0,
        }

    def unresolved_evidence_node(state: GraphState) -> Dict[str, Any]:
        """Handles cases where location resolution failed."""
        return {
            "road_data": {
                "road": None,
                "project": None,
                "match_method": "none",
                "confidence": 0.0,
                "notes": "[DEMO] Road research skipped because location could not be determined.",
            },
            "contract_data": {
                "retrieved_chunks": [],
                "best_contract_id": None,
                "best_tender_reference": None,
                "contract_record": None,
                "contractor_record": None,
                "rag_confidence": 0.0,
                "structured_match": False,
                "notes": "[DEMO] Contract research skipped because location could not be determined.",
            },
            "evidence_conflicts": ["Unresolved road location — insufficient evidence to locate maintenance records."],
            "requires_human_review": True,
        }

    def officer_research_node(state: GraphState) -> Dict[str, Any]:
        """Queries structured database to identify the responsible officer."""
        road_data = state.get("road_data")
        contract_data = state.get("contract_data")

        result = run_officer_research_agent(
            road_data=road_data,
            contract_data=contract_data,
            db=db,
        )
        return {"officer_data": result}

    def verification_node(state: GraphState) -> Dict[str, Any]:
        """Verifies evidence consistency, flags conflicts and missing data."""
        result = run_verification_agent(
            vision_result=state.get("vision_result"),
            location_result=state.get("location_result"),
            road_data=state.get("road_data"),
            contract_data=state.get("contract_data"),
            officer_data=state.get("officer_data"),
            existing_conflicts=state.get("evidence_conflicts"),
        )
        return {
            "evidence_conflicts": result["conflicts"],
            "verification_confidence": result["verification_confidence"],
            "requires_human_review": result["requires_human_review"],
            # Store full verification result in complaint_record temporarily
            # (complaint node will replace it with the full record)
            "complaint_record": {"_verification_result": result},
        }

    def complaint_node(state: GraphState) -> Dict[str, Any]:
        """Generates a structured complaint record from verified evidence."""
        # Extract the full verification result stored in the previous step
        verification_result = (state.get("complaint_record") or {}).get(
            "_verification_result"
        )
        record = run_complaint_agent(
            run_id=state.get("run_id", "unknown"),
            vision_result=state.get("vision_result"),
            location_result=state.get("location_result"),
            road_data=state.get("road_data"),
            contract_data=state.get("contract_data"),
            officer_data=state.get("officer_data"),
            verification_result=verification_result,
        )
        return {"complaint_record": record}

    def quality_evaluation_node(state: GraphState) -> Dict[str, Any]:
        """Computes a deterministic quality score for the pipeline run."""
        # Reconstruct verification result from state fields
        verification_result = {
            "conflicts": state.get("evidence_conflicts") or [],
            "verification_confidence": state.get("verification_confidence") or 0.0,
        }
        result = run_quality_evaluation_agent(
            vision_result=state.get("vision_result"),
            location_result=state.get("location_result"),
            road_data=state.get("road_data"),
            contract_data=state.get("contract_data"),
            officer_data=state.get("officer_data"),
            verification_result=verification_result,
        )
        return {
            "final_quality_score": result["final_quality_score"],
            "quality_explanation": result["quality_explanation"],
        }

    # ── 2. Conditional Routing Logic ─────────────────────────────────────────

    def route_after_location(state: GraphState):
        """
        Determines whether to proceed to parallel research or to the unresolved handler.
        """
        loc_res = state.get("location_result") or {}
        method = loc_res.get("resolution_method", "unknown")
        confidence = loc_res.get("confidence", 0.0)

        if method == "unknown" or confidence <= 0.0:
            return "unresolved_evidence"
        return ["road_research", "contract_research"]

    # ── 3. Graph Assembly ────────────────────────────────────────────────────

    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("vision", vision_node)
    workflow.add_node("location", location_node)
    workflow.add_node("road_research", road_research_node)
    workflow.add_node("contract_research", contract_research_node)
    workflow.add_node("aggregate_evidence", aggregate_evidence_node)
    workflow.add_node("unresolved_evidence", unresolved_evidence_node)
    workflow.add_node("officer_research", officer_research_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("complaint", complaint_node)
    workflow.add_node("quality_evaluation", quality_evaluation_node)

    # Add Edges
    workflow.add_edge(START, "vision")
    workflow.add_edge("vision", "location")

    # Conditional routing after location
    workflow.add_conditional_edges(
        "location",
        route_after_location,
        {
            "unresolved_evidence": "unresolved_evidence",
            "road_research": "road_research",
            "contract_research": "contract_research",
        },
    )

    # Parallel join to aggregation
    workflow.add_edge("road_research", "aggregate_evidence")
    workflow.add_edge("contract_research", "aggregate_evidence")

    # Aggregation & Unresolved both lead to officer research
    workflow.add_edge("aggregate_evidence", "officer_research")
    workflow.add_edge("unresolved_evidence", "officer_research")

    # New Milestone 5 chain
    workflow.add_edge("officer_research", "verification")
    workflow.add_edge("verification", "complaint")
    workflow.add_edge("complaint", "quality_evaluation")
    workflow.add_edge("quality_evaluation", END)

    return workflow.compile()
