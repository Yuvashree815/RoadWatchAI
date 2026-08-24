"""
LangGraph Workflow Orchestration for RoadWatch AI

Orchestrates the multi-agent workflow:
  START
    ↓
  Vision Agent
    ↓
  Location Agent
    ↓ (Conditional Route)
    ├─→ [Location Known] ──→ Parallel: [Road Research] + [Contract Research] ──→ Evidence Aggregation ──→ Officer Research
    └─→ [Location Unknown] ──→ Unresolved Evidence Handler ──────────────────────────────────────────→ Officer Research
                                                                                                            ↓
                                                                                                    Verification Agent
                                                                                                            ↓
                                                                                                    Complaint Agent
                                                                                                            ↓
                                                                                                Quality Evaluation Agent
                                                                                                            ↓ (Conditional Decision)
                                                                                  ┌─────────────────────────┴────────────────────────┐
                                                                                  ↓                                                  ↓
                                                                        [Quality Approved]                                 [Quality Rejected]
                                                                        Email Submission Node                              Submission Rejected Node
                                                                                  ↓                                                  ↓
                                                                                 END                                                END

Design notes:
  - Multi-agent state is passed explicitly via GraphState.
  - Dependency injection for LLM, DatabaseRepository, HybridSearcher, and EmailSubmissionService
    ensures the graph can be tested offline without external API keys or sending real emails.
  - Strict preservation of [DEMO] synthetic-data annotations.
"""
from datetime import datetime, timezone
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
from backend.services.email_service import EmailSubmissionService, default_email_service


# ── Workflow Factory ─────────────────────────────────────────────────────────

def build_roadwatch_graph(
    llm: Optional[Any] = None,
    db: Optional[DatabaseRepository] = None,
    hybrid_searcher: Optional[Any] = None,
    email_service: Optional[EmailSubmissionService] = None,
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
    email_service : EmailSubmissionService | None
        Injected email service for complaint submission.

    Returns
    -------
    CompiledStateGraph
        The executable LangGraph app.
    """
    active_email_service = email_service or default_email_service

    # ── 1. Node Implementations ──────────────────────────────────────────────

    def vision_node(state: GraphState) -> Dict[str, Any]:
        """Runs the Vision Agent to detect potholes and estimate severity."""
        if state.get("vision_result") is not None:
            # Pre-populated for testing or prior step
            return {
                "vision_result": state["vision_result"],
                "submission_status": "DETECTED",
            }

        image_url = state.get("image_url", "")
        if not image_url:
            return {
                "vision_result": {
                    "pothole_detected": False,
                    "severity": "none",
                    "confidence": 0.0,
                    "visual_evidence": "[DEMO] No image URL provided.",
                },
                "submission_status": "DETECTED",
            }

        if llm is None:
            raise RuntimeError(
                "Vision Agent in workflow requires an LLM instance. "
                "Pass an LLM to build_roadwatch_graph(llm=...) or supply "
                "a pre-populated vision_result in GraphState for offline testing."
            )

        result = run_vision_agent(image_url=image_url, llm=llm)
        return {
            "vision_result": result,
            "submission_status": "DETECTED",
        }

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
            "submission_status": "RESEARCHED",
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
            "submission_status": "RESEARCHED",
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
            "submission_status": "VERIFIED",
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
        existing_cid = state.get("complaint_id")
        record = run_complaint_agent(
            run_id=state.get("run_id", "unknown"),
            vision_result=state.get("vision_result"),
            location_result=state.get("location_result"),
            road_data=state.get("road_data"),
            contract_data=state.get("contract_data"),
            officer_data=state.get("officer_data"),
            verification_result=verification_result,
            complaint_id=existing_cid,
        )
        return {
            "complaint_record": record,
            "complaint_id": record.get("complaint_id"),
            "submission_status": "COMPLAINT_GENERATED",
        }

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
        score = result["final_quality_score"]
        status = "QUALITY_APPROVED" if score >= 70.0 else "QUALITY_REVIEW"
        return {
            "final_quality_score": score,
            "quality_explanation": result["quality_explanation"],
            "submission_status": status,
        }

    def email_submission_node(state: GraphState) -> Dict[str, Any]:
        """Submits the verified complaint with PDF attachment via EmailSubmissionService."""
        submission_res = active_email_service.submit_complaint(state)
        status_val = submission_res.get("status", "SUBMISSION_FAILED")
        return {
            "submission_status": status_val,
            "submission_result": submission_res,
        }

    def submission_rejected_node(state: GraphState) -> Dict[str, Any]:
        """Handles cases where automated submission was rejected due to quality/evidence issues."""
        score = state.get("final_quality_score") or 0.0
        pothole_detected = (state.get("vision_result") or {}).get("pothole_detected", False)
        if not pothole_detected:
            reason = "Automated submission skipped: No road damage/pothole detected."
        elif score < 50.0:
            reason = f"Automated submission rejected: Quality score ({score}/100) below minimum automated threshold (50/100)."
        else:
            reason = "Automated submission held for human review due to evidence conflicts or missing fields."

        return {
            "submission_status": "QUALITY_REJECTED",
            "submission_result": {
                "status": "REJECTED",
                "reason": reason,
                "quality_score": score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
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

    def route_after_quality(state: GraphState):
        """
        Determines whether the verified complaint is approved for automated email submission.
        """
        cr = state.get("complaint_record")
        score = state.get("final_quality_score") or 0.0
        pothole_detected = (state.get("vision_result") or {}).get("pothole_detected", False)

        # Do not submit if complaint record is missing, no damage detected, or quality score is poor (< 50)
        if not cr or not pothole_detected or score < 50.0:
            return "submission_rejected"
        return "email_submission"

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
    workflow.add_node("email_submission", email_submission_node)
    workflow.add_node("submission_rejected", submission_rejected_node)

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

    # Verification -> Complaint -> Quality Evaluation
    workflow.add_edge("officer_research", "verification")
    workflow.add_edge("verification", "complaint")
    workflow.add_edge("complaint", "quality_evaluation")

    # Conditional routing after quality evaluation: submission vs rejected
    workflow.add_conditional_edges(
        "quality_evaluation",
        route_after_quality,
        {
            "email_submission": "email_submission",
            "submission_rejected": "submission_rejected",
        },
    )

    workflow.add_edge("email_submission", END)
    workflow.add_edge("submission_rejected", END)

    return workflow.compile()
