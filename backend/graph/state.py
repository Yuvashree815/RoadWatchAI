"""
GraphState — the shared typed state dictionary passed between all LangGraph nodes.

Every agent reads from and writes to this state. LangGraph merges partial
updates returned by each node into the global state before passing it forward.
"""
from typing import TypedDict, Optional, List, Dict, Any


class VisionResult(TypedDict):
    pothole_detected: bool
    severity: str              # "none" | "low" | "moderate" | "severe"
    confidence: float          # model-estimated confidence, NOT a statistical probability
    visual_evidence: str       # natural-language description of what was observed


class LocationResult(TypedDict):
    resolution_method: str     # e.g. "exif_gps" | "user_hint" | "demo_mapping" | "unknown"
    latitude: Optional[float]
    longitude: Optional[float]
    estimated_road_name: Optional[str]
    location_hint_used: Optional[str]
    confidence: float          # estimated confidence in the location mapping
    demo_case_id: Optional[str]
    notes: str                 # human-readable explanation of how location was resolved


class GraphState(TypedDict):
    run_id: str
    complaint_id: Optional[str]
    image_url: str
    user_location_hint: Optional[str]
    exif_gps: Optional[Dict[str, float]]  # {"latitude": ..., "longitude": ...} if available

    # Agent outputs — populated sequentially / in parallel as workflow progresses
    vision_result: Optional[VisionResult]
    location_result: Optional[LocationResult]

    road_data: Optional[Dict[str, Any]]
    contract_data: Optional[Any]
    officer_data: Optional[Dict[str, Any]]

    # Verification & routing
    evidence_conflicts: Optional[List[str]]
    verification_confidence: Optional[float]
    requires_human_review: Optional[bool]
    human_feedback: Optional[Dict[str, Any]]

    # Final outputs
    complaint_record: Optional[Dict[str, Any]]
    final_quality_score: Optional[float]
    quality_explanation: Optional[str]

    # Submission status & tracking
    submission_status: Optional[str]  # "DETECTED" | "RESEARCHED" | "VERIFIED" | "COMPLAINT_GENERATED" | "QUALITY_APPROVED" | "QUALITY_REJECTED" | "SUBMITTED" | "SUBMISSION_FAILED" | "SUBMISSION_SKIPPED"
    submission_result: Optional[Dict[str, Any]]
