"""
Complaint Agent

Generates a structured, complaint-ready record from verified evidence.

Rules:
  - Only uses facts present in GraphState — never invents missing fields.
  - Every field in the output is clearly traceable to an agent's result.
  - All records are marked as SYNTHETIC DEMO RECORD.
  - Complaint is still generated even when verification found issues;
    the verification_status field communicates the evidence quality.
  - No LLM required; the record is deterministically assembled.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any


DISCLAIMER = (
    "SYNTHETIC DEMO RECORD — All data in this complaint record is fictional "
    "and used for demonstration purposes only. No real government contracts, "
    "officers, or locations are referenced."
)


def run_complaint_agent(
    run_id: str,
    vision_result: Optional[Dict[str, Any]],
    location_result: Optional[Dict[str, Any]],
    road_data: Optional[Dict[str, Any]],
    contract_data: Optional[Dict[str, Any]],
    officer_data: Optional[Dict[str, Any]],
    verification_result: Optional[Dict[str, Any]],
) -> dict:
    """
    Assemble a structured complaint record from verified agent evidence.

    Parameters
    ----------
    All parameters map to their corresponding GraphState fields.

    Returns
    -------
    dict
        Ready to merge into GraphState["complaint_record"].
    """
    # ── Extract available evidence ────────────────────────────────────────────
    road = (road_data or {}).get("road")
    project = (road_data or {}).get("project")
    contract = (contract_data or {}).get("contract_record")
    contractor = (contract_data or {}).get("contractor_record")
    officer = (officer_data or {}).get("officer")
    verification = verification_result or {}

    # ── Severity from vision ──────────────────────────────────────────────────
    severity = "unknown"
    vision_confidence = 0.0
    visual_description = "No image analysis available."
    pothole_detected = False
    if vision_result:
        severity = vision_result.get("severity", "unknown")
        vision_confidence = vision_result.get("confidence", 0.0)
        visual_description = vision_result.get("visual_evidence", "No description.")
        pothole_detected = vision_result.get("pothole_detected", False)

    # ── Location summary ──────────────────────────────────────────────────────
    location_summary = "Location not resolved."
    if location_result:
        est_road = location_result.get("estimated_road_name")
        method = location_result.get("resolution_method", "unknown")
        lat = location_result.get("latitude")
        lon = location_result.get("longitude")
        loc_parts = []
        if est_road:
            loc_parts.append(est_road)
        if lat is not None and lon is not None:
            loc_parts.append(f"({lat:.4f}, {lon:.4f})")
        loc_parts.append(f"[resolved via {method}]")
        location_summary = " ".join(loc_parts)

    # ── Road / project summary ────────────────────────────────────────────────
    road_summary = None
    project_summary = None
    if road:
        road_summary = {
            "road_id": road.get("road_id"),
            "road_name": road.get("road_name"),
            "district": road.get("district"),
            "area": road.get("area"),
        }
    if project:
        project_summary = {
            "project_id": project.get("project_id"),
            "contract_id": project.get("contract_id"),
            "maintenance_type": project.get("maintenance_type"),
            "status": project.get("status"),
        }

    # ── Contract / contractor summary ─────────────────────────────────────────
    contract_summary = None
    if contract:
        contract_summary = {
            "contract_id": contract.get("contract_id"),
            "tender_reference": contract.get("tender_reference"),
            "title": contract.get("title"),
            "start_date": contract.get("start_date"),
            "end_date": contract.get("end_date"),
            "contract_value": contract.get("contract_value"),
        }

    contractor_summary = None
    if contractor:
        contractor_summary = {
            "contractor_id": contractor.get("contractor_id"),
            "contractor_name": contractor.get("contractor_name"),
            "contact_email": contractor.get("contact_email"),
            "contact_phone": contractor.get("contact_phone"),
            "rating": contractor.get("rating"),
        }

    # ── Officer summary ───────────────────────────────────────────────────────
    officer_summary = None
    if officer:
        officer_summary = {
            "officer_id": officer.get("officer_id"),
            "officer_name": officer.get("officer_name"),
            "department": officer.get("department"),
            "role": officer.get("role"),
            "jurisdiction": officer.get("jurisdiction"),
        }

    # ── Verification status ───────────────────────────────────────────────────
    verified = verification.get("verified", False)
    requires_review = verification.get("requires_human_review", True)
    verification_confidence = verification.get("verification_confidence", 0.0)

    if verified:
        verification_status = "VERIFIED"
    elif requires_review:
        verification_status = "REQUIRES_HUMAN_REVIEW"
    else:
        verification_status = "PARTIAL"

    # ── Issue description ─────────────────────────────────────────────────────
    if pothole_detected and road:
        issue_desc = (
            f"Pothole detected on '{road.get('road_name', 'unknown road')}' "
            f"in {road.get('district', 'unknown district')}. "
            f"Severity: {severity.upper()}. "
            f"{visual_description}"
        )
    elif pothole_detected:
        issue_desc = (
            f"Pothole detected (location partially resolved). "
            f"Severity: {severity.upper()}. "
            f"{visual_description}"
        )
    else:
        issue_desc = (
            "Potential road damage reported. "
            "Pothole detection was inconclusive or negative. "
            f"{visual_description}"
        )

    # ── Assemble the complaint record ─────────────────────────────────────────
    complaint_id = f"DEMO-COMPLAINT-{str(uuid.uuid4())[:8].upper()}"
    generated_at = datetime.now(timezone.utc).isoformat()

    return {
        "complaint_id": complaint_id,
        "run_id": run_id,
        "generated_at": generated_at,
        "disclaimer": DISCLAIMER,

        # Issue
        "issue_description": issue_desc,
        "severity": severity,
        "pothole_detected": pothole_detected,
        "vision_confidence": vision_confidence,

        # Location
        "location_summary": location_summary,
        "location_detail": {
            "resolution_method": (location_result or {}).get("resolution_method"),
            "latitude": (location_result or {}).get("latitude"),
            "longitude": (location_result or {}).get("longitude"),
        },

        # Road & Project
        "road": road_summary,
        "maintenance_project": project_summary,

        # Contract & Contractor
        "contract": contract_summary,
        "contractor": contractor_summary,

        # Officer
        "responsible_officer": officer_summary,

        # Verification
        "verification_status": verification_status,
        "verification_confidence": verification_confidence,
        "evidence_conflicts": verification.get("conflicts", []),
        "evidence_warnings": verification.get("warnings", []),
        "missing_evidence": verification.get("missing_evidence", []),
    }
