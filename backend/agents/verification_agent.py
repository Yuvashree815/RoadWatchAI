"""
Verification Agent

Examines the full evidence collected by prior agents and produces a
structured verification result.

Checks performed:
  1. Pothole detection: vision result must report pothole_detected=True.
  2. Vision confidence: flags low model-estimated confidence.
  3. Location resolution: flags "unknown" or low-confidence location.
  4. Road-project consistency: road data must have both road and project.
  5. Contract match consistency: project contract_id must match RAG result.
  6. Officer attribution: officer must have been resolved.
  7. Structured evidence completeness: all expected records present.

Design notes:
  - Fully deterministic — no LLM required.
  - Reads only from GraphState fields; does not re-query any database.
  - Sets requires_human_review=True on any conflict or critical gap.
  - Distinguishes WARNINGS (informational) from CONFLICTS (require review).
"""
from typing import Optional, Dict, Any, List


# ── Thresholds ────────────────────────────────────────────────────────────────
VISION_CONFIDENCE_WARNING = 0.5    # Below this → warn
LOCATION_CONFIDENCE_WARNING = 0.7  # Below this → warn


def run_verification_agent(
    vision_result: Optional[Dict[str, Any]],
    location_result: Optional[Dict[str, Any]],
    road_data: Optional[Dict[str, Any]],
    contract_data: Optional[Dict[str, Any]],
    officer_data: Optional[Dict[str, Any]],
    existing_conflicts: Optional[List[str]] = None,
) -> dict:
    """
    Verify the collected evidence and produce a structured verification result.

    Parameters
    ----------
    All parameters map directly to their corresponding GraphState fields.
    existing_conflicts : list | None
        Conflicts already detected by aggregate_evidence_node.

    Returns
    -------
    dict:
        {
            "verified": bool,
            "conflicts": List[str],
            "warnings": List[str],
            "missing_evidence": List[str],
            "requires_human_review": bool,
            "verification_confidence": float,
            "notes": str,
        }
    """
    conflicts: List[str] = list(existing_conflicts or [])
    warnings: List[str] = []
    missing: List[str] = []

    # ── 1. Vision check ──────────────────────────────────────────────────────
    if not vision_result:
        missing.append("vision_result: No vision analysis was performed.")
    else:
        if not vision_result.get("pothole_detected"):
            conflicts.append(
                "Vision analysis did not detect a pothole. "
                "The submitted image may not show road damage."
            )
        vc = vision_result.get("confidence", 0.0)
        if vc < VISION_CONFIDENCE_WARNING:
            warnings.append(
                f"Vision model-estimated confidence is low ({vc:.2f}). "
                "Detection may be unreliable."
            )

    # ── 2. Location check ────────────────────────────────────────────────────
    if not location_result:
        missing.append("location_result: Location resolution was not performed.")
    else:
        method = location_result.get("resolution_method", "unknown")
        lc = location_result.get("confidence", 0.0)
        if method == "unknown" or lc == 0.0:
            conflicts.append("Road location could not be determined.")
        elif lc < LOCATION_CONFIDENCE_WARNING:
            warnings.append(
                f"Location confidence is low ({lc:.2f}) via '{method}'. "
                "The road identification may be approximate."
            )

    # ── 3. Road / project check ──────────────────────────────────────────────
    if not road_data:
        missing.append("road_data: No road research result.")
    else:
        road = road_data.get("road")
        project = road_data.get("project")
        if not road:
            conflicts.append("Road record could not be identified from location evidence.")
        if road and not project:
            missing.append(
                f"No active maintenance project found for road "
                f"'{road.get('road_name', 'unknown')}'."
            )

    # ── 4. Contract / tender check ───────────────────────────────────────────
    if not contract_data:
        missing.append("contract_data: No contract research result.")
    else:
        if not contract_data.get("best_contract_id"):
            missing.append("No contract/tender identified from document retrieval.")
        if not contract_data.get("structured_match"):
            warnings.append(
                "Contract evidence is from unstructured documents only — "
                "no matching structured DB record confirmed."
            )
        # Cross-check project contract vs RAG contract
        project = (road_data or {}).get("project")
        if project and contract_data.get("best_contract_id"):
            proj_cid = project.get("contract_id")
            rag_cid = contract_data.get("best_contract_id")
            if proj_cid and rag_cid and proj_cid != rag_cid:
                conflicts.append(
                    f"Contract mismatch: Project links to '{proj_cid}' but "
                    f"document retrieval identified '{rag_cid}'."
                )

    # ── 5. Officer check ─────────────────────────────────────────────────────
    if not officer_data:
        missing.append("officer_data: Officer research not performed.")
    else:
        if not officer_data.get("officer"):
            warnings.append(
                "Responsible officer could not be identified. "
                "The complaint will lack officer attribution."
            )

    # ── 6. Compute verification_confidence ───────────────────────────────────
    # Start at 1.0 and deduct for each problem found
    score = 1.0
    score -= len(conflicts) * 0.25
    score -= len(missing) * 0.15
    score -= len(warnings) * 0.05
    verification_confidence = max(0.0, min(1.0, round(score, 3)))

    # Require human review for conflicts or significant missing evidence
    requires_human_review = len(conflicts) > 0 or len(missing) >= 2

    verified = len(conflicts) == 0 and len(missing) == 0

    # ── 7. Build summary notes ────────────────────────────────────────────────
    if verified:
        notes = (
            "[DEMO] All evidence verified successfully. "
            f"Verification confidence: {verification_confidence:.2f}."
        )
    else:
        parts = []
        if conflicts:
            parts.append(f"{len(conflicts)} conflict(s)")
        if missing:
            parts.append(f"{len(missing)} missing evidence item(s)")
        if warnings:
            parts.append(f"{len(warnings)} warning(s)")
        notes = (
            f"[DEMO] Verification completed with {', '.join(parts)}. "
            f"Confidence: {verification_confidence:.2f}."
        )

    return {
        "verified": verified,
        "conflicts": conflicts,
        "warnings": warnings,
        "missing_evidence": missing,
        "requires_human_review": requires_human_review,
        "verification_confidence": verification_confidence,
        "notes": notes,
    }
