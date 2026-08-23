"""
Quality Evaluation Agent

Produces a deterministic quality score and explanation for the full
RoadWatch AI pipeline run.

Scoring methodology (all factors explicit and auditable):
  Component                    Max Weight
  ─────────────────────────────────────────
  Vision confidence             20 pts
  Location confidence           20 pts
  Road record found             15 pts
  Maintenance project found     10 pts
  Contract / tender found       15 pts
  Structured contract match     5 pts
  Officer attributed             10 pts
  No evidence conflicts          5 pts
  ─────────────────────────────────────────
  Total                         100 pts

final_quality_score is expressed as a percentage (0–100).
Scores are rounded to 1 decimal place.
No LLM is required.
"""
from typing import Optional, Dict, Any, List


# ── Score weights ─────────────────────────────────────────────────────────────
WEIGHTS = {
    "vision_confidence":        20,
    "location_confidence":      20,
    "road_record_found":        15,
    "project_found":            10,
    "contract_found":           15,
    "structured_contract_match": 5,
    "officer_found":            10,
    "no_conflicts":              5,
}
assert sum(WEIGHTS.values()) == 100, "Weights must sum to 100"


def run_quality_evaluation_agent(
    vision_result: Optional[Dict[str, Any]],
    location_result: Optional[Dict[str, Any]],
    road_data: Optional[Dict[str, Any]],
    contract_data: Optional[Dict[str, Any]],
    officer_data: Optional[Dict[str, Any]],
    verification_result: Optional[Dict[str, Any]],
) -> dict:
    """
    Compute a deterministic quality score for the pipeline run.

    Returns
    -------
    dict
        {
            "final_quality_score": float,   # 0.0 – 100.0
            "quality_explanation": str,
            "component_scores": dict,
            "quality_flags": List[str],
        }
    """
    component_scores: Dict[str, float] = {}
    flags: List[str] = []

    # ── Vision confidence ─────────────────────────────────────────────────────
    vc = (vision_result or {}).get("confidence", 0.0)
    component_scores["vision_confidence"] = round(vc * WEIGHTS["vision_confidence"], 2)
    if vc == 0.0:
        flags.append("NO_VISION_CONFIDENCE: Vision analysis missing or zero confidence.")
    elif vc < 0.5:
        flags.append(f"LOW_VISION_CONFIDENCE: {vc:.2f}")

    # ── Location confidence ───────────────────────────────────────────────────
    lc = (location_result or {}).get("confidence", 0.0)
    component_scores["location_confidence"] = round(lc * WEIGHTS["location_confidence"], 2)
    if lc == 0.0:
        flags.append("UNRESOLVED_LOCATION: Location could not be determined.")
    elif lc < 0.7:
        flags.append(f"LOW_LOCATION_CONFIDENCE: {lc:.2f}")

    # ── Road record found ─────────────────────────────────────────────────────
    road = (road_data or {}).get("road")
    road_found = road is not None
    component_scores["road_record_found"] = WEIGHTS["road_record_found"] if road_found else 0.0
    if not road_found:
        flags.append("ROAD_NOT_FOUND: No road record could be identified.")

    # ── Maintenance project found ─────────────────────────────────────────────
    project = (road_data or {}).get("project")
    project_found = project is not None
    component_scores["project_found"] = WEIGHTS["project_found"] if project_found else 0.0
    if not project_found:
        flags.append("NO_MAINTENANCE_PROJECT: No active project for this road.")

    # ── Contract / tender found ───────────────────────────────────────────────
    contract_id = (contract_data or {}).get("best_contract_id")
    contract_found = contract_id is not None
    component_scores["contract_found"] = WEIGHTS["contract_found"] if contract_found else 0.0
    if not contract_found:
        flags.append("CONTRACT_NOT_FOUND: No contract/tender retrieved.")

    # ── Structured contract match ─────────────────────────────────────────────
    struct_match = (contract_data or {}).get("structured_match", False)
    component_scores["structured_contract_match"] = (
        WEIGHTS["structured_contract_match"] if struct_match else 0.0
    )
    if not struct_match:
        flags.append("NO_STRUCTURED_CONTRACT: Contract evidence is from unstructured docs only.")

    # ── Officer attributed ────────────────────────────────────────────────────
    officer = (officer_data or {}).get("officer")
    officer_found = officer is not None
    component_scores["officer_found"] = WEIGHTS["officer_found"] if officer_found else 0.0
    if not officer_found:
        flags.append("NO_OFFICER: Responsible officer could not be identified.")

    # ── No evidence conflicts ─────────────────────────────────────────────────
    conflicts = (verification_result or {}).get("conflicts", [])
    no_conflicts = len(conflicts) == 0
    component_scores["no_conflicts"] = WEIGHTS["no_conflicts"] if no_conflicts else 0.0
    if not no_conflicts:
        flags.append(f"EVIDENCE_CONFLICTS: {len(conflicts)} conflict(s) detected.")

    # ── Final score ───────────────────────────────────────────────────────────
    total = round(sum(component_scores.values()), 1)

    # ── Explanation ───────────────────────────────────────────────────────────
    level = (
        "EXCELLENT" if total >= 85 else
        "GOOD"      if total >= 70 else
        "FAIR"      if total >= 50 else
        "POOR"
    )

    if flags:
        flag_summary = "; ".join(flags[:3])
        if len(flags) > 3:
            flag_summary += f" (and {len(flags) - 3} more)"
        explanation = (
            f"[DEMO] Quality score: {total}/100 ({level}). "
            f"Issues: {flag_summary}."
        )
    else:
        explanation = (
            f"[DEMO] Quality score: {total}/100 ({level}). "
            "All evidence components are complete and consistent."
        )

    return {
        "final_quality_score": total,
        "quality_explanation": explanation,
        "component_scores": component_scores,
        "quality_flags": flags,
        "quality_level": level,
    }
