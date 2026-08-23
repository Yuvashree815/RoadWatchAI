"""
Officer Research Agent

Identifies and resolves the responsible synthetic government department and officer
associated with the road maintenance project or jurisdiction.

Responsibilities:
  - Extract officer_id from road_data["project"] (established by Road Research).
  - Retrieve full Officer record from the structured DatabaseRepository.
  - If project has no officer_id or is missing, attempt fallback by matching
    the road's district with the officers' jurisdiction in the database.
  - Return structured details: officer_id, officer_name, department, role, jurisdiction,
    confidence score, and human-readable [DEMO] notes.

Design notes:
  - Reuses DatabaseRepository from backend.database.repository.
  - No LLM required for deterministic relational lookup.
  - Purely synthetic/demo records.
"""
from typing import Optional, Dict, Any, List

from backend.database.models import Officer
from backend.database.repository import DatabaseRepository


def _officer_to_dict(officer: Officer) -> dict:
    return {
        "officer_id": officer.officer_id,
        "officer_name": officer.officer_name,
        "department": officer.department,
        "role": officer.role,
        "jurisdiction": officer.jurisdiction,
    }


def run_officer_research_agent(
    road_data: Optional[Dict[str, Any]],
    contract_data: Optional[Dict[str, Any]] = None,
    db: Optional[DatabaseRepository] = None,
) -> dict:
    """
    Resolve the responsible officer and department from road and project records.

    Parameters
    ----------
    road_data : dict | None
        Output from Road Research Agent containing "road" and "project" dicts.
    contract_data : dict | None
        Optional output from Contract Research Agent for cross-referencing.
    db : DatabaseRepository | None
        Injected database repository.

    Returns
    -------
    dict
        Ready to merge into GraphState["officer_data"]:
        {
            "officer": dict | None,
            "match_method": str,
            "confidence": float,
            "notes": str,
        }
    """
    if db is None:
        db = DatabaseRepository()

    officer: Optional[Officer] = None
    match_method = "none"

    project_info = road_data.get("project") if road_data else None
    road_info = road_data.get("road") if road_data else None

    # ── Strategy 1: Direct officer_id from maintenance project ───────────────
    if project_info and project_info.get("officer_id"):
        officer_id = project_info["officer_id"]
        officer = db.get_officer(officer_id)
        if officer:
            match_method = "project_direct_link"

    # ── Strategy 2: Fallback by district / jurisdiction matching ─────────────
    if officer is None and road_info and road_info.get("district"):
        district = road_info["district"].lower()
        all_officers = db._read_csv("officers.csv") if not db.use_supabase else []
        if db.use_supabase and db.client:
            res = db.client.table("officers").select("*").execute()
            all_officers = res.data if res.data else []

        for row in all_officers:
            if row.get("jurisdiction") and row["jurisdiction"].lower() in district or district in row.get("jurisdiction", "").lower():
                officer = db.get_officer(row["officer_id"])
                if officer:
                    match_method = "jurisdiction_fallback"
                    break

    # ── Build output ──────────────────────────────────────────────────────────
    if officer and match_method == "project_direct_link":
        confidence = 1.0
        notes = (
            f"[DEMO] Responsible officer '{officer.officer_name}' ({officer.officer_id}) "
            f"identified via active project record. "
            f"Department: {officer.department}, Role: {officer.role}, "
            f"Jurisdiction: {officer.jurisdiction}."
        )
    elif officer and match_method == "jurisdiction_fallback":
        confidence = 0.65
        notes = (
            f"[DEMO] No specific project officer assigned. Officer '{officer.officer_name}' "
            f"({officer.officer_id}) resolved via district jurisdiction '{officer.jurisdiction}' fallback."
        )
    else:
        confidence = 0.0
        notes = (
            "[DEMO] Responsible officer could not be identified. "
            "No project officer linked and no matching jurisdiction found."
        )

    return {
        "officer": _officer_to_dict(officer) if officer else None,
        "match_method": match_method,
        "confidence": confidence,
        "notes": notes,
    }
