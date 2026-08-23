"""
Road Research Agent

Queries the synthetic road database (Supabase when configured, local CSV fallback
otherwise) to identify the road and its associated maintenance project for a given
estimated location.

The agent accepts an estimated road name or road ID from the Location Agent's output
and resolves the full structured record: Road + RoadMaintenanceProject.

Design notes:
  - Reuses DatabaseRepository from backend.database.repository — no second DB layer.
  - The repository itself handles Supabase vs CSV fallback transparently.
  - No LLM is required; this is structured data lookup only.
  - Returns None fields rather than raising exceptions for missing records, so the
    Verification Agent can detect and flag gaps.
"""
from typing import Optional, Dict, Any

from backend.database.models import Road, RoadMaintenanceProject
from backend.database.repository import DatabaseRepository


# ── Agent output schema ───────────────────────────────────────────────────────

def _road_to_dict(road: Road) -> dict:
    return {
        "road_id": road.road_id,
        "road_name": road.road_name,
        "district": road.district,
        "area": road.area,
        "latitude": road.latitude,
        "longitude": road.longitude,
    }


def _project_to_dict(project: RoadMaintenanceProject) -> dict:
    return {
        "project_id": project.project_id,
        "road_id": project.road_id,
        "contract_id": project.contract_id,
        "contractor_id": project.contractor_id,
        "officer_id": project.officer_id,
        "maintenance_type": project.maintenance_type,
        "status": project.status,
    }


# ── Main agent function ───────────────────────────────────────────────────────

def run_road_research_agent(
    estimated_road_name: Optional[str],
    location_result: Optional[Dict[str, Any]],
    db: Optional[DatabaseRepository] = None,
) -> dict:
    """
    Look up the road record and associated maintenance project from structured data.

    Parameters
    ----------
    estimated_road_name : str | None
        Road name or road ID string from the Location Agent output.
    location_result : dict | None
        The full LocationResult dict from GraphState so we can also try
        demo_case_id if a direct name match fails.
    db : DatabaseRepository | None
        Injected database repository. If None, a new instance is created
        (which auto-falls back to CSV when Supabase is not configured).

    Returns
    -------
    dict
        Ready to merge into GraphState["road_data"]:
        {
            "road": dict | None,
            "project": dict | None,
            "match_method": str,
            "confidence": float,
            "notes": str,
        }
    """
    if db is None:
        db = DatabaseRepository()

    road: Optional[Road] = None
    match_method = "none"

    # ── Strategy 1: Direct road_id match if the name looks like an ID ────────
    if estimated_road_name and estimated_road_name.startswith("RD-"):
        road = db.get_road(estimated_road_name)
        if road:
            match_method = "road_id_direct"

    # ── Strategy 2: Scan all roads for a name substring match ─────────────────
    if road is None and estimated_road_name:
        all_roads = db._read_csv("roads.csv") if not db.use_supabase else []
        if db.use_supabase and db.client:
            res = db.client.table("roads").select("*").execute()
            all_roads = res.data if res.data else []
        needle = estimated_road_name.lower()
        for row in all_roads:
            if needle in row["road_name"].lower() or needle in row["road_id"].lower():
                road = db.get_road(row["road_id"])
                match_method = "name_substring"
                break

    # ── Strategy 3: Use demo_case_id from location_result ────────────────────
    if road is None and location_result:
        demo_case_id = location_result.get("demo_case_id")
        if demo_case_id:
            road = db.get_road_from_demo_location(demo_case_id)
            if road:
                match_method = "demo_case_mapping"

    # ── Resolve maintenance project ───────────────────────────────────────────
    project: Optional[RoadMaintenanceProject] = None
    if road:
        project = db.get_maintenance_project(road.road_id)

    # ── Build output ──────────────────────────────────────────────────────────
    if road and project:
        confidence = 1.0 if match_method == "road_id_direct" else 0.85
        notes = (
            f"[DEMO] Road '{road.road_name}' ({road.road_id}) found via '{match_method}'. "
            f"Active maintenance project '{project.project_id}' identified. "
            f"Contract ID: {project.contract_id}."
        )
    elif road:
        confidence = 0.5
        notes = (
            f"[DEMO] Road '{road.road_name}' found via '{match_method}', "
            f"but no maintenance project record was found for this road."
        )
    else:
        confidence = 0.0
        notes = (
            f"[DEMO] Could not identify road from estimated name "
            f"'{estimated_road_name}'. No structured record found."
        )

    return {
        "road": _road_to_dict(road) if road else None,
        "project": _project_to_dict(project) if project else None,
        "match_method": match_method,
        "confidence": confidence,
        "notes": notes,
    }
