"""
Location Agent

Resolves the probable road location from available evidence using a clearly labelled
DEMO LOCATION RESOLUTION mechanism.

Resolution strategy (tried in priority order):
  1. EXIF GPS — if the uploaded image contains GPS metadata, use it directly.
  2. User location hint — if the user typed a road name or area, match it against
     the synthetic road database.
  3. Demo case mapping — if a demo_case_id is present (provided by the UI for
     presentation purposes), look it up in demo_locations.csv.
  4. Unknown — if none of the above yields a match, report low confidence.

IMPORTANT:
  This agent does NOT claim to magically determine exact real-world coordinates
  from arbitrary image pixels. Every resolution path is explicitly labelled so
  the UI and evaluators know which mechanism was used.

  In a real production system, steps 1–3 would be replaced by real GPS tagging,
  geocoding APIs, and a live government road database.
"""
import csv
import math
import os
from typing import Optional, Dict, Any

# ── Utility ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data")
)


def _load_roads() -> list[dict]:
    path = os.path.join(DATA_DIR, "roads.csv")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_demo_locations() -> list[dict]:
    path = os.path.join(DATA_DIR, "demo_locations.csv")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_road(lat: float, lon: float, roads: list[dict]) -> tuple[dict, float]:
    """Return (road_row, distance_km) for the synthetic road closest to the point."""
    best, best_dist = None, float("inf")
    for road in roads:
        dist = _haversine_km(lat, lon, float(road["latitude"]), float(road["longitude"]))
        if dist < best_dist:
            best_dist = dist
            best = road
    return best, best_dist


def _hint_match(hint: str, roads: list[dict]) -> Optional[dict]:
    """
    Try to match a free-text user hint against synthetic road names.
    Case-insensitive substring match; returns the first matching road or None.
    """
    hint_lower = hint.lower()
    for road in roads:
        if hint_lower in road["road_name"].lower() or hint_lower in road["road_id"].lower():
            return road
    return None


# ── Main agent function ───────────────────────────────────────────────────────

def run_location_agent(
    exif_gps: Optional[Dict[str, float]] = None,
    user_location_hint: Optional[str] = None,
    demo_case_id: Optional[str] = None,
) -> dict:
    """
    Resolve the probable road location using the DEMO LOCATION RESOLUTION mechanism.

    Parameters
    ----------
    exif_gps : dict | None
        GPS coordinates extracted from image EXIF data, e.g.
        {"latitude": 34.07, "longitude": -117.93}
    user_location_hint : str | None
        Free-text location hint provided by the user, e.g. "Synthetic Road 7".
    demo_case_id : str | None
        A demo case identifier (DEMO-001 … DEMO-008) used for presentation/evaluation.

    Returns
    -------
    dict
        A dict matching the LocationResult TypedDict shape, ready to be merged
        into GraphState["location_result"].
    """
    roads = _load_roads()
    demo_locations = _load_demo_locations()

    # ── Strategy 1: EXIF GPS ────────────────────────────────────────────────
    if exif_gps and "latitude" in exif_gps and "longitude" in exif_gps:
        lat = float(exif_gps["latitude"])
        lon = float(exif_gps["longitude"])
        nearest, dist_km = _nearest_road(lat, lon, roads)
        # Confidence degrades beyond 0.5 km; the synthetic roads are ~1.1 km apart
        confidence = max(0.0, 1.0 - (dist_km / 1.0))
        return {
            "resolution_method": "exif_gps",
            "latitude": lat,
            "longitude": lon,
            "estimated_road_name": nearest["road_name"] if nearest else None,
            "location_hint_used": None,
            "confidence": round(confidence, 3),
            "demo_case_id": None,
            "notes": (
                f"[DEMO] Location resolved from image EXIF GPS. "
                f"Nearest synthetic road: {nearest['road_name']} "
                f"({dist_km:.3f} km away)."
            ),
        }

    # ── Strategy 2: User location hint ──────────────────────────────────────
    if user_location_hint:
        matched = _hint_match(user_location_hint, roads)
        if matched:
            return {
                "resolution_method": "user_hint",
                "latitude": float(matched["latitude"]),
                "longitude": float(matched["longitude"]),
                "estimated_road_name": matched["road_name"],
                "location_hint_used": user_location_hint,
                "confidence": 0.75,
                "demo_case_id": None,
                "notes": (
                    f"[DEMO] Location resolved from user-provided hint "
                    f"'{user_location_hint}' matched to '{matched['road_name']}'."
                ),
            }

    # ── Strategy 3: Demo case mapping ────────────────────────────────────────
    if demo_case_id:
        for row in demo_locations:
            if row["demo_case_id"] == demo_case_id:
                road_id = row["road_id"]
                road = next((r for r in roads if r["road_id"] == road_id), None)
                return {
                    "resolution_method": "demo_mapping",
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "estimated_road_name": road["road_name"] if road else road_id,
                    "location_hint_used": None,
                    "confidence": 0.90,
                    "demo_case_id": demo_case_id,
                    "notes": (
                        f"[DEMO] Location resolved from demo case mapping. "
                        f"Case '{demo_case_id}' maps to road '{road_id}'. "
                        f"This is a pre-defined demonstration mapping, not real GPS."
                    ),
                }

    # ── Strategy 4: Unknown ──────────────────────────────────────────────────
    return {
        "resolution_method": "unknown",
        "latitude": None,
        "longitude": None,
        "estimated_road_name": None,
        "location_hint_used": user_location_hint,
        "confidence": 0.0,
        "demo_case_id": demo_case_id,
        "notes": (
            "[DEMO] Could not resolve location. No EXIF GPS, no matching user hint, "
            "and no demo case ID was provided. Human review is recommended."
        ),
    }
