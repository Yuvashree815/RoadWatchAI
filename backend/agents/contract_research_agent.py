"""
Contract / Tender Research Agent

Retrieves relevant tender and contract information for a known road using the
existing hybrid search infrastructure (vector + BM25 + RRF).

Responsibilities:
  - Build a rich query from road name, road ID, and project context.
  - Run hybrid_search() over ChromaDB to retrieve relevant contract document chunks.
  - Optionally filter by metadata (road_name, contract_id) when available from
    the road research output.
  - Extract the best-matching contract_id and tender_reference from chunk metadata.
  - Also look up the full contract and contractor records from the structured DB
    to supplement the unstructured RAG result.

Design notes:
  - Reuses HybridSearcher from backend.rag.hybrid_search — no second RAG system.
  - The HybridSearcher itself is injected; tests can pass a lightweight mock.
  - No LLM is required for retrieval; the agent uses metadata from ChromaDB chunks.
  - An optional LLM may be injected for future re-ranking or summarisation, but is
    not called in this milestone.
"""
from typing import Optional, Dict, Any, List

from backend.database.models import Contract, Contractor
from backend.database.repository import DatabaseRepository


# ── Main agent function ───────────────────────────────────────────────────────

def run_contract_research_agent(
    road_data: Optional[Dict[str, Any]],
    hybrid_searcher,                        # HybridSearcher instance
    db: Optional[DatabaseRepository] = None,
    top_k: int = 5,
) -> dict:
    """
    Retrieve tender / contract information using hybrid search + structured DB lookup.

    Parameters
    ----------
    road_data : dict | None
        The road_data dict from GraphState (output of Road Research Agent).
        Expected keys: "road" (with road_id, road_name) and "project"
        (with contract_id, contractor_id).
    hybrid_searcher : HybridSearcher
        Injected hybrid searcher (vector + BM25 + RRF over ChromaDB).
    db : DatabaseRepository | None
        Injected DB repository. Auto-created if None.
    top_k : int
        Number of document chunks to retrieve.

    Returns
    -------
    dict
        Ready to merge into GraphState["contract_data"]:
        {
            "retrieved_chunks": List[dict],
            "best_contract_id": str | None,
            "best_tender_reference": str | None,
            "contract_record": dict | None,
            "contractor_record": dict | None,
            "rag_confidence": float,
            "structured_match": bool,
            "notes": str,
        }
    """
    if db is None:
        db = DatabaseRepository()

    # ── Build query from road context ─────────────────────────────────────────
    road_info = road_data.get("road") if road_data else None
    project_info = road_data.get("project") if road_data else None

    if road_info:
        road_name = road_info.get("road_name", "")
        road_id = road_info.get("road_id", "")
        query = f"maintenance tender contract {road_name} {road_id}"
    else:
        query = "road maintenance tender contract"

    # ── Optional metadata filter using known contract_id from project ─────────
    # If Road Research already resolved a contract_id, we can pin the filter.
    metadata_filter = None
    known_contract_id = project_info.get("contract_id") if project_info else None
    if known_contract_id:
        metadata_filter = {"contract_id": known_contract_id}

    # ── Run hybrid search ─────────────────────────────────────────────────────
    chunks: List[dict] = hybrid_searcher.hybrid_search(
        query,
        metadata_filter=metadata_filter,
        top_k=top_k,
    )

    # ── Extract best contract from top chunk metadata ─────────────────────────
    best_contract_id: Optional[str] = None
    best_tender_ref: Optional[str] = None

    if chunks:
        top_metadata = chunks[0].get("metadata", {})
        best_contract_id = top_metadata.get("contract_id")
        best_tender_ref = top_metadata.get("tender_reference")

    # ── Prefer the structured contract_id when we already know it ────────────
    if known_contract_id:
        best_contract_id = known_contract_id

    # ── Look up full contract + contractor records from DB ────────────────────
    contract_record: Optional[Contract] = None
    contractor_record: Optional[Contractor] = None

    if best_contract_id:
        contract_record = db.get_contract(best_contract_id)
        if contract_record:
            contractor_record = db.get_contractor(contract_record.contractor_id)

    # ── Compute a simple RAG confidence based on top RRF score ───────────────
    rag_confidence = 0.0
    if chunks:
        # RRF scores are small positive numbers; normalise to 0-1 range
        # using a sigmoid-style clamp. A score >= 0.03 is considered high confidence.
        top_score = chunks[0].get("rrf_score", 0.0)
        rag_confidence = min(1.0, top_score / 0.03)

    structured_match = contract_record is not None

    # ── Build readable chunk summaries for GraphState ─────────────────────────
    chunk_summaries = [
        {
            "contract_id": c.get("metadata", {}).get("contract_id"),
            "tender_reference": c.get("metadata", {}).get("tender_reference"),
            "road_name": c.get("metadata", {}).get("road_name"),
            "rrf_score": c.get("rrf_score"),
            "snippet": c.get("page_content", "")[:200],
        }
        for c in chunks
    ]

    # ── Build notes ───────────────────────────────────────────────────────────
    if structured_match:
        notes = (
            f"[DEMO] Hybrid search retrieved {len(chunks)} chunk(s). "
            f"Best contract: {best_contract_id} (Tender: {best_tender_ref}). "
            f"Structured DB record confirmed. "
            f"Contractor: {contractor_record.contractor_name if contractor_record else 'Unknown'}."
        )
    elif chunks:
        notes = (
            f"[DEMO] Hybrid search retrieved {len(chunks)} chunk(s) "
            f"but structured DB lookup for '{best_contract_id}' returned no record. "
            f"Evidence is from unstructured documents only."
        )
    else:
        notes = (
            f"[DEMO] Hybrid search returned no results for query: '{query}'. "
            f"Contract evidence is missing."
        )

    return {
        "retrieved_chunks": chunk_summaries,
        "best_contract_id": best_contract_id,
        "best_tender_reference": best_tender_ref,
        "contract_record": contract_record.model_dump() if contract_record else None,
        "contractor_record": contractor_record.model_dump() if contractor_record else None,
        "rag_confidence": round(rag_confidence, 3),
        "structured_match": structured_match,
        "notes": notes,
    }
