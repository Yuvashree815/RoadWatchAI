"""
RoadWatch AI — Analysis Service & SSE Streaming

Orchestrates file upload validation, LangGraph workflow execution, and
Server-Sent Events (SSE) streaming with OpenAI multimodal vision integration.
"""
import os
import uuid
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, AsyncGenerator
from fastapi import UploadFile, HTTPException

from backend.graph.state import GraphState
from backend.graph.workflow import build_roadwatch_graph
from backend.database.repository import DatabaseRepository
from backend.rag.vector_store import VectorStoreManager
from backend.rag.keyword_search import KeywordSearchManager
from backend.rag.hybrid_search import HybridSearcher
from backend.llm import get_llm, is_llm_configured
from backend.config import settings
from backend.services.email_service import EmailSubmissionService, default_email_service
import backend.observability  # Auto-configures LangSmith tracing

# Supported image MIME types and extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

# Default upload directory
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


class AnalysisService:
    """
    Manages workflow instantiation and execution for API requests.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        db: Optional[DatabaseRepository] = None,
        hybrid_searcher: Optional[Any] = None,
        email_service: Optional[EmailSubmissionService] = None,
    ):
        self._injected_llm = llm
        self.db = db if db is not None else DatabaseRepository()
        self._hybrid_searcher = hybrid_searcher
        self.email_service = email_service or default_email_service
        self._app = None

    @property
    def llm(self):
        """Returns injected LLM or dynamically resolves from OPENAI_API_KEY."""
        if self._injected_llm is not None:
            return self._injected_llm
        return get_llm()

    @llm.setter
    def llm(self, value):
        self._injected_llm = value
        self._app = None  # Invalidate cached graph on LLM change

    def get_hybrid_searcher(self) -> HybridSearcher:
        if self._hybrid_searcher is None:
            try:
                persist_dir = settings.CHROMA_PERSIST_DIRECTORY
                os.makedirs(persist_dir, exist_ok=True)
                vm = VectorStoreManager(persist_directory=persist_dir)
                # Ingest if empty
                if vm.vector_store._collection.count() == 0:
                    vm.ingest_documents()
                km = KeywordSearchManager()
                self._hybrid_searcher = HybridSearcher(vm, km)
            except Exception as e:
                print(f"Warning: Could not initialize full HybridSearcher: {e}")
                self._hybrid_searcher = None
        return self._hybrid_searcher

    def get_workflow_app(self):
        if self._app is None:
            searcher = self.get_hybrid_searcher()
            self._app = build_roadwatch_graph(
                llm=self.llm,
                db=self.db,
                hybrid_searcher=searcher,
                email_service=self.email_service,
            )
        return self._app

    async def save_and_validate_file(self, file: UploadFile) -> str:
        """
        Validates content type and extension, saves file locally, and returns filepath.
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS or (file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file format '{ext}'. "
                    "Only JPG, JPEG, and PNG images are supported."
                ),
            )

        unique_name = f"{uuid.uuid4().hex[:10]}_{os.path.basename(file.filename)}"
        filepath = os.path.join(UPLOAD_DIR, unique_name)

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with open(filepath, "wb") as f:
            f.write(content)

        return filepath

    def create_initial_state(
        self,
        filepath: str,
        user_location_hint: Optional[str] = None,
        exif_gps: Optional[Dict[str, float]] = None,
        vision_result_override: Optional[Dict[str, Any]] = None,
    ) -> GraphState:
        """
        Prepares the initial GraphState for execution with unique tracking IDs.
        """
        hex_suffix = uuid.uuid4().hex[:8].upper()
        run_id = f"run-{hex_suffix.lower()}"
        complaint_id = f"DEMO-COMPLAINT-RW26-{hex_suffix[:6]}"
        return {
            "run_id": run_id,
            "complaint_id": complaint_id,
            "image_url": filepath,
            "user_location_hint": user_location_hint,
            "exif_gps": exif_gps,
            "vision_result": vision_result_override,
            "location_result": None,
            "road_data": None,
            "contract_data": None,
            "officer_data": None,
            "evidence_conflicts": None,
            "verification_confidence": None,
            "requires_human_review": None,
            "human_feedback": None,
            "complaint_record": None,
            "final_quality_score": None,
            "quality_explanation": None,
            "submission_status": "DETECTED",
            "submission_result": None,
        }

    async def run_analysis(
        self,
        file: UploadFile,
        user_location_hint: Optional[str] = None,
        vision_result_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the LangGraph workflow synchronously/asynchronously and returns final state.
        """
        if vision_result_override is None and self.llm is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "GEMINI_API_KEY is not configured. "
                    "Please configure GEMINI_API_KEY in your .env file to enable multimodal vision analysis."
                ),
            )

        filepath = await self.save_and_validate_file(file)
        initial_state = self.create_initial_state(
            filepath=filepath,
            user_location_hint=user_location_hint,
            vision_result_override=vision_result_override,
        )

        app = self.get_workflow_app()
        # Run workflow in a threadpool to prevent blocking the async event loop
        final_state = await asyncio.to_thread(app.invoke, initial_state)

        # Include standard synthetic disclaimer
        final_state["disclaimer"] = (
            "SYNTHETIC DEMO RECORD — All records and relationships in this analysis "
            "are fictional and intended for demonstration purposes only."
        )
        return final_state

    async def stream_analysis(
        self,
        file: UploadFile,
        user_location_hint: Optional[str] = None,
        vision_result_override: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Yields Server-Sent Events as the LangGraph workflow executes.
        """
        if vision_result_override is None and self.llm is None:
            error_payload = {
                "event": "workflow_error",
                "error": (
                    "GEMINI_API_KEY is not configured. "
                    "Please configure GEMINI_API_KEY in your .env file to enable multimodal vision analysis."
                ),
                "status_code": 400,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            yield f"event: workflow_error\ndata: {json.dumps(error_payload)}\n\n"
            return

        try:
            filepath = await self.save_and_validate_file(file)
        except HTTPException as e:
            error_payload = {
                "event": "workflow_error",
                "error": e.detail,
                "status_code": e.status_code,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            yield f"event: workflow_error\ndata: {json.dumps(error_payload)}\n\n"
            return

        initial_state = self.create_initial_state(
            filepath=filepath,
            user_location_hint=user_location_hint,
            vision_result_override=vision_result_override,
        )
        run_id = initial_state["run_id"]

        # 1. Event: workflow_started
        start_payload = {
            "event": "workflow_started",
            "run_id": run_id,
            "image_filename": os.path.basename(filepath),
            "user_location_hint": user_location_hint,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "[DEMO] Workflow started. Initializing multi-agent pipeline.",
        }
        yield f"event: workflow_started\ndata: {json.dumps(start_payload)}\n\n"

        app = self.get_workflow_app()
        aggregated_state = dict(initial_state)

        try:
            # We iterate through the stream chunks in a background thread/iterator
            def stream_chunks():
                return list(app.stream(initial_state, stream_mode="updates"))

            chunks = await asyncio.to_thread(stream_chunks)

            for chunk in chunks:
                for node_name, updates in chunk.items():
                    aggregated_state.update(updates)
                    now_str = datetime.now(timezone.utc).isoformat()

                    if node_name == "vision":
                        v_res = updates.get("vision_result") or {}
                        payload = {
                            "event": "vision_completed",
                            "node": "vision",
                            "pothole_detected": v_res.get("pothole_detected"),
                            "severity": v_res.get("severity"),
                            "confidence": v_res.get("confidence"),
                            "visual_evidence": v_res.get("visual_evidence"),
                            "timestamp": now_str,
                        }
                        yield f"event: vision_completed\ndata: {json.dumps(payload)}\n\n"

                    elif node_name == "location":
                        loc_res = updates.get("location_result") or {}
                        payload = {
                            "event": "location_completed",
                            "node": "location",
                            "resolution_method": loc_res.get("resolution_method"),
                            "estimated_road_name": loc_res.get("estimated_road_name"),
                            "confidence": loc_res.get("confidence"),
                            "notes": loc_res.get("notes"),
                            "timestamp": now_str,
                        }
                        yield f"event: location_completed\ndata: {json.dumps(payload)}\n\n"

                    elif node_name == "road_research":
                        road_res = updates.get("road_data") or {}
                        road = road_res.get("road")
                        project = road_res.get("project")
                        payload = {
                            "event": "evidence_found",
                            "node": "road_research",
                            "evidence_type": "road_and_project",
                            "road": road,
                            "project": project,
                            "road_id": road.get("road_id") if road else None,
                            "road_name": road.get("road_name") if road else None,
                            "district": road.get("district") if road else None,
                            "area": road.get("area") if road else None,
                            "project_id": project.get("project_id") if project else None,
                            "project_status": project.get("status") if project else None,
                            "contract_id": project.get("contract_id") if project else None,
                            "notes": road_res.get("notes"),
                            "timestamp": now_str,
                        }
                        yield f"event: evidence_found\ndata: {json.dumps(payload)}\n\n"

                    elif node_name == "contract_research":
                        cnt_res = updates.get("contract_data") or {}
                        payload = {
                            "event": "evidence_found",
                            "node": "contract_research",
                            "evidence_type": "contract_and_tender",
                            "best_contract_id": cnt_res.get("best_contract_id"),
                            "best_tender_reference": cnt_res.get("best_tender_reference"),
                            "contract_record": cnt_res.get("contract_record"),
                            "contractor_record": cnt_res.get("contractor_record"),
                            "structured_match": cnt_res.get("structured_match"),
                            "rag_confidence": cnt_res.get("rag_confidence"),
                            "notes": cnt_res.get("notes"),
                            "timestamp": now_str,
                        }
                        yield f"event: evidence_found\ndata: {json.dumps(payload)}\n\n"

                    elif node_name == "unresolved_evidence":
                        payload = {
                            "event": "human_review_required",
                            "node": "unresolved_evidence",
                            "reason": "Unresolved road location — insufficient evidence to locate maintenance records.",
                            "timestamp": now_str,
                        }
                        yield f"event: human_review_required\ndata: {json.dumps(payload)}\n\n"

                    elif node_name == "officer_research":
                        off_res = updates.get("officer_data") or {}
                        officer = off_res.get("officer")
                        payload = {
                            "event": "evidence_found",
                            "node": "officer_research",
                            "evidence_type": "officer",
                            "officer": officer,
                            "officer_id": officer.get("officer_id") if officer else None,
                            "officer_name": officer.get("officer_name") if officer else None,
                            "department": officer.get("department") if officer else None,
                            "role": officer.get("role") if officer else None,
                            "jurisdiction": officer.get("jurisdiction") if officer else None,
                            "notes": off_res.get("notes"),
                            "timestamp": now_str,
                        }
                        yield f"event: evidence_found\ndata: {json.dumps(payload)}\n\n"

                    elif node_name == "verification":
                        conflicts = updates.get("evidence_conflicts") or []
                        req_review = updates.get("requires_human_review", False)
                        ver_conf = updates.get("verification_confidence", 0.0)
                        payload = {
                            "event": "verification_completed",
                            "node": "verification",
                            "requires_human_review": req_review,
                            "verification_confidence": ver_conf,
                            "conflicts": conflicts,
                            "timestamp": now_str,
                        }
                        yield f"event: verification_completed\ndata: {json.dumps(payload)}\n\n"

                        if req_review:
                            rev_payload = {
                                "event": "human_review_required",
                                "conflicts": conflicts,
                                "message": "[DEMO] Human review required due to evidence conflicts or gaps.",
                                "timestamp": now_str,
                            }
                            yield f"event: human_review_required\ndata: {json.dumps(rev_payload)}\n\n"

                    elif node_name == "complaint":
                        complaint = updates.get("complaint_record") or {}
                        payload = {
                            "event": "complaint_generated",
                            "node": "complaint",
                            "complaint_id": complaint.get("complaint_id"),
                            "severity": complaint.get("severity"),
                            "verification_status": complaint.get("verification_status"),
                            "issue_description": complaint.get("issue_description"),
                            "timestamp": now_str,
                        }
                        yield f"event: complaint_generated\ndata: {json.dumps(payload)}\n\n"

                    elif node_name == "quality_evaluation":
                        score = updates.get("final_quality_score")
                        explanation = updates.get("quality_explanation")
                        payload = {
                            "event": "quality_evaluated",
                            "node": "quality_evaluation",
                            "final_quality_score": score,
                            "quality_explanation": explanation,
                            "timestamp": now_str,
                        }
                        yield f"event: quality_evaluated\ndata: {json.dumps(payload)}\n\n"

                    elif node_name == "email_submission":
                        sub_res = updates.get("submission_result") or {}
                        sub_status = updates.get("submission_status", "SUBMITTED")
                        payload = {
                            "event": "submission_completed" if sub_status == "SUBMITTED" else ("submission_skipped" if sub_status == "SUBMISSION_SKIPPED" else "submission_failed"),
                            "node": "email_submission",
                            "submission_status": sub_status,
                            "recipient": sub_res.get("recipient"),
                            "is_mock": sub_res.get("is_mock", True),
                            "complaint_id": sub_res.get("complaint_id"),
                            "pdf_attached": sub_res.get("pdf_attached", True),
                            "timestamp": now_str,
                        }
                        yield f"event: submission_completed\ndata: {json.dumps(payload)}\n\n"

                    elif node_name == "submission_rejected":
                        sub_res = updates.get("submission_result") or {}
                        payload = {
                            "event": "submission_rejected",
                            "node": "submission_rejected",
                            "submission_status": "QUALITY_REJECTED",
                            "reason": sub_res.get("reason"),
                            "timestamp": now_str,
                        }
                        yield f"event: submission_rejected\ndata: {json.dumps(payload)}\n\n"

            # Final Event: workflow_completed
            completed_payload = {
                "event": "workflow_completed",
                "run_id": run_id,
                "complaint_id": aggregated_state.get("complaint_id") or (aggregated_state.get("complaint_record") or {}).get("complaint_id"),
                "complaint_record": aggregated_state.get("complaint_record"),
                "final_quality_score": aggregated_state.get("final_quality_score"),
                "quality_explanation": aggregated_state.get("quality_explanation"),
                "requires_human_review": aggregated_state.get("requires_human_review"),
                "submission_status": aggregated_state.get("submission_status", "COMPLAINT_GENERATED"),
                "submission_result": aggregated_state.get("submission_result"),
                "disclaimer": (
                    "SYNTHETIC DEMO RECORD — All data is fictional and intended "
                    "for demonstration purposes only."
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            yield f"event: workflow_completed\ndata: {json.dumps(completed_payload)}\n\n"

        except Exception as e:
            err_payload = {
                "event": "workflow_error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            yield f"event: workflow_error\ndata: {json.dumps(err_payload)}\n\n"


# Global singleton instance for default application use
default_analysis_service = AnalysisService()
