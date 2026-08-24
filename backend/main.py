"""
RoadWatch AI — FastAPI Backend Application

Provides endpoints for health checks, image upload analysis, and Server-Sent Events (SSE) streaming.
"""
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from backend.config import settings
from backend.api.service import default_analysis_service, UPLOAD_DIR
from backend.utils.complaint_pdf import generate_complaint_pdf

app = FastAPI(
    title="RoadWatch AI API",
    description=(
        "Backend API for RoadWatch AI — an automated multi-agent road damage "
        "investigation and complaint generation system using synthetic government data."
    ),
    version="0.1.0",
)

# ── Production CORS Configuration ─────────────────────────────────────────────
def get_allowed_origins() -> list:
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    frontend_url = os.environ.get("FRONTEND_URL", "").strip().rstrip("/")
    if frontend_url and frontend_url not in origins:
        origins.append(frontend_url)

    extra_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if extra_origins:
        for o in extra_origins.split(","):
            cleaned = o.strip().rstrip("/")
            if cleaned and cleaned not in origins:
                origins.append(cleaned)
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static File Serving for Uploaded Images ──────────────────────────────────
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ── Response Models ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str


class AnalysisResponse(BaseModel):
    run_id: str
    complaint_id: Optional[str] = None
    vision_result: Optional[Dict[str, Any]] = None
    location_result: Optional[Dict[str, Any]] = None
    road_data: Optional[Dict[str, Any]] = None
    contract_data: Optional[Dict[str, Any]] = None
    officer_data: Optional[Dict[str, Any]] = None
    evidence_conflicts: Optional[list] = None
    verification_confidence: Optional[float] = None
    requires_human_review: Optional[bool] = None
    complaint_record: Optional[Dict[str, Any]] = None
    final_quality_score: Optional[float] = None
    quality_explanation: Optional[str] = None
    submission_status: Optional[str] = None
    submission_result: Optional[Dict[str, Any]] = None
    disclaimer: str = Field(
        default=(
            "SYNTHETIC DEMO RECORD — All data is fictional and intended "
            "for demonstration purposes only."
        )
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify backend operational status.
    """
    return HealthResponse(
        status="ok",
        message="RoadWatch AI backend is running.",
        version="0.1.0",
    )


@app.get("/api/config")
async def get_system_config():
    """
    Returns sanitized system and observability configuration for frontend and test diagnostics.
    """
    from backend.observability import get_sanitized_config
    return get_sanitized_config()


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_road_image(
    file: UploadFile = File(...),
    location_hint: Optional[str] = Form(None),
):
    """
    Upload a road image with an optional location hint to run the full
    RoadWatch AI multi-agent workflow synchronously.
    """
    try:
        result = await default_analysis_service.run_analysis(
            file=file,
            user_location_hint=location_hint,
        )
        return AnalysisResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during analysis: {str(e)}",
        )


@app.post("/api/analyze/stream")
async def analyze_road_image_stream(
    file: UploadFile = File(...),
    location_hint: Optional[str] = Form(None),
):
    """
    Upload a road image with an optional location hint to stream real-time
    Server-Sent Events (SSE) as each agent in the workflow executes.
    """
    generator = default_analysis_service.stream_analysis(
        file=file,
        user_location_hint=location_hint,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/complaints/pdf")
async def download_complaint_pdf(payload: Dict[str, Any]):
    """
    Generates and returns a professionally formatted PDF complaint report
    from the structured complaint and analysis state.
    """
    try:
        pdf_bytes = generate_complaint_pdf(payload)
        cr = payload.get("complaint_record") or payload
        complaint_id = cr.get("complaint_id") or payload.get("run_id") or "report"
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(complaint_id))
        filename = f"RoadWatch_Complaint_{safe_id}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate complaint PDF: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)


