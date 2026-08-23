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

# ── CORS Configuration ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

