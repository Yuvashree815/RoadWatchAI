"""
RoadWatch AI — Professional PDF Complaint Report Generator

Generates a presentation-ready government-style complaint document in PDF format
from the multi-agent analysis state using ReportLab.

Features:
- Clean A4 layout with consistent professional typography and margins.
- Two-pass NumberedCanvas for dynamic "Page X of Y" and running footer/timestamp.
- Structured tables for Issue, Location, Maintenance/Contract, Authority, Verification, and Quality.
- Robust line wrapping and automatic pagination for long descriptions/explanations.
- Prominent synthetic demo disclaimer box.
"""
import io
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


# ── Color Palette ─────────────────────────────────────────────────────────────
NAVY_DARK = colors.HexColor("#0F172A")    # slate-900
NAVY_MID = colors.HexColor("#1E293B")     # slate-800
INDIGO = colors.HexColor("#4F46E5")       # indigo-600
INDIGO_LIGHT = colors.HexColor("#EEF2FF") # indigo-50
SLATE_TEXT = colors.HexColor("#334155")   # slate-700
SLATE_MUTED = colors.HexColor("#64748B")  # slate-500
BORDER_COLOR = colors.HexColor("#CBD5E1") # slate-300
BG_ALT = colors.HexColor("#F8FAFC")       # slate-50
SUCCESS_GREEN = colors.HexColor("#16A34A")# green-600
WARN_AMBER = colors.HexColor("#D97706")   # amber-600
DANGER_RED = colors.HexColor("#DC2626")   # red-600
AMBER_BG = colors.HexColor("#FFFBEB")     # amber-50
AMBER_BORDER = colors.HexColor("#FCD34D") # amber-300


# ── Numbered Canvas for Two-Pass Page Numbering ──────────────────────────────
class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print 'Page X of Y' in footer,
    along with running header and synthetic demo notice.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        page_width, page_height = A4
        margin = 36  # 0.5 in

        # Header (on pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(SLATE_MUTED)
            self.drawString(margin, page_height - 24, "ROADWATCH AI — ROAD DAMAGE COMPLAINT REPORT")
            self.drawRightString(page_width - margin, page_height - 24, "[SYNTHETIC DEMO RECORD]")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(margin, page_height - 28, page_width - margin, page_height - 28)

        # Footer (all pages)
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(margin, 30, page_width - margin, 30)

        self.setFont("Helvetica", 8)
        self.setFillColor(SLATE_MUTED)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self.drawString(margin, 18, f"Generated: {now_str}  |  RoadWatch AI Autonomous System")
        self.drawRightString(page_width - margin, 18, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


# ── PDF Generation Function ───────────────────────────────────────────────────
def generate_complaint_pdf(analysis_state: Dict[str, Any]) -> bytes:
    """
    Generates a presentation-ready PDF complaint report from the analysis state.

    Parameters
    ----------
    analysis_state : dict
        The full analysis state or complaint dictionary.

    Returns
    -------
    bytes
        The raw PDF file content.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42,
    )

    # Styles
    base_styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=NAVY_DARK,
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=SLATE_MUTED,
        spaceAfter=10,
    )
    section_head_style = ParagraphStyle(
        "SectionHead",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=NAVY_DARK,
        spaceBefore=8,
        spaceAfter=4,
    )
    label_style = ParagraphStyle(
        "FieldLabel",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=SLATE_MUTED,
    )
    value_style = ParagraphStyle(
        "FieldValue",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=NAVY_DARK,
    )
    value_bold_style = ParagraphStyle(
        "FieldValueBold",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=NAVY_DARK,
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#92400E"),
    )

    story = []

    # Extract state sub-objects safely
    cr = analysis_state.get("complaint_record") or {}
    # If the root object itself is the complaint record:
    if "complaint_id" in analysis_state and not cr:
        cr = analysis_state

    vision = analysis_state.get("vision_result") or {}
    location = analysis_state.get("location_result") or {}
    road_data = analysis_state.get("road_data") or {}
    road_rec = road_data.get("road") or {}
    project_rec = road_data.get("project") or {}
    contract_data = analysis_state.get("contract_data") or {}
    officer_rec = cr.get("responsible_officer") or analysis_state.get("officer_data") or {}
    contractor_rec = cr.get("contractor") or {}

    complaint_id = cr.get("complaint_id") or analysis_state.get("run_id") or "UNKNOWN-ID"
    generated_at = cr.get("generated_at") or datetime.now(timezone.utc).isoformat()
    verification_status = cr.get("verification_status") or "UNVERIFIED"

    # ── Top Document Header ───────────────────────────────────────────────────
    story.append(Paragraph("ROADWATCH AI — ROAD DAMAGE COMPLAINT", title_style))
    story.append(Paragraph("AI-Assisted Road Issue Analysis & Structured Complaint Record", subtitle_style))

    # Top Metadata Card
    status_color = SUCCESS_GREEN if verification_status == "VERIFIED" else WARN_AMBER
    meta_table_data = [
        [
            Paragraph("<b>COMPLAINT ID:</b>", label_style),
            Paragraph(f"<b>{complaint_id}</b>", value_bold_style),
            Paragraph("<b>STATUS:</b>", label_style),
            Paragraph(f"<font color='{status_color.hexval()}'><b>{verification_status}</b></font>", value_bold_style),
        ],
        [
            Paragraph("<b>GENERATED AT:</b>", label_style),
            Paragraph(str(generated_at).replace("T", " ")[:19] + " UTC", value_style),
            Paragraph("<b>DEMO RUN ID:</b>", label_style),
            Paragraph(str(analysis_state.get("run_id") or "N/A"), value_style),
        ],
    ]
    meta_table = Table(meta_table_data, colWidths=[90, 170, 75, 185])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_ALT),
        ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Helper function to build 2-column key-value tables
    def build_kv_table(rows):
        table_data = []
        for label, val in rows:
            table_data.append([
                Paragraph(f"<b>{label}</b>", label_style),
                Paragraph(str(val) if val is not None else "N/A", value_style),
            ])
        t = Table(table_data, colWidths=[140, 380])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), BG_ALT),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        return t

    # ── 1. ISSUE DETAILS ──────────────────────────────────────────────────────
    story.append(Paragraph("1. ISSUE DETAILS", section_head_style))
    story.append(HRFlowable(width="100%", thickness=1, color=INDIGO, spaceBefore=1, spaceAfter=4))
    
    pothole_detected_str = "Yes" if vision.get("pothole_detected") else ("No" if "pothole_detected" in vision else "N/A")
    severity_str = str(vision.get("severity") or "N/A").upper()
    conf_val = vision.get("confidence")
    conf_str = f"{round(conf_val * 100)}%" if conf_val is not None else "N/A"
    
    issue_rows = [
        ("Issue Description", cr.get("issue_description") or vision.get("visual_evidence") or "Road surface damage detected."),
        ("Pothole Detected", pothole_detected_str),
        ("Estimated Severity", severity_str),
        ("AI Vision Confidence", f"{conf_str} (Model-estimated, non-calibrated)"),
        ("Visual Evidence", vision.get("visual_evidence") or "Visual inspection evidence recorded."),
    ]
    story.append(build_kv_table(issue_rows))
    story.append(Spacer(1, 8))

    # ── 2. LOCATION DETAILS ───────────────────────────────────────────────────
    story.append(Paragraph("2. LOCATION DETAILS", section_head_style))
    story.append(HRFlowable(width="100%", thickness=1, color=INDIGO, spaceBefore=1, spaceAfter=4))
    
    coords = location.get("coordinates")
    if isinstance(coords, dict):
        coords_str = f"{coords.get('latitude', 'N/A')}, {coords.get('longitude', 'N/A')}"
    elif coords:
        coords_str = str(coords)
    else:
        coords_str = "Not available (Photo lacked GPS metadata)"

    loc_conf = location.get("confidence")
    loc_conf_str = f"{round(loc_conf * 100)}%" if loc_conf is not None else "N/A"

    loc_rows = [
        ("Estimated Road Name", location.get("estimated_road_name") or cr.get("location_summary") or "Unresolved"),
        ("Road ID", location.get("road_id") or road_rec.get("road_id") or "N/A"),
        ("District / Area", location.get("district") or road_rec.get("district") or "N/A"),
        ("Coordinates", coords_str),
        ("Resolution Method", str(location.get("resolution_method") or "N/A")),
        ("Location Confidence", loc_conf_str),
    ]
    story.append(build_kv_table(loc_rows))
    story.append(Spacer(1, 8))

    # ── 3. MAINTENANCE / CONTRACT DETAILS ─────────────────────────────────────
    story.append(Paragraph("3. MAINTENANCE / CONTRACT DETAILS", section_head_style))
    story.append(HRFlowable(width="100%", thickness=1, color=INDIGO, spaceBefore=1, spaceAfter=4))

    rag_conf = contract_data.get("rag_confidence")
    rag_conf_str = f"{round(rag_conf * 100)}%" if rag_conf is not None else "N/A"

    contract_rows = [
        ("Project ID", project_rec.get("project_id") or road_data.get("project_id") or "N/A"),
        ("Project Status", project_rec.get("project_status") or "N/A"),
        ("Contract ID", contract_data.get("best_contract_id") or contractor_rec.get("contract_id") or "N/A"),
        ("Tender Reference", contract_data.get("best_tender_reference") or "N/A"),
        ("Contractor Name", contractor_rec.get("contractor_name") or contract_data.get("contractor_name") or "Unassigned"),
        ("RAG Match Status", f"{contract_data.get('match_status', 'N/A')} (Confidence: {rag_conf_str})"),
    ]
    story.append(build_kv_table(contract_rows))
    story.append(Spacer(1, 8))

    # ── 4. RESPONSIBLE AUTHORITY ──────────────────────────────────────────────
    story.append(Paragraph("4. RESPONSIBLE AUTHORITY", section_head_style))
    story.append(HRFlowable(width="100%", thickness=1, color=INDIGO, spaceBefore=1, spaceAfter=4))

    auth_rows = [
        ("Officer Name", officer_rec.get("officer_name") or "Unassigned"),
        ("Officer ID", officer_rec.get("officer_id") or "N/A"),
        ("Department", officer_rec.get("department") or "Road Maintenance & Public Works"),
        ("Designated Role", officer_rec.get("role") or "Jurisdiction Officer"),
        ("Jurisdiction Area", officer_rec.get("jurisdiction") or location.get("district") or "N/A"),
    ]
    story.append(build_kv_table(auth_rows))
    story.append(Spacer(1, 8))

    # ── 5. VERIFICATION ───────────────────────────────────────────────────────
    story.append(Paragraph("5. VERIFICATION", section_head_style))
    story.append(HRFlowable(width="100%", thickness=1, color=INDIGO, spaceBefore=1, spaceAfter=4))

    ver_conf = cr.get("verification_confidence") or analysis_state.get("verification_confidence")
    ver_conf_str = f"{round(ver_conf * 100)}%" if ver_conf is not None else "N/A"
    conflicts = cr.get("evidence_conflicts") or analysis_state.get("evidence_conflicts") or []
    conflicts_str = "; ".join(conflicts) if conflicts else "None (All multi-agent evidence cross-validated)"
    human_review = cr.get("requires_human_review") if "requires_human_review" in cr else analysis_state.get("requires_human_review")
    human_review_str = "YES — Human review required before automated filing" if human_review else "NO — Automated verification passed"

    ver_rows = [
        ("Verification Status", verification_status),
        ("Verification Confidence", ver_conf_str),
        ("Conflicts Detected", conflicts_str),
        ("Human Review Required", human_review_str),
    ]
    story.append(build_kv_table(ver_rows))
    story.append(Spacer(1, 8))

    # ── 6. AI QUALITY EVALUATION ──────────────────────────────────────────────
    story.append(Paragraph("6. AI QUALITY EVALUATION", section_head_style))
    story.append(HRFlowable(width="100%", thickness=1, color=INDIGO, spaceBefore=1, spaceAfter=4))

    quality_score = analysis_state.get("final_quality_score")
    if quality_score is None:
        quality_score = cr.get("final_quality_score")
    quality_score_str = f"{quality_score} / 100" if quality_score is not None else "N/A"
    quality_exp = analysis_state.get("quality_explanation") or cr.get("quality_explanation") or "Quality evaluation based on 8 weighted deterministic evidence components."

    qual_rows = [
        ("Final Quality Score", f"<b>{quality_score_str}</b>"),
        ("Quality Breakdown & Explanation", quality_exp),
    ]
    story.append(build_kv_table(qual_rows))
    story.append(Spacer(1, 10))

    # ── Final Disclaimer Box ──────────────────────────────────────────────────
    disclaimer_box_data = [
        [
            Paragraph(
                "<b>SYNTHETIC DEMO RECORD</b><br/>"
                "All roads, contracts, contractors, officers, locations, and complaint information contained in "
                "this document are fictional and intended strictly for demonstration purposes. "
                "This document is not an actual government complaint.",
                disclaimer_style,
            )
        ]
    ]
    disclaimer_box = Table(disclaimer_box_data, colWidths=[520])
    disclaimer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AMBER_BG),
        ("BOX", (0, 0), (-1, -1), 1, AMBER_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(KeepTogether([disclaimer_box]))

    # Build document with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
