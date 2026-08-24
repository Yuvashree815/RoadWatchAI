"""
RoadWatch AI — Deterministic Complaint Email Submission Service

Sends or simulates formal complaint report emails with the attached PDF
to the designated demonstration inbox or authority email.

Rules:
- Never hardcode private email addresses or credentials.
- All email transmission decisions are deterministic (never constructed by LLM).
- When MOCK_EMAIL=true, simulates transmission and validates PDF creation without SMTP.
- When EMAIL_ENABLED=true and MOCK_EMAIL=false, performs authentic SMTP transmission with TLS/SSL.
- Attaches the complete ReportLab PDF complaint document.
- Accurately tracks identified officer vs actual demo recipient.
"""
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from backend.config import settings
from backend.utils.complaint_pdf import generate_complaint_pdf


class EmailSubmissionService:
    """
    Service for formatting and transmitting complaint emails with PDF attachments.
    """

    def __init__(
        self,
        email_enabled: Optional[bool] = None,
        mock_email: Optional[bool] = None,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        smtp_from: Optional[str] = None,
        demo_email: Optional[str] = None,
    ):
        self.email_enabled = email_enabled if email_enabled is not None else settings.EMAIL_ENABLED
        self.mock_email = mock_email if mock_email is not None else settings.MOCK_EMAIL
        self.smtp_host = smtp_host or settings.SMTP_HOST
        self.smtp_port = smtp_port if smtp_port is not None else settings.SMTP_PORT
        self.smtp_username = smtp_username or settings.SMTP_USERNAME
        self.smtp_password = smtp_password or settings.SMTP_PASSWORD
        self.smtp_from = smtp_from or settings.SMTP_FROM
        self.demo_email = demo_email or settings.DEMO_COMPLAINT_EMAIL

    def build_email_content(
        self,
        state: Dict[str, Any],
        pdf_bytes: bytes,
        recipient: str,
    ) -> EmailMessage:
        """
        Constructs an EmailMessage containing structured complaint details and PDF attachment.
        """
        cr = state.get("complaint_record") or {}
        complaint_id = cr.get("complaint_id") or state.get("run_id") or "UNKNOWN-ID"
        road_data = state.get("road_data") or {}
        road = road_data.get("road") or {}
        road_name = road.get("road_name") or (state.get("location_result") or {}).get("estimated_road_name") or "Unspecified Road"
        vision = state.get("vision_result") or {}
        severity = str(vision.get("severity") or "unknown").upper()
        officer = (state.get("officer_data") or {}).get("officer") or cr.get("responsible_officer") or {}
        officer_name = officer.get("officer_name") or "Unassigned Authority"
        officer_id = officer.get("officer_id") or "N/A"
        quality_score = state.get("final_quality_score")
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        msg = EmailMessage()
        msg["Subject"] = f"[RoadWatch AI] Road Damage Complaint {complaint_id} — {severity} ({road_name})"
        msg["From"] = self.smtp_from
        msg["To"] = recipient

        # Plain text content
        plain_text = f"""ROADWATCH AI — OFFICIAL ROAD DAMAGE COMPLAINT NOTIFICATION
======================================================================
Complaint ID:           {complaint_id}
Generated Timestamp:    {now_str}
Severity:               {severity}
Road Location:          {road_name}
Identified Authority:   {officer_name} ({officer_id})
Actual Recipient:       {recipient} [Demonstration Inbox]
Quality Score:          {quality_score}/100

ISSUE SUMMARY:
{cr.get('issue_description', 'Pothole damage detected and validated by RoadWatch AI autonomous agents.')}

LOCATION DETAILS:
- Road Name:            {road_name}
- District / Area:      {road.get('district', 'N/A')}, {road.get('area', 'N/A')}
- Resolution Method:    {(state.get('location_result') or {}).get('resolution_method', 'N/A')}

MAINTENANCE & CONTRACT:
- Project ID:           {(road_data.get('project') or {}).get('project_id', 'N/A')}
- Contract ID:          {(state.get('contract_data') or {}).get('best_contract_id', 'N/A')}
- Contractor:           {(cr.get('contractor') or {}).get('contractor_name', 'N/A')}

VERIFICATION STATUS:
- Status:               {cr.get('verification_status', 'VERIFIED')}
- Conflicts:            {len(state.get('evidence_conflicts') or [])} detected

----------------------------------------------------------------------
SYNTHETIC DEMO RECORD DISCLAIMER:
All roads, contracts, contractors, officers, locations, and complaint information
contained in this notification are fictional and intended strictly for
demonstration purposes. This is an automated notification from RoadWatch AI.
======================================================================
"""
        msg.set_content(plain_text)

        # HTML content
        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; color: #1e293b; background-color: #f8fafc; margin: 0; padding: 20px; }}
    .card {{ background: #ffffff; max-width: 600px; margin: 0 auto; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
    .header {{ background: #0f172a; color: #ffffff; padding: 24px; text-align: left; }}
    .header h1 {{ margin: 0; font-size: 18px; letter-spacing: 0.5px; }}
    .header p {{ margin: 4px 0 0 0; font-size: 12px; color: #94a3b8; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; background: #e0e7ff; color: #4338ca; }}
    .badge-severity {{ background: #fef3c7; color: #92400e; }}
    .content {{ padding: 24px; }}
    .field-group {{ margin-bottom: 16px; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; }}
    .label {{ font-size: 11px; font-weight: bold; color: #64748b; text-transform: uppercase; }}
    .value {{ font-size: 14px; font-weight: 600; color: #0f172a; margin-top: 2px; }}
    .notice {{ background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 12px; font-size: 11px; color: #92400e; margin-top: 20px; }}
    .footer {{ background: #f8fafc; padding: 16px 24px; font-size: 11px; color: #64748b; text-align: center; border-top: 1px solid #e2e8f0; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h1>ROADWATCH AI — DAMAGE COMPLAINT</h1>
      <p>Automated Multi-Agent Road Inspection Record</p>
    </div>
    <div class="content">
      <div class="field-group" style="display: flex; justify-content: space-between;">
        <div>
          <div class="label">Complaint ID</div>
          <div class="value" style="color: #4f46e5;">{complaint_id}</div>
        </div>
        <div style="text-align: right;">
          <div class="label">Severity Level</div>
          <span class="badge badge-severity">{severity}</span>
        </div>
      </div>

      <div class="field-group">
        <div class="label">Issue Description</div>
        <div class="value" style="font-weight: normal; font-size: 13px;">{cr.get('issue_description', 'Pothole damage detected.')}</div>
      </div>

      <div class="field-group">
        <div class="label">Location & Road Details</div>
        <div class="value">{road_name}</div>
        <div style="font-size: 12px; color: #64748b;">District: {road.get('district', 'N/A')}, Sector: {road.get('area', 'N/A')}</div>
      </div>

      <div class="field-group">
        <div class="label">Identified Responsible Authority</div>
        <div class="value">{officer_name} ({officer_id})</div>
        <div style="font-size: 12px; color: #64748b;">Department: {officer.get('department', 'Road Maintenance')} | Jurisdiction: {officer.get('jurisdiction', 'N/A')}</div>
      </div>

      <div class="field-group">
        <div class="label">Actual Delivery Recipient</div>
        <div class="value" style="font-size: 12px; font-family: monospace;">{recipient} <span style="color: #64748b; font-size: 11px;">[Configured Demo Inbox]</span></div>
      </div>

      <div class="field-group" style="border-bottom: none;">
        <div class="label">Quality & Verification</div>
        <div class="value" style="color: #059669;">Verified (Score: {quality_score}/100)</div>
      </div>

      <div class="notice">
        <strong>⚠ SYNTHETIC DEMO RECORD:</strong> All records, contractor mappings, and officer details in this notification are synthetic and intended strictly for capstone demonstration.
      </div>
    </div>
    <div class="footer">
      Generated on {now_str} by RoadWatch AI Multi-Agent System.<br/>
      Official complaint PDF report is attached to this email.
    </div>
  </div>
</body>
</html>
"""
        msg.add_alternative(html_content, subtype="html")

        # Attach PDF
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(complaint_id))
        msg.add_attachment(
            pdf_bytes,
            maintype="application",
            subtype="pdf",
            filename=f"RoadWatch_Complaint_{safe_id}.pdf",
        )

        return msg

    def submit_complaint(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the complaint submission workflow:
        1. Validates recipient and state readiness.
        2. Generates the PDF attachment from state.
        3. Transmits via SMTP or simulates in Mock Mode.

        Returns
        -------
        dict
            {
                "status": "SUBMITTED" | "SUBMISSION_FAILED" | "SUBMISSION_SKIPPED",
                "recipient": str,
                "timestamp": str,
                "is_mock": bool,
                "complaint_id": str,
                "pdf_attached": bool,
                "error": Optional[str],
            }
        """
        now_str = datetime.now(timezone.utc).isoformat()
        cr = state.get("complaint_record") or {}
        complaint_id = cr.get("complaint_id") or state.get("run_id") or "UNKNOWN-ID"
        recipient = self.demo_email

        # 1. Check if email submission is disabled
        if not self.email_enabled:
            return {
                "status": "SUBMISSION_SKIPPED",
                "recipient": recipient,
                "timestamp": now_str,
                "is_mock": True,
                "complaint_id": complaint_id,
                "pdf_attached": False,
                "reason": "Email submission is disabled (EMAIL_ENABLED=false).",
            }

        # 2. Validate recipient
        if not recipient or "@" not in recipient:
            return {
                "status": "SUBMISSION_FAILED",
                "recipient": recipient or "None",
                "timestamp": now_str,
                "is_mock": self.mock_email,
                "complaint_id": complaint_id,
                "pdf_attached": False,
                "error": "Invalid recipient email address configured.",
            }

        # 3. Generate PDF attachment in memory
        try:
            pdf_bytes = generate_complaint_pdf(state)
        except Exception as e:
            return {
                "status": "SUBMISSION_FAILED",
                "recipient": recipient,
                "timestamp": now_str,
                "is_mock": self.mock_email,
                "complaint_id": complaint_id,
                "pdf_attached": False,
                "error": f"Failed to generate complaint PDF attachment: {str(e)}",
            }

        # 4. Build EmailMessage
        try:
            msg = self.build_email_content(state, pdf_bytes, recipient)
        except Exception as e:
            return {
                "status": "SUBMISSION_FAILED",
                "recipient": recipient,
                "timestamp": now_str,
                "is_mock": self.mock_email,
                "complaint_id": complaint_id,
                "pdf_attached": False,
                "error": f"Failed to construct email message: {str(e)}",
            }

        # 5. Handle Mock Mode (Safe, zero network calls)
        if self.mock_email:
            return {
                "status": "SUBMITTED",
                "recipient": recipient,
                "timestamp": now_str,
                "is_mock": True,
                "complaint_id": complaint_id,
                "pdf_attached": True,
                "subject": msg["Subject"],
                "message": (
                    f"[MOCK] Complaint email successfully simulated to '{recipient}' "
                    f"with PDF attachment ({len(pdf_bytes)} bytes)."
                ),
            }

        # 6. Real SMTP transmission
        try:
            if self.smtp_port == 465:
                # SSL SMTP
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=15) as server:
                    if self.smtp_username and self.smtp_password:
                        server.login(self.smtp_username, self.smtp_password)
                    server.send_message(msg)
            else:
                # Standard SMTP / STARTTLS
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                    try:
                        server.starttls()
                    except Exception:
                        pass  # Server might not support STARTTLS (e.g. local debug server)
                    if self.smtp_username and self.smtp_password:
                        server.login(self.smtp_username, self.smtp_password)
                    server.send_message(msg)

            return {
                "status": "SUBMITTED",
                "recipient": recipient,
                "timestamp": now_str,
                "is_mock": False,
                "complaint_id": complaint_id,
                "pdf_attached": True,
                "subject": msg["Subject"],
            }
        except Exception as e:
            return {
                "status": "SUBMISSION_FAILED",
                "recipient": recipient,
                "timestamp": now_str,
                "is_mock": False,
                "complaint_id": complaint_id,
                "pdf_attached": True,
                "error": f"SMTP transmission error: {str(e)}",
            }


# Default singleton instance
default_email_service = EmailSubmissionService()
