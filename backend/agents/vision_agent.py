"""
Vision Agent

Analyses an uploaded road image to:
  - Detect whether a pothole is present
  - Estimate severity (none / low / moderate / severe)
  - Return model-estimated confidence (NOT a calibrated probability)
  - Describe the visual evidence observed

Design notes:
  - Handles local file paths by converting them to base64 data URLs for multimodal LLMs.
  - The LLM is injected at call time so that unit tests can pass a mock.
  - If no LLM is provided (e.g. during offline testing without API key), the agent raises
    a clear RuntimeError rather than silently returning invented data.
  - Structured output is enforced via a Pydantic model so the graph state
    always receives a typed, validated result.
"""
import os
import json
import base64
import mimetypes
from typing import Optional, Any

from pydantic import BaseModel, Field


# ── Local Image to Base64 Data URL Helper ────────────────────────────────────

def ensure_image_data_url(image_source: str) -> str:
    """
    Converts a local file path into a base64 data URL if it is not already a remote URL.

    Parameters
    ----------
    image_source : str
        Local file path, data URL (data:image/...), or remote HTTP(S) URL.

    Returns
    -------
    str
        A URL format acceptable by multimodal LangChain chat models.
    """
    if not image_source:
        return image_source

    # If it's already a data URL or remote web URL, pass it as-is
    if image_source.startswith("data:image/") or image_source.startswith("http://") or image_source.startswith("https://"):
        return image_source

    # Treat as local filesystem path
    abs_path = os.path.abspath(image_source)
    if not os.path.exists(abs_path):
        # If relative or unverified path, try checking current working directory
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Local image file not found at: {image_source}")
        abs_path = os.path.abspath(image_source)

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(abs_path)
    if not mime_type or not mime_type.startswith("image/"):
        ext = os.path.splitext(abs_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        elif ext == ".webp":
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"

    with open(abs_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# ── Structured output schema ────────────────────────────────────────────────

class VisionAgentOutput(BaseModel):
    pothole_detected: bool = Field(
        description="True if a pothole or significant road damage is visible."
    )
    severity: str = Field(
        description="Estimated severity: 'none', 'low', 'moderate', or 'severe'."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Model-estimated confidence in the detection (0.0–1.0). "
            "This is NOT a statistically calibrated probability."
        )
    )
    visual_evidence: str = Field(
        description="Plain-English description of what was observed in the image."
    )

    def validate_severity(self) -> None:
        valid = {"none", "low", "moderate", "severe"}
        if self.severity not in valid:
            raise ValueError(f"severity must be one of {valid}, got '{self.severity}'")


# ── Prompt template ──────────────────────────────────────────────────────────

VISION_SYSTEM_PROMPT = """You are a road-condition analysis assistant for the RoadWatch AI demonstration system.

Your task is to analyse a road photograph and determine whether a pothole or significant
road surface damage is visible.

Respond ONLY with a valid JSON object matching this schema exactly:
{
  "pothole_detected": <true|false>,
  "severity": "<none|low|moderate|severe>",
  "confidence": <float between 0.0 and 1.0>,
  "visual_evidence": "<plain-English description of what you observed>"
}

Severity scale:
  none     – no visible pothole or road damage
  low      – minor surface cracking or shallow depression, low risk
  moderate – visible pothole with clear edges, potential hazard
  severe   – deep or wide pothole, serious safety risk

Confidence:
  Report your model-estimated confidence in the detection result.
  This is NOT a statistically calibrated probability — it is your best estimate.
  A value of 1.0 means you are certain; 0.0 means you cannot determine anything.

Do NOT include any additional text outside the JSON object.
"""

VISION_USER_PROMPT = "Analyse the following road image and return your assessment as JSON."


# ── Agent function ───────────────────────────────────────────────────────────

def run_vision_agent(image_url: str, llm: Any) -> dict:
    """
    Run the Vision Agent on a road image.

    Parameters
    ----------
    image_url : str
        A local image file path, base64 data URL, or remote HTTP URL.
    llm : Any
        A LangChain chat model instance that supports multimodal (vision) input.
        Must implement the `invoke()` interface.

    Returns
    -------
    dict
        A dict matching the VisionResult TypedDict shape, ready to be merged
        into GraphState["vision_result"].

    Raises
    ------
    RuntimeError
        If no LLM is provided.
    ValueError
        If the LLM response cannot be parsed into VisionAgentOutput.
    """
    if llm is None:
        raise RuntimeError(
            "Vision Agent requires an LLM instance, but none was provided. "
            "Please ensure GEMINI_API_KEY is configured in your .env file."
        )

    from langchain_core.messages import HumanMessage, SystemMessage

    formatted_image_url = ensure_image_data_url(image_url)

    messages = [
        SystemMessage(content=VISION_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": VISION_USER_PROMPT},
                {"type": "image_url", "image_url": {"url": formatted_image_url}},
            ]
        ),
    ]

    response = llm.invoke(messages)
    raw_content = response.content

    # ChatGoogleGenerativeAI returns content as a list of dicts (blocks),
    # while ChatOpenAI returns a plain string.  Handle both.
    if isinstance(raw_content, list):
        # Extract the first text block from the list
        text_parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        ]
        raw_text = "\n".join(p for p in text_parts if p).strip()
    else:
        raw_text = str(raw_content).strip()

    # Strip markdown code fences if the model wraps the JSON
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:-1])

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Vision Agent: LLM returned non-JSON content.\n"
            f"Raw response:\n{raw_text}\nError: {e}"
        )

    output = VisionAgentOutput(**parsed)
    output.validate_severity()

    return {
        "pothole_detected": output.pothole_detected,
        "severity": output.severity,
        "confidence": output.confidence,
        "visual_evidence": output.visual_evidence,
    }
