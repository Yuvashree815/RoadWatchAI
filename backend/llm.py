"""
RoadWatch AI — LLM Configuration & Initialization (Google Gemini)

Initializes LangChain's ChatGoogleGenerativeAI for multimodal vision analysis using
environment variables loaded via python-dotenv.
Supports Mock LLM mode (MOCK_LLM=true) to completely prevent API calls and save quota during testing.
"""
import json
from typing import Optional, Any
from backend.config import settings

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


class MockChatMultimodal:
    """
    Deterministic mock for multimodal LLM vision calls during tests/demo mode.
    Implements invoke() interface returning structured JSON matching VisionAgentOutput.
    """
    def __init__(self, model_name: str = "mock-gemini-3.6-flash"):
        self.model_name = model_name

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage
        mock_payload = {
            "pothole_detected": True,
            "severity": "moderate",
            "confidence": 0.95,
            "visual_evidence": (
                "[MOCK DEMO] Roadway surface shows a moderate-depth pothole with "
                "visible asphalt degradation and edge fracturing."
            ),
        }
        return AIMessage(content=json.dumps(mock_payload))


def get_gemini_api_key() -> Optional[str]:
    """
    Retrieves the Gemini API key from the environment.
    Checks GEMINI_API_KEY first, with GOOGLE_API_KEY as fallback.
    """
    return settings.get_effective_gemini_key()


def get_gemini_model() -> str:
    """Retrieves the configured Gemini model name from the environment."""
    return settings.GEMINI_MODEL


def is_llm_configured() -> bool:
    """Checks if the Gemini API key is available or mock mode is active."""
    return settings.MOCK_LLM or (get_gemini_api_key() is not None)


def get_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    force_mock: Optional[bool] = None,
):
    """
    Initializes and returns a multimodal chat model instance.

    If force_mock is True or MOCK_LLM=true in settings, returns a MockChatMultimodal.
    Returns None if GEMINI_API_KEY is not configured and MOCK_LLM is false,
    allowing offline testing with pre-populated states.

    Parameters
    ----------
    model : str | None
        Model identifier. Defaults to GEMINI_MODEL env var or 'gemini-3.6-flash'.
    api_key : str | None
        Gemini API key. Defaults to GEMINI_API_KEY / GOOGLE_API_KEY env var.
    temperature : float
        Sampling temperature (default 0.0 for deterministic analysis).
    force_mock : bool | None
        Explicitly override mock behavior.

    Returns
    -------
    ChatGoogleGenerativeAI | MockChatMultimodal | None
    """
    use_mock = force_mock if force_mock is not None else settings.MOCK_LLM
    if use_mock:
        return MockChatMultimodal(model_name=model or get_gemini_model())

    resolved_key = api_key or get_gemini_api_key()
    if not resolved_key:
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        resolved_model = model or get_gemini_model()
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            api_key=resolved_key,
            temperature=temperature,
        )
    except Exception as e:
        print(f"Warning: Failed to initialize ChatGoogleGenerativeAI: {e}")
        return None
