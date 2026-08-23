"""
RoadWatch AI — LLM Configuration & Initialization (Google Gemini)

Initializes LangChain's ChatGoogleGenerativeAI for multimodal vision analysis using
environment variables loaded via python-dotenv.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root if present
load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


def get_gemini_api_key() -> Optional[str]:
    """
    Retrieves the Gemini API key from the environment.
    Checks GEMINI_API_KEY first, with GOOGLE_API_KEY as fallback.
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        key = os.environ.get("GOOGLE_API_KEY", "").strip()
    return key if key else None


def get_gemini_model() -> str:
    """Retrieves the configured Gemini model name from the environment."""
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def is_llm_configured() -> bool:
    """Checks if the Gemini API key is available."""
    return get_gemini_api_key() is not None


def get_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
):
    """
    Initializes and returns a ChatGoogleGenerativeAI model instance for multimodal analysis.

    Returns None if GEMINI_API_KEY is not configured, allowing offline / mock testing.

    Parameters
    ----------
    model : str | None
        Model identifier. Defaults to GEMINI_MODEL env var or 'gemini-3.6-flash'.
    api_key : str | None
        Gemini API key. Defaults to GEMINI_API_KEY / GOOGLE_API_KEY env var.
    temperature : float
        Sampling temperature (default 0.0 for deterministic analysis).

    Returns
    -------
    ChatGoogleGenerativeAI | None
    """
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
