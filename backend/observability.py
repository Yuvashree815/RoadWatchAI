"""
RoadWatch AI — Observability & Tracing (LangSmith)

Configures LangSmith tracing for LangGraph multi-agent execution,
individual agent nodes, and LLM calls.
Ensures tracing is optional and safe, with no leakage of sensitive credentials.
"""
import os
from typing import Optional, Dict, Any
from backend.config import settings


def configure_langsmith() -> bool:
    """
    Sets up the environment variables required by LangChain / LangSmith SDK
    if LangSmith tracing is enabled and configured.

    Returns
    -------
    bool
        True if LangSmith is enabled and configured with an API key, False otherwise.
    """
    # Check if tracing is enabled
    is_tracing = settings.LANGSMITH_TRACING
    api_key = settings.LANGSMITH_API_KEY

    if is_tracing and api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
        os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
        return True
    else:
        # If disabled or key not present, ensure tracing is not forced on
        if not is_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ["LANGSMITH_TRACING"] = "false"
        return False


def get_sanitized_config() -> Dict[str, Any]:
    """
    Returns a sanitized dictionary of observability and system configuration,
    suitable for logging or API diagnostics without exposing secrets.
    """
    return {
        "langsmith_tracing": settings.LANGSMITH_TRACING,
        "langsmith_configured": bool(settings.LANGSMITH_API_KEY),
        "langsmith_project": settings.LANGSMITH_PROJECT,
        "langsmith_endpoint": settings.LANGSMITH_ENDPOINT,
        "mock_llm": settings.MOCK_LLM,
        "mock_email": settings.MOCK_EMAIL,
        "email_enabled": settings.EMAIL_ENABLED,
        "demo_mode": settings.DEMO_MODE,
        "demo_complaint_email": settings.DEMO_COMPLAINT_EMAIL,
        "smtp_host": settings.SMTP_HOST,
        "smtp_port": settings.SMTP_PORT,
        "smtp_from": settings.SMTP_FROM,
        "gemini_model": settings.GEMINI_MODEL,
        "gemini_configured": bool(settings.get_effective_gemini_key()),
    }


# Auto-configure on module import
configure_langsmith()
