"""
RoadWatch AI — Centralized Application Configuration

Manages environment variables dynamically for:
- Google Gemini LLM
- LangSmith Observability & Tracing
- Mock Modes (LLM, Email)
- Email Submission (SMTP)
- Supabase & ChromaDB
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


def get_bool_env(name: str, default: bool = False) -> bool:
    """Helper to parse boolean environment variables dynamically from os.environ."""
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("true", "1", "yes", "on")


class Settings:
    """
    Dynamic settings accessor reading from os.environ on each property access.
    Allows monkeypatching, runtime configuration changes, and testing without restart.
    """

    # ── Application Modes ─────────────────────────────────────────────────────
    @property
    def DEMO_MODE(self) -> bool:
        return get_bool_env("DEMO_MODE", default=True)

    @property
    def MOCK_LLM(self) -> bool:
        return get_bool_env("MOCK_LLM", default=False)

    @property
    def MOCK_EMAIL(self) -> bool:
        return get_bool_env("MOCK_EMAIL", default=True)

    # ── LLM (Google Gemini) ───────────────────────────────────────────────────
    @property
    def GEMINI_API_KEY(self) -> Optional[str]:
        return os.environ.get("GEMINI_API_KEY", "").strip() or None

    @property
    def GOOGLE_API_KEY(self) -> Optional[str]:
        return os.environ.get("GOOGLE_API_KEY", "").strip() or None

    @property
    def GEMINI_MODEL(self) -> str:
        return os.environ.get("GEMINI_MODEL", "gemini-3.6-flash").strip() or "gemini-3.6-flash"

    # ── LangSmith Tracing ─────────────────────────────────────────────────────
    @property
    def LANGSMITH_TRACING(self) -> bool:
        return get_bool_env("LANGSMITH_TRACING", False) or get_bool_env("LANGCHAIN_TRACING_V2", False)

    @property
    def LANGSMITH_API_KEY(self) -> Optional[str]:
        return os.environ.get("LANGSMITH_API_KEY", "").strip() or os.environ.get("LANGCHAIN_API_KEY", "").strip() or None

    @property
    def LANGSMITH_PROJECT(self) -> str:
        return os.environ.get("LANGSMITH_PROJECT", "").strip() or os.environ.get("LANGCHAIN_PROJECT", "RoadWatchAI").strip() or "RoadWatchAI"

    @property
    def LANGSMITH_ENDPOINT(self) -> str:
        return os.environ.get("LANGSMITH_ENDPOINT", "").strip() or os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com").strip() or "https://api.smith.langchain.com"

    # ── Email Submission (SMTP) ───────────────────────────────────────────────
    @property
    def EMAIL_ENABLED(self) -> bool:
        return get_bool_env("EMAIL_ENABLED", default=False)

    @property
    def EMAIL_PROVIDER(self) -> str:
        return os.environ.get("EMAIL_PROVIDER", "smtp").strip() or "smtp"

    @property
    def SMTP_HOST(self) -> str:
        return os.environ.get("SMTP_HOST", "localhost").strip() or "localhost"

    @property
    def SMTP_PORT(self) -> int:
        try:
            return int(os.environ.get("SMTP_PORT", "587").strip() or "587")
        except ValueError:
            return 587

    @property
    def SMTP_USERNAME(self) -> Optional[str]:
        return os.environ.get("SMTP_USERNAME", "").strip() or None

    @property
    def SMTP_PASSWORD(self) -> Optional[str]:
        return os.environ.get("SMTP_PASSWORD", "").strip() or None

    @property
    def SMTP_FROM(self) -> str:
        return os.environ.get("SMTP_FROM", "roadwatch-noreply@demo.local").strip() or "roadwatch-noreply@demo.local"

    @property
    def DEMO_COMPLAINT_EMAIL(self) -> str:
        return os.environ.get("DEMO_COMPLAINT_EMAIL", "demo-authority@roadwatch.local").strip() or "demo-authority@roadwatch.local"

    # ── Database & Vector Store ───────────────────────────────────────────────
    @property
    def SUPABASE_URL(self) -> Optional[str]:
        return os.environ.get("SUPABASE_URL", "").strip() or None

    @property
    def SUPABASE_KEY(self) -> Optional[str]:
        return os.environ.get("SUPABASE_KEY", "").strip() or None

    @property
    def CHROMA_PERSIST_DIRECTORY(self) -> str:
        return os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_data").strip() or "./chroma_data"

    def get_effective_gemini_key(self) -> Optional[str]:
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY

    def is_langsmith_enabled(self) -> bool:
        return self.LANGSMITH_TRACING and bool(self.LANGSMITH_API_KEY)


settings = Settings()
