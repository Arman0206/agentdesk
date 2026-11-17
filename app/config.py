"""Central configuration. Everything is overridable via env vars (12-factor)."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # provider: "mock" (offline, deterministic) | "gemini" (Gemini API / AI Studio key, free tier)
    #           | "vertex" (Gemini on Vertex AI, needs a billed GCP project)
    llm_provider: str = "mock"
    embedding_provider: str = "hash"  # "hash" | "gemini" | "vertex"
    gemini_api_key: str = ""          # from https://aistudio.google.com/app/apikey -- never commit it
    gemini_embedding_model: str = "gemini-embedding-001"  # verify the current name in AI Studio
    gemini_min_interval_s: float = 4.0  # client-side throttle to stay under free-tier requests/minute
    gcp_project: str = ""
    gcp_location: str = "us-central1"
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-005"

    # retrieval
    chroma_dir: str = ".chroma"
    top_k: int = 5
    rrf_k: int = 60
    min_score: float = 0.0

    # agent loop
    max_react_steps: int = 4
    max_revisions: int = 2

    # service
    api_key: str = ""            # if empty, auth is disabled (local dev only)
    rate_limit_per_min: int = 30  # per client IP, protects the free Gemini quota
    data_dir: str = "data"
    trace_file: str = "traces.jsonl"
    orders_db: str = "data/orders.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
