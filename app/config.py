"""Central configuration. Everything is overridable via env vars (12-factor)."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # provider: "mock" (offline, deterministic, used in CI) | "vertex" (Gemini on Vertex AI)
    llm_provider: str = "mock"
    embedding_provider: str = "hash"  # "hash" | "vertex"
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
    trace_file: str = "traces.jsonl"
    orders_db: str = "data/orders.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
