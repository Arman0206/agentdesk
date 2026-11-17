from app.config import get_settings
from app.llm.base import LLMClient


def get_llm() -> LLMClient:
    s = get_settings()
    if s.llm_provider == "gemini":
        from app.llm.gemini import GeminiLLM
        return GeminiLLM(s.gemini_api_key, s.llm_model, s.gemini_min_interval_s)
    if s.llm_provider == "vertex":
        from app.llm.vertex import VertexLLM
        return VertexLLM(s.gcp_project, s.gcp_location, s.llm_model)
    from app.llm.mock import MockLLM
    return MockLLM()
