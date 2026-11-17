from app.config import get_settings, reveal
from app.llm.base import LLMClient


def get_llm() -> LLMClient:
    s = get_settings()
    if s.llm_provider == "gemini":
        from app.llm.gemini import GeminiLLM
        llm = GeminiLLM(reveal(s.gemini_api_key), s.llm_model, s.gemini_min_interval_s)
    elif s.llm_provider == "azure":
        from app.llm.azure_openai import AzureOpenAILLM
        llm = AzureOpenAILLM(s.azure_openai_endpoint, reveal(s.azure_openai_api_key), s.llm_model,
                             s.azure_openai_api_version, s.azure_reasoning_effort)
    elif s.llm_provider == "vertex":
        from app.llm.vertex import VertexLLM
        llm = VertexLLM(s.gcp_project, s.gcp_location, s.llm_model)
    else:
        from app.llm.mock import MockLLM
        return MockLLM()                     # offline: nothing to cache
    if s.llm_cache:
        from app.llm.cache import CachingLLM
        llm = CachingLLM(llm, s.llm_cache_dir)
    return llm
