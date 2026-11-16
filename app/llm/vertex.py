"""Gemini on Vertex AI via the google-genai SDK (Application Default Credentials on Cloud Run)."""
from __future__ import annotations
import time
from app.llm.base import LLMResult
from app.llm.tenacity_lite import retry


class VertexLLM:
    def __init__(self, project: str, location: str, model: str):
        from google import genai  # lazy import: not needed for offline/CI runs
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self.model = model

    @retry(attempts=3, base_delay=0.5)
    def generate(self, prompt: str, *, system: str = "", task: str = "generic",
                 json_mode: bool = False) -> LLMResult:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=0.2,
            response_mime_type="application/json" if json_mode else None,
        )
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        dt = time.perf_counter() - t0
        u = resp.usage_metadata
        return LLMResult(
            text=resp.text or "",
            model=self.model,
            prompt_tokens=getattr(u, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(u, "candidates_token_count", 0) or 0,
            latency_s=dt,
        )
