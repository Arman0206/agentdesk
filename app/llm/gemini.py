"""
Gemini via the Gemini Developer API (Google AI Studio key). Works on the free tier, no billing needed.

* client-side throttle  -> stays under the free tier's requests-per-minute limit
* retry with backoff    -> only for rate-limit / transient errors (429, 5xx); bad keys fail fast
* thinking tokens are counted as output tokens, so cost / tokens-per-sec are not under-reported
"""
from __future__ import annotations
import threading, time
from app.llm.base import LLMResult
from app.llm.tenacity_lite import retry

_TRANSIENT = ("429", "500", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE",
              "Server disconnected", "disconnected", "Connection reset", "ConnectionReset", "timed out",
              "timeout", "Timeout", "ConnectError", "RemoteProtocolError", "Temporary failure")


def _is_transient(e: Exception) -> bool:
    """Rate limits, 5xx and dropped connections are worth retrying; bad keys / bad requests are not."""
    return any(t in f"{type(e).__name__} {e}" for t in _TRANSIENT)


class GeminiLLM:
    def __init__(self, api_key: str, model: str, min_interval_s: float = 4.0):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set (create one at https://aistudio.google.com/app/apikey)")
        from google import genai  # lazy import: offline/CI runs never need it
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self._min, self._last, self._lock = min_interval_s, 0.0, threading.Lock()

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last + self._min - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    @retry(attempts=5, base_delay=3.0, retry_if=_is_transient)
    def generate(self, prompt: str, *, system: str = "", task: str = "generic",
                 json_mode: bool = False) -> LLMResult:
        from google.genai import types
        self._throttle()
        cfg = types.GenerateContentConfig(
            system_instruction=system or None, temperature=0.2,
            response_mime_type="application/json" if json_mode else None)
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        dt = time.perf_counter() - t0
        u = resp.usage_metadata
        out = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
        return LLMResult(text=resp.text or "", model=self.model,
                         prompt_tokens=getattr(u, "prompt_token_count", 0) or 0,
                         completion_tokens=out, latency_s=dt)
