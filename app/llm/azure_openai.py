"""
OpenAI models on Azure (Microsoft Foundry / Azure OpenAI), same interface as the Gemini / Vertex clients.

* the deployment name (LLM_MODEL) is what the API calls; it is also the name used for the cost table
* retry with backoff for rate limits (429), 5xx and dropped connections; bad requests fail fast
* Azure content-filter rejections (e.g. a prompt-injection ticket) return an empty answer, which the agents
  treat as fail-closed (escalate) instead of crashing the request
* reasoning tokens are already included in `usage.completion_tokens`, so cost / tokens-per-sec are not
  under-reported (same accounting rule as the Gemini thinking tokens)
* reasoning models (gpt-5 family) reject a custom temperature and accept `reasoning_effort`; set
  AZURE_REASONING_EFFORT="" for a non-reasoning model (e.g. gpt-4.1-mini) to send temperature=0.2 instead
"""
from __future__ import annotations
import logging, time
from app.llm.base import LLMResult
from app.llm.gemini import _is_transient
from app.llm.tenacity_lite import retry

log = logging.getLogger(__name__)
_TRANSIENT_TYPES = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}


def _azure_transient(e: Exception) -> bool:
    """Worth retrying: rate limit, 5xx, timeouts, dropped connections. Not: bad key, bad request, content filter."""
    return any(c.__name__ in _TRANSIENT_TYPES for c in type(e).__mro__) or _is_transient(e)


def _is_content_filter(e: Exception) -> bool:
    """Azure's input guardrails (e.g. prompt-injection / jailbreak shields) reject the request with a 400."""
    t = f"{getattr(e, 'code', '')} {e}".lower()
    return "content_filter" in t or "content management policy" in t or "responsibleaipolicyviolation" in t


class AzureOpenAILLM:
    def __init__(self, endpoint: str, api_key: str, deployment: str,
                 api_version: str = "2025-04-01-preview", reasoning_effort: str = "low",
                 timeout_s: float = 120.0):
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set")
        if not deployment:
            raise ValueError("LLM_MODEL must be the Azure deployment name (e.g. gpt-5-mini)")
        from openai import AzureOpenAI  # lazy import: offline/CI runs never need it
        # max_retries=0: our own retry decorator handles backoff, so attempts are not multiplied
        self._client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version,
                                   max_retries=0, timeout=timeout_s)
        self.model, self._effort = deployment, (reasoning_effort or "").strip()

    @retry(attempts=5, base_delay=3.0, retry_if=_azure_transient)
    def generate(self, prompt: str, *, system: str = "", task: str = "generic",
                 json_mode: bool = False) -> LLMResult:
        if json_mode and "json" not in f"{system} {prompt}".lower():
            system = f"{system}\nRespond with valid JSON only.".strip()   # Azure requires the word "JSON"
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        kw: dict = {}
        if self._effort:
            kw["reasoning_effort"] = self._effort        # reasoning model: default temperature only
        else:
            kw["temperature"] = 0.2
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(model=self.model, messages=messages, **kw)
        except Exception as e:
            if not _is_content_filter(e):
                raise
            # The platform refused the input. Return an empty answer: every agent already treats empty /
            # unparseable output as "fail closed" and escalates to a human, which is the safe outcome.
            log.warning("Azure content filter blocked the request (task=%s); failing closed", task)
            return LLMResult(text="", model=self.model, prompt_tokens=0, completion_tokens=0,
                             latency_s=time.perf_counter() - t0)
        dt = time.perf_counter() - t0
        u = getattr(resp, "usage", None)
        text = (resp.choices[0].message.content or "") if resp.choices else ""   # empty if filtered
        return LLMResult(text=text, model=self.model,
                         prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                         completion_tokens=getattr(u, "completion_tokens", 0) or 0, latency_s=dt)
