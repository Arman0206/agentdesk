import threading
from types import SimpleNamespace
import pytest
import app.llm.tenacity_lite as tl
from app.llm.gemini import GeminiLLM, _is_transient


def _llm(models):
    llm = GeminiLLM.__new__(GeminiLLM)             # skip real client creation
    llm.model, llm._min, llm._last, llm._lock = "m", 0.0, 0.0, threading.Lock()
    llm._client = SimpleNamespace(models=models)
    return llm


def _ok():
    return SimpleNamespace(text="hi", usage_metadata=SimpleNamespace(
        prompt_token_count=3, candidates_token_count=2, thoughts_token_count=5))


def test_classifies_errors():
    assert _is_transient(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert _is_transient(RuntimeError("Server disconnected without sending a response."))
    assert not _is_transient(RuntimeError("400 API key not valid"))


def test_dropped_connection_is_retried_and_thinking_tokens_counted(monkeypatch):
    monkeypatch.setattr(tl.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Models:
        def generate_content(self, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Server disconnected without sending a response.")
            return _ok()

    res = _llm(Models()).generate("x")
    assert res.text == "hi" and calls["n"] == 3
    assert res.completion_tokens == 7               # 2 visible + 5 thinking


def test_bad_key_fails_fast_without_retry(monkeypatch):
    monkeypatch.setattr(tl.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Models:
        def generate_content(self, **kw):
            calls["n"] += 1
            raise RuntimeError("400 API key not valid")

    with pytest.raises(RuntimeError):
        _llm(Models()).generate("x")
    assert calls["n"] == 1


def test_daily_quota_fails_fast_with_clear_error(monkeypatch):
    from app.llm.gemini import DailyQuotaExceeded
    monkeypatch.setattr(tl.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Models:
        def generate_content(self, **kw):
            calls["n"] += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED ... GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(DailyQuotaExceeded):
        _llm(Models()).generate("x")
    assert calls["n"] == 1                          # no pointless retries against a daily cap


def test_cache_replays_without_calling_model_and_keeps_measurements(tmp_path):
    from app.llm.base import LLMResult
    from app.llm.cache import CachingLLM
    calls = {"n": 0}

    class Inner:
        model = "m"
        def generate(self, prompt, **kw):
            calls["n"] += 1
            return LLMResult(text="answer", model="m", prompt_tokens=10, completion_tokens=20, latency_s=3.5)

    c = CachingLLM(Inner(), str(tmp_path))
    first = c.generate("q", task="draft")
    again = c.generate("q", task="draft")
    other = c.generate("different", task="draft")
    assert calls["n"] == 2 and c.hits == 1 and c.misses == 2
    assert again.text == "answer" and again.latency_s == 3.5 and again.completion_tokens == 20
