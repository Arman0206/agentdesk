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
