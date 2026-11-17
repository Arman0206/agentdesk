from types import SimpleNamespace
import pytest
import app.llm.tenacity_lite as tl
from app.llm.vertex import VertexLLM


def _llm(models):
    llm = VertexLLM.__new__(VertexLLM)             # skip real client / credentials
    llm.model, llm._client = "m", SimpleNamespace(models=models)
    return llm


def test_vertex_counts_thinking_tokens_and_retries_dropped_connection(monkeypatch):
    monkeypatch.setattr(tl.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Models:
        def generate_content(self, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("503 UNAVAILABLE")
            return SimpleNamespace(text="ok", usage_metadata=SimpleNamespace(
                prompt_token_count=4, candidates_token_count=1, thoughts_token_count=9))

    r = _llm(Models()).generate("x")
    assert r.text == "ok" and calls["n"] == 2 and r.completion_tokens == 10


def test_vertex_requires_project():
    with pytest.raises(ValueError):
        VertexLLM("", "us-central1", "m")
