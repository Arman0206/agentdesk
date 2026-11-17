from types import SimpleNamespace
import pytest
import app.llm.tenacity_lite as tl
from app.llm.azure_openai import AzureOpenAILLM, _azure_transient


class APIConnectionError(Exception):             # same class name the openai SDK uses
    pass


def _llm(completions, effort="low"):
    llm = AzureOpenAILLM.__new__(AzureOpenAILLM)   # skip real client / credentials
    llm.model, llm._effort = "gpt-5-mini", effort
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm


def _ok(text="ok", prompt=4, completion=30):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


def test_retries_dropped_connection_and_uses_reasoning_inclusive_completion_tokens(monkeypatch):
    monkeypatch.setattr(tl.time, "sleep", lambda s: None)
    calls = {"n": 0, "kw": None}

    class Completions:
        def create(self, **kw):
            calls["n"] += 1
            calls["kw"] = kw
            if calls["n"] == 1:
                raise APIConnectionError("Connection error.")
            return _ok()

    r = _llm(Completions()).generate("hi", system="s")
    assert r.text == "ok" and calls["n"] == 2 and r.completion_tokens == 30 and r.prompt_tokens == 4
    assert calls["kw"]["reasoning_effort"] == "low" and "temperature" not in calls["kw"]
    assert calls["kw"]["messages"][0] == {"role": "system", "content": "s"}


def test_non_reasoning_model_sends_temperature_instead():
    seen = {}

    class Completions:
        def create(self, **kw):
            seen.update(kw)
            return _ok()

    _llm(Completions(), effort="").generate("hi")
    assert seen["temperature"] == 0.2 and "reasoning_effort" not in seen


def test_json_mode_sets_response_format_and_mentions_json():
    seen = {}

    class Completions:
        def create(self, **kw):
            seen.update(kw)
            return _ok("{}")

    _llm(Completions()).generate("classify this ticket", json_mode=True)
    assert seen["response_format"] == {"type": "json_object"}
    assert "json" in " ".join(m["content"] for m in seen["messages"]).lower()


def test_bad_request_fails_fast_and_empty_answer_is_empty_text(monkeypatch):
    monkeypatch.setattr(tl.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Bad:
        def create(self, **kw):
            calls["n"] += 1
            raise RuntimeError("Error code: 400 - invalid request")

    with pytest.raises(RuntimeError):
        _llm(Bad()).generate("x")
    assert calls["n"] == 1
    assert not _azure_transient(RuntimeError("Error code: 401 - access denied"))
    assert _azure_transient(RuntimeError("Error code: 429 - rate limit"))

    class Filtered:
        def create(self, **kw):
            return _ok(text=None)

    assert _llm(Filtered()).generate("x").text == ""


def test_azure_requires_endpoint_key_and_deployment():
    for args in [("", "k", "d"), ("https://x", "", "d"), ("https://x", "k", "")]:
        with pytest.raises(ValueError):
            AzureOpenAILLM(*args)
