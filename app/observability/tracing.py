"""
Lightweight tracing with LLM-native metrics.

Every agent node runs inside a span. LLM calls are recorded on the active span with
prompt/completion tokens, latency, tokens/sec and cost-per-request. A finished trace is
exported as one JSON line (ingestible by Cloud Logging / BigQuery) and is also returned
by the API so any answer can be debugged after the fact.
"""
from __future__ import annotations
import contextvars, json, threading, time, uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any

_current: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("trace", default=None)
_stack: contextvars.ContextVar[tuple] = contextvars.ContextVar("span_stack", default=())
_write_lock = threading.Lock()


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    start: float
    duration_s: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[dict] = field(default_factory=list)


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    spans: list[Span] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    # ---- aggregates ----
    def _calls(self):
        return [c for s in self.spans for c in s.llm_calls]

    def summary(self) -> dict:
        calls = self._calls()
        pt = sum(c["prompt_tokens"] for c in calls)
        ct = sum(c["completion_tokens"] for c in calls)
        llm_time = sum(c["latency_s"] for c in calls)
        return {
            "trace_id": self.trace_id,
            "total_latency_s": round(time.perf_counter() - self.started, 4),
            "llm_calls": len(calls),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
            "tokens_per_sec": round(ct / llm_time, 2) if llm_time > 0 else 0.0,
            "per_node_latency_s": {s.name: round(s.duration_s, 4) for s in self.spans},
        }

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "spans": [asdict(s) for s in self.spans]}


def start_trace() -> Trace:
    t = Trace()
    _current.set(t)
    _stack.set(())
    return t


def current_trace() -> Trace | None:
    return _current.get()


@contextmanager
def span(name: str, **attrs):
    tr = _current.get()
    if tr is None:  # tracing disabled -> no-op
        yield None
        return
    stack = _stack.get()
    sp = Span(name=name, span_id=uuid.uuid4().hex[:8],
              parent_id=stack[-1] if stack else None, start=time.perf_counter(), attrs=dict(attrs))
    token = _stack.set(stack + (sp.span_id,))
    try:
        yield sp
    except Exception as e:  # record, re-raise
        sp.attrs["error"] = repr(e)
        raise
    finally:
        sp.duration_s = time.perf_counter() - sp.start
        tr.spans.append(sp)
        _stack.reset(token)


def record_llm_call(sp: Span | None, task: str, res) -> None:
    if sp is None:
        return
    sp.llm_calls.append({
        "task": task, "model": res.model,
        "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
        "latency_s": round(res.latency_s, 4), "tokens_per_sec": round(res.tokens_per_sec, 2),
        "cost_usd": round(res.cost_usd, 6),
    })


def export_trace(tr: Trace, path: str) -> None:
    with _write_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(tr.to_dict()) + "\n")
