"""In-process rolling aggregates exposed at /metrics (swap for OpenTelemetry/Prometheus in prod)."""
from __future__ import annotations
import statistics, threading
from collections import deque


class Metrics:
    def __init__(self, window: int = 500):
        self._lock = threading.Lock()
        self._lat = deque(maxlen=window)
        self._cost = deque(maxlen=window)
        self._tok = deque(maxlen=window)
        self.outcomes: dict[str, int] = {}

    def observe(self, summary: dict, outcome: str) -> None:
        with self._lock:
            self._lat.append(summary["total_latency_s"])
            self._cost.append(summary["cost_usd"])
            self._tok.append(summary["prompt_tokens"] + summary["completion_tokens"])
            self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1

    @staticmethod
    def _pct(xs, p):
        if not xs:
            return 0.0
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]

    def snapshot(self) -> dict:
        with self._lock:
            lat, cost, tok = list(self._lat), list(self._cost), list(self._tok)
            return {
                "requests": sum(self.outcomes.values()),
                "outcomes": dict(self.outcomes),
                "latency_p50_s": round(self._pct(lat, 0.5), 4),
                "latency_p95_s": round(self._pct(lat, 0.95), 4),
                "avg_cost_usd": round(statistics.fmean(cost), 6) if cost else 0.0,
                "avg_tokens": round(statistics.fmean(tok), 1) if tok else 0.0,
            }


METRICS = Metrics()
