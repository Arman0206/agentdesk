"""
FastAPI service.

  GET  /health            liveness + provider info
  POST /v1/resolve        run the multi-agent workflow on one ticket
  GET  /v1/runs/{run_id}  full per-agent trace for a past request (debugging)
  GET  /v1/metrics        rolling latency / cost / outcome aggregates

Protections: optional API key (X-API-Key), per-IP sliding-window rate limit (protects the free
Gemini quota), strict input validation. Demo simplification: customer_id/tenant come from the
request body -- in a real deployment they must come from the authenticated identity, never the client.
"""
from __future__ import annotations
import secrets, threading, time
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from app.config import get_settings, reveal
from app.observability.metrics import METRICS
from app.service import Service


class ResolveRequest(BaseModel):
    ticket: str = Field(min_length=1, max_length=4000)
    customer_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    tenant: str = Field(default="default", max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class Citation(BaseModel):
    id: str
    source: str
    section: str = ""


class ResolveResponse(BaseModel):
    run_id: str
    outcome: str            # auto_send | needs_review | escalated | dropped
    intent: str | None
    reply: str
    reasons: list[str]
    citations: list[Citation]
    revisions: int
    metrics: dict


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute, self.hits, self.lock = per_minute, defaultdict(deque), threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


def create_app(service: Service | None = None) -> FastAPI:
    s = get_settings()
    limiter = RateLimiter(s.rate_limit_per_min)
    runs: "OrderedDict[str, list]" = OrderedDict()          # bounded in-memory trace store
    state: dict = {"svc": service}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["svc"] is None:                             # production path: build + ingest at startup
            from app.tools.seed import seed
            if not Path(s.orders_db).exists():
                seed(s.orders_db)
            svc = Service()
            svc.ingest_dir(s.data_dir)                       # idempotent upsert (storage may be ephemeral)
            state["svc"] = svc
        yield

    app = FastAPI(title="AgentDesk", version="0.1.0", lifespan=lifespan)

    def guard(request: Request, x_api_key: str | None = Header(default=None)):
        expected = reveal(s.api_key)
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(401, "invalid or missing X-API-Key")
        ip = request.client.host if request.client else "unknown"
        if not limiter.allow(ip):
            raise HTTPException(429, "rate limit exceeded, try again in a minute")

    @app.get("/health")
    def health():
        return {"status": "ok", "llm_provider": s.llm_provider, "ready": state["svc"] is not None}

    @app.post("/v1/resolve", response_model=ResolveResponse, dependencies=[Depends(guard)])
    def resolve(req: ResolveRequest):
        svc: Service = state["svc"]
        try:
            r = svc.resolve(req.ticket, req.customer_id, req.tenant)
        except Exception as e:                               # e.g. LLM quota exhausted -> clean 503, no stack trace
            raise HTTPException(503, f"upstream model error: {type(e).__name__}") from e
        runs[r["run_id"]] = r["trace"]
        while len(runs) > 200:
            runs.popitem(last=False)
        return {k: r[k] for k in ResolveResponse.model_fields}

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(guard)])
    def get_run(run_id: str):
        if run_id not in runs:
            raise HTTPException(404, "run not found")
        return {"run_id": run_id, "spans": runs[run_id]}

    @app.get("/v1/metrics", dependencies=[Depends(guard)])
    def metrics():
        return METRICS.snapshot()

    return app


app = create_app()
