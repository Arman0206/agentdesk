"""Wiring: build the singletons once, expose resolve() used by the API and the eval harness."""
from __future__ import annotations
import uuid
from pathlib import Path
from app.agents.graph import build_graph
from app.agents.nodes import Agents
from app.config import get_settings
from app.llm.factory import get_llm
from app.observability.metrics import METRICS
from app.observability.tracing import export_trace, start_trace
from app.rag.embeddings import get_embedder
from app.rag.ingest import load_directory
from app.rag.retriever import HybridRetriever
from app.rag.store import VectorStore


class Service:
    def __init__(self, llm=None, store: VectorStore | None = None, export: bool = True):
        self.s = get_settings()
        self.store = store or VectorStore(get_embedder())
        self.retriever = HybridRetriever(self.store)
        self.agents = Agents(llm or get_llm(), self.retriever)
        self.graph = build_graph(self.agents)
        self.export = export

    def ingest_dir(self, path: str, tenant: str = "default") -> int:
        return self.store.upsert(load_directory(Path(path), tenant))

    def resolve(self, ticket: str, customer_id: str | None = None, tenant: str = "default") -> dict:
        tr = start_trace()
        run_id = tr.trace_id
        final = self.graph.invoke({"ticket": ticket, "customer_id": customer_id, "tenant": tenant},
                                  config={"configurable": {"thread_id": run_id}})
        summ = tr.summary()
        METRICS.observe(summ, final["outcome"])
        if self.export:
            export_trace(tr, self.s.trace_file)
        return {"run_id": run_id, "outcome": final["outcome"], "intent": final.get("intent"),
                "reply": final.get("reply", ""), "reasons": final.get("reasons", []),
                "citations": [{"id": p["id"], "source": p["meta"]["source"], "section": p["meta"].get("section", "")}
                              for p in final.get("passages", [])[:4]],
                "revisions": final.get("revisions", 0), "metrics": summ, "trace": tr.to_dict()["spans"]}
