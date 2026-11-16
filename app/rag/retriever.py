"""Hybrid retrieval: dense (vector) + sparse (BM25) fused with Reciprocal Rank Fusion, tenant-isolated."""
from __future__ import annotations
from app.config import get_settings
from app.rag.bm25 import BM25
from app.rag.store import VectorStore


def rrf(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """score(d) = sum_i 1 / (k + rank_i(d)). Rank-based, so no need to calibrate dense vs BM25 score scales."""
    fused: dict[str, float] = {}
    for lst in rank_lists:
        for r, doc_id in enumerate(lst, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + r)
    return fused


class HybridRetriever:
    def __init__(self, store: VectorStore):
        self.store = store
        self._bm25_cache: dict[str, tuple[int, BM25, list[dict]]] = {}

    def _bm25(self, tenant: str):
        n = self.store.count()
        cached = self._bm25_cache.get(tenant)
        if cached and cached[0] == n:
            return cached[1], cached[2]
        docs = self.store.all_for_tenant(tenant)
        idx = BM25([d["text"] for d in docs])
        self._bm25_cache[tenant] = (n, idx, docs)
        return idx, docs

    def search(self, query: str, tenant: str = "default", k: int | None = None) -> list[dict]:
        s = get_settings()
        k = k or s.top_k
        dense = self.store.dense_search(query, k * 3, tenant)
        idx, docs = self._bm25(tenant)
        sparse: list[dict] = []
        if docs:
            sc = idx.scores(query)
            order = sorted(range(len(docs)), key=lambda i: sc[i], reverse=True)[: k * 3]
            sparse = [docs[i] | {"score": sc[i]} for i in order if sc[i] > 0]
        fused = rrf([[d["id"] for d in dense], [d["id"] for d in sparse]], s.rrf_k)
        by_id = {d["id"]: d for d in dense + sparse}
        ranked = sorted(fused, key=fused.get, reverse=True)[:k]
        dscore = {d["id"]: d["score"] for d in dense}
        sids = {d["id"] for d in sparse}
        return [{**by_id[i], "score": round(fused[i], 5), "dense_score": round(dscore.get(i, 0.0), 4),
                 "dense": i in dscore, "sparse": i in sids} for i in ranked]
