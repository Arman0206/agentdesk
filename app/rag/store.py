"""Vector store backed by Chroma (persistent, local). Swap for Vertex AI Vector Search / AlloyDB pgvector
in production by implementing the same 4 methods -- the retriever only depends on this interface."""
from __future__ import annotations
import chromadb
from app.config import get_settings
from app.rag.chunking import Chunk
from app.rag.embeddings import Embedder


class VectorStore:
    def __init__(self, embedder: Embedder, path: str | None = None, name: str = "agentdesk_kb", ephemeral: bool = False):
        s = get_settings()
        self.embedder = embedder
        self._client = chromadb.EphemeralClient() if ephemeral else chromadb.PersistentClient(path=path or s.chroma_dir)
        # unique name for ephemeral clients so tests don't collide
        self._col = self._client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self.embedder.embed([c.text for c in chunks]),
            metadatas=[{"source": c.source, "section": c.section, "kind": c.kind, "tenant": c.tenant,
                        **{k: str(v) for k, v in c.meta.items()}} for c in chunks],
        )
        return len(chunks)

    def dense_search(self, query: str, k: int, tenant: str) -> list[dict]:
        n = self._col.count()
        if n == 0:
            return []
        r = self._col.query(query_embeddings=self.embedder.embed([query]), n_results=min(k, n),
                            where={"tenant": tenant})
        return [{"id": i, "text": d, "meta": m, "score": 1.0 - dist}
                for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def all_for_tenant(self, tenant: str) -> list[dict]:
        r = self._col.get(where={"tenant": tenant})
        return [{"id": i, "text": d, "meta": m} for i, d, m in zip(r["ids"], r["documents"], r["metadatas"])]

    def count(self) -> int:
        return self._col.count()
