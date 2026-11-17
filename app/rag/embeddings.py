"""Embedding providers. 'hash' is an offline feature-hashing embedder (CI/dev); 'vertex' is production."""
from __future__ import annotations
import hashlib, math, re
from typing import Protocol
from app.config import get_settings

_TOK = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Unigram + bigram feature hashing, L2-normalised. Deterministic, zero network, good enough for tests."""
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        toks = _TOK.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class VertexEmbedder:
    def __init__(self, project: str, location: str, model: str, dim: int = 768):
        from google import genai
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._model, self.dim = model, dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):  # API batch limit
            r = self._client.models.embed_content(model=self._model, contents=texts[i:i + 100])
            out.extend(e.values for e in r.embeddings)
        return out


class GeminiEmbedder:
    """Gemini API embeddings (AI Studio key). Model name comes from settings -- verify it in AI Studio."""
    def __init__(self, api_key: str, model: str, dim: int = 768):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model, self.dim = model, dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 50):
            r = self._client.models.embed_content(model=self._model, contents=texts[i:i + 50])
            out.extend(e.values for e in r.embeddings)
        return out


def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_provider == "gemini":
        return GeminiEmbedder(s.gemini_api_key, s.gemini_embedding_model)
    if s.embedding_provider == "vertex":
        return VertexEmbedder(s.gcp_project, s.gcp_location, s.embedding_model)
    return HashEmbedder()
