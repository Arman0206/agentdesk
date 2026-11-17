"""
Disk cache for LLM responses (record/replay).

Why: free API tiers allow very few requests per day. Every answer is stored under a hash of
(model, system, prompt, task, json_mode); re-running the eval replays finished tickets for free,
so a full evaluation can be completed over several days. Cached results keep the ORIGINAL measured
tokens and latency, so the reported metrics still describe real model calls.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import asdict
from pathlib import Path
from app.llm.base import LLMClient, LLMResult


class CachingLLM:
    def __init__(self, inner: LLMClient, directory: str = ".llm_cache"):
        self.inner, self.dir = inner, Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "unknown")

    def _path(self, system: str, prompt: str, task: str, json_mode: bool) -> Path:
        key = json.dumps([self.model, system, prompt, task, json_mode], ensure_ascii=False)
        return self.dir / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def generate(self, prompt: str, *, system: str = "", task: str = "generic",
                 json_mode: bool = False) -> LLMResult:
        p = self._path(system, prompt, task, json_mode)
        if p.exists():
            self.hits += 1
            return LLMResult(**json.loads(p.read_text(encoding="utf-8")))
        res = self.inner.generate(prompt, system=system, task=task, json_mode=json_mode)
        self.misses += 1
        if res.text:                                   # never cache empty / blocked answers
            p.write_text(json.dumps(asdict(res), ensure_ascii=False), encoding="utf-8")
        return res
