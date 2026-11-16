"""Okapi BM25 (sparse lexical retrieval) -- catches exact IDs/SKUs/error codes that dense vectors blur."""
from __future__ import annotations
import math, re
from collections import Counter

_TOK = re.compile(r"[a-z0-9\-_]+")
STOP = set("a an the and or of to in on for is are was be can i my me you your it this that with at by from how do does what when why not no we our if as so".split())


def tokenize(t: str) -> list[str]:
    return _TOK.findall(t.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if docs else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        q = [t for t in tokenize(query) if t not in STOP]
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            out.append(s)
        return out
