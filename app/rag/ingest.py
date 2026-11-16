"""
Ingestion for BOTH data shapes enterprise customers actually have:

  unstructured : .md / .txt knowledge-base docs  -> heading-aware chunks
  structured   : .csv / .json records (plans, SKUs, policies) -> one natural-language 'record card'
                 per row, so structured facts are retrievable semantically *and* keep typed metadata.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from app.rag.chunking import Chunk, chunk_document


def record_to_card(row: dict, title_key: str) -> str:
    title = row.get(title_key, "record")
    facts = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in row.items() if k != title_key and v not in ("", None))
    return f"{title}. {facts}."


def ingest_structured(path: Path, tenant: str) -> list[Chunk]:
    rows = (list(csv.DictReader(path.open(encoding="utf-8"))) if path.suffix == ".csv"
            else json.loads(path.read_text(encoding="utf-8")))
    title_key = next(iter(rows[0]))
    return [Chunk(text=record_to_card(r, title_key), source=path.name, section=str(r[title_key]),
                  kind="structured", tenant=tenant, meta={"row": i}, chunk_id=f"{tenant}:{path.name}:{i}")
            for i, r in enumerate(rows)]


def load_directory(root: Path, tenant: str = "default") -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(root.rglob("*")):
        if p.suffix in {".md", ".txt"}:
            chunks += chunk_document(p.read_text(encoding="utf-8"), p.name, tenant)
        elif p.suffix in {".csv", ".json"}:
            chunks += ingest_structured(p, tenant)
    return chunks
