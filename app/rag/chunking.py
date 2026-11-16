"""Heading-aware chunking for unstructured docs (markdown/plain text)."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass
class Chunk:
    text: str
    source: str
    section: str = ""
    kind: str = "unstructured"      # unstructured | structured
    tenant: str = "default"
    meta: dict = field(default_factory=dict)
    chunk_id: str = ""


def split_sections(md: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_path, body) so chunks never straddle unrelated sections."""
    out, path, buf = [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            out.append((" > ".join(path), body))
        buf.clear()

    for line in md.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return out


def window(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    return [words[i:i + size] for i in range(0, len(words), step) if words[i:i + size]]


def chunk_document(text: str, source: str, tenant: str = "default",
                   size: int = 120, overlap: int = 20) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in split_sections(text):
        for i, w in enumerate(window(body.split(), size, overlap)):
            # Prepend the heading path so the embedding "knows" its context (contextual chunking)
            content = f"{section}: {' '.join(w)}" if section else " ".join(w)
            chunks.append(Chunk(text=content, source=source, section=section, tenant=tenant,
                                chunk_id=f"{tenant}:{source}:{len(chunks)}"))
    return chunks
