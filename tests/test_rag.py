from app.rag.bm25 import BM25
from app.rag.chunking import chunk_document, split_sections, window
from app.rag.chunking import Chunk
from app.rag.retriever import rrf


def test_sections_are_heading_aware():
    md = "# Policy\n\n## Annual\nFull refund in 14 days.\n\n## Monthly\nNo refund after billing."
    secs = split_sections(md)
    assert secs[0][0] == "Policy > Annual" and "14 days" in secs[0][1]
    assert secs[1][0] == "Policy > Monthly"


def test_chunks_carry_heading_context_and_ids():
    chunks = chunk_document("# A\n## B\nhello world " * 1, "doc.md", tenant="t1")
    assert chunks and chunks[0].text.startswith("A > B:") and chunks[0].chunk_id.startswith("t1:doc.md:")


def test_window_overlap_covers_all_words():
    words = [str(i) for i in range(100)]
    wins = window(words, size=40, overlap=10)
    assert len(wins) >= 3 and wins[0][-10:] == wins[1][:10]
    assert {w for win in wins for w in win} == set(words)


def test_bm25_ranks_exact_error_code_first():
    docs = ["error e-4102 cannot reach gateway", "error e-5310 file too large", "refund policy annual plans"]
    scores = BM25(docs).scores("what is e-4102")
    assert scores.index(max(scores)) == 0


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] > fused["c"] and fused["b"] > fused["d"]
    assert max(fused, key=fused.get) in {"a", "b"}


def test_hybrid_retrieval_finds_error_doc(service):
    hits = service.retriever.search("I get error E-4102", tenant="default")
    assert hits[0]["meta"]["source"] == "sync_troubleshooting.md"


def test_structured_rows_become_retrievable_cards(service):
    hits = service.retriever.search("Enterprise plan price per user", tenant="default")
    assert any(h["meta"]["kind"] == "structured" and h["meta"]["source"] == "plans.csv" for h in hits)


def test_tenant_isolation(service):
    service.store.upsert([Chunk(text="Secret acme pricing is 99 dollars", source="acme.md", tenant="acme",
                                chunk_id="acme:acme.md:0")])
    assert not any(h["meta"]["tenant"] == "acme" for h in service.retriever.search("secret acme pricing", "default"))
    assert any(h["meta"]["tenant"] == "acme" for h in service.retriever.search("secret acme pricing", "acme"))
