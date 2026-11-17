import uuid
import pytest
from app.config import get_settings
from app.llm.mock import MockLLM
from app.rag.embeddings import HashEmbedder
from app.rag.store import VectorStore
from app.service import Service
from app.tools.seed import seed


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = get_settings()
    db = str(tmp_path / "orders.db")
    seed(db)
    monkeypatch.setattr(s, "orders_db", db)
    monkeypatch.setattr(s, "api_key", "")
    return s


@pytest.fixture
def make_service(settings):
    def _make(llm=None) -> Service:
        # unique collection per test: Chroma's in-memory client is shared across a process
        store = VectorStore(HashEmbedder(), ephemeral=True, name=f"t{uuid.uuid4().hex[:12]}")
        svc = Service(llm=llm or MockLLM(), store=store, export=False)
        svc.ingest_dir("data")
        return svc
    return _make


@pytest.fixture
def service(make_service):
    return make_service()
