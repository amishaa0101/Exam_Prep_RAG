"""
Shared fixtures. Tests never call an LLM API and never touch ./chroma_db:
the vector store tests build a throwaway index in a temp directory.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vector_store import VectorStore  # noqa: E402


# Small, clearly separated corpus: two sections with two chunks each + one unrelated section
SAMPLE_CHUNKS = [
    ("notes.md", 0, "Normalization", "Normalization: BCNF requires every determinant to be a super key."),
    ("notes.md", 1, "Normalization", "Normalization: 3NF allows a dependency when the right side is a prime attribute."),
    ("notes.md", 2, "Transactions", "Transactions: two phase locking has a growing phase and a shrinking phase."),
    ("notes.md", 3, "Transactions", "Transactions: deadlock can occur under basic two phase locking."),
    ("algo.md", 0, "Sorting", "Sorting: merge sort runs in O(n log n) time and needs extra space."),
]


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    """A VectorStore over SAMPLE_CHUNKS in a temp dir (embedding model loads once per run)."""
    path = tmp_path_factory.mktemp("chroma")
    vs = VectorStore(collection_name="test_material", persist_dir=str(path))
    vs.add_chunks([
        {
            "id": f"{src}_chunk_{idx}",
            "text": text,
            "metadata": {"source": src, "section": section, "chunk_index": idx, "char_count": len(text)},
        }
        for src, idx, section, text in SAMPLE_CHUNKS
    ])
    return vs


@pytest.fixture(scope="session")
def empty_store(store, tmp_path_factory):
    """An empty collection that reuses the already-loaded embedding model."""
    import chromadb
    empty = VectorStore.__new__(VectorStore)
    empty.model = store.model
    empty.model_name = store.model_name
    empty.client = chromadb.PersistentClient(path=str(tmp_path_factory.mktemp("empty")))
    empty.collection = empty.client.get_or_create_collection("empty", metadata={"hnsw:space": "cosine"})
    return empty
