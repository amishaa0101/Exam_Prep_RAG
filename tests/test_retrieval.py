from retriever import Retriever


def test_search_ranks_relevant_chunk_first(store):
    results = store.search("growing and shrinking phase of locking", k=3)
    assert results[0]["id"] == "notes.md_chunk_2"
    assert results == sorted(results, key=lambda r: r["score"], reverse=True)
    assert all(-1.0 <= r["score"] <= 1.0 for r in results)


def test_search_k_larger_than_index(store):
    assert len(store.search("anything", k=100)) == store.collection.count()


def test_search_empty_index_returns_empty(empty_store):
    # Used to crash: ChromaDB rejects n_results=0
    assert empty_store.search("anything") == []


def test_search_with_exact_metadata_filter(store):
    results = store.search("sorting", k=5, filters={"source": "notes.md"})
    assert results and all(r["metadata"]["source"] == "notes.md" for r in results)


def test_get_by_id(store):
    assert store.get_by_id("algo.md_chunk_0")["metadata"]["section"] == "Sorting"
    assert store.get_by_id("does_not_exist") is None


def test_context_mode_adds_neighbours_in_reading_order(store):
    results = Retriever(store).retrieve_with_context("BCNF super key determinant", k=1)
    ids = [r["id"] for r in results]
    assert "notes.md_chunk_0" in ids and "notes.md_chunk_1" in ids
    keys = [(r["metadata"]["source"], r["metadata"]["chunk_index"]) for r in results]
    assert keys == sorted(keys)


def test_context_mode_keeps_real_score_for_adjacent_hits(store):
    # Chunks 2 and 3 are both hits and neighbours of each other; neither may be
    # demoted to "context" with a discounted score
    retriever = Retriever(store)
    query = "two phase locking deadlock growing shrinking"
    primary = {r["id"]: r["score"] for r in store.search(query, k=2)}
    assert set(primary) == {"notes.md_chunk_2", "notes.md_chunk_3"}

    for r in retriever.retrieve_with_context(query, k=2):
        if r["id"] in primary:
            assert not r.get("is_context")
            assert r["score"] == primary[r["id"]]


def candidate(cid, score, difficulty, section="S"):
    return {"id": cid, "text": f"{section} text", "score": score,
            "metadata": {"difficulty": difficulty, "section": section}}


def adaptive_with(candidates, level, monkeypatch, store, **kwargs):
    """Run retrieve_adaptive over hand-made re-ranked candidates."""
    retriever = Retriever(store)
    monkeypatch.setattr(retriever, "retrieve_with_reranking", lambda *a, **kw: list(candidates))
    return [r["id"] for r in retriever.retrieve_adaptive("q", student_level=level, **kwargs)]


CANDIDATES = [
    candidate("adv_strong", 0.99, "advanced"),
    candidate("mid_strong", 0.97, "intermediate"),
    candidate("basic_strong", 0.95, "basic"),
    candidate("basic_weak", 0.40, "basic"),   # level fit must not lift a weak match
    candidate("adv_weak", 0.30, "advanced"),
]


def test_adaptive_reorders_confident_chunks_by_level(monkeypatch, store):
    assert adaptive_with(CANDIDATES, "beginner", monkeypatch, store, k=3) == ["basic_strong", "mid_strong", "adv_strong"]
    assert adaptive_with(CANDIDATES, "advanced", monkeypatch, store, k=3) == ["adv_strong", "mid_strong", "basic_strong"]


def test_adaptive_intermediate_keeps_relevance_order(monkeypatch, store):
    assert adaptive_with(CANDIDATES, "intermediate", monkeypatch, store, k=5) == [c["id"] for c in CANDIDATES]


def test_adaptive_never_lifts_weak_match_above_confident_one(monkeypatch, store):
    ids = adaptive_with(CANDIDATES, "beginner", monkeypatch, store, k=5)
    assert ids.index("basic_weak") > ids.index("adv_strong")


def test_adaptive_topic_filter_is_case_insensitive(monkeypatch, store):
    # The old {"$contains": ...} metadata filter silently matched nothing
    cands = [candidate("a", 0.9, "basic", "Sorting"), candidate("b", 0.8, "basic", "Graphs"),
             candidate("c", 0.7, "basic", "sorting algorithms")]
    for topic in ("sorting", "SORTING"):
        assert adaptive_with(cands, "intermediate", monkeypatch, store, topic=topic) == ["a", "c"]


def test_adaptive_topic_filter_falls_back_when_too_narrow(monkeypatch, store):
    cands = [candidate("a", 0.9, "basic", "Sorting"), candidate("b", 0.8, "basic", "Graphs")]
    assert adaptive_with(cands, "intermediate", monkeypatch, store, topic="sorting") == ["a", "b"]


def test_format_context_labels_and_budget(store):
    retriever = Retriever(store)
    results = store.search("normalization", k=3)
    context = retriever.format_context(results)
    assert context.startswith("[Source 1: notes.md | Normalization")
    assert context.count("[Source ") == 3

    tiny = retriever.format_context(results, max_tokens_approx=30)  # ~120 chars
    assert tiny.count("[Source ") == 1


class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the chunk, so the order is predictable."""

    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs = list(pairs)
        return [float(sum(w in text.lower() for w in query.lower().split())) - 2 for query, text in pairs]


def test_reranking_reorders_candidates_by_cross_encoder(store):
    retriever = Retriever(store)
    retriever._reranker = FakeCrossEncoder()
    results = retriever.retrieve_with_reranking("deadlock locking", k=2, candidates=5)

    assert len(retriever._reranker.pairs) == 5, "every candidate is scored"
    assert results[0]["id"] == "notes.md_chunk_3"  # only chunk with both words
    assert [r["rerank_logit"] for r in results] == sorted((r["rerank_logit"] for r in results), reverse=True)


def test_reranking_scores_are_probabilities_and_keep_cosine(store):
    retriever = Retriever(store)
    retriever._reranker = FakeCrossEncoder()
    for r in retriever.retrieve_with_reranking("merge sort", k=3):
        assert 0.0 <= r["score"] <= 1.0
        assert "bi_encoder_score" in r


def test_reranking_on_empty_index_skips_model(empty_store):
    retriever = Retriever(empty_store)
    retriever._reranker = FakeCrossEncoder()
    assert retriever.retrieve_with_reranking("anything") == []
    assert retriever._reranker.pairs == []
