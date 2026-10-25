"""RAGPipeline tests with a fake LLM, so no API key or network is needed."""

import pytest

from rag_pipeline import RAGPipeline
from retriever import Retriever
from test_retrieval import FakeCrossEncoder


class FakeLLM:
    provider = "fake"
    model_name = "fake"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def generate(self, system_prompt, messages, temperature=0.2):
        self.calls.append({"system_prompt": system_prompt, "messages": messages})
        if self.fail:
            raise RuntimeError("quota exceeded")
        return f"answer to: {messages[-1]['content']}"


@pytest.fixture
def make_rag(store):
    def _make(fail=False):
        rag = RAGPipeline.__new__(RAGPipeline)  # skip __init__ (it builds a real LLM client)
        rag.store = store
        rag.retriever = Retriever(store)
        rag.retriever._reranker = FakeCrossEncoder()  # no 90MB model download in tests
        rag.llm = FakeLLM(fail=fail)
        rag.conversation_history = []
        rag.student_level = "intermediate"
        rag.max_history_turns = 6
        rag.num_chunks = 3
        rag.temperature = 0.2
        rag.min_rerank_score = 0.05
        return rag
    return _make


def test_answer_includes_sources_and_context(make_rag):
    rag = make_rag()
    out = rag.ask_with_sources("What is BCNF?", mode="basic")
    assert out["answer"] == "answer to: What is BCNF?"
    assert len(out["results"]) == 3
    prompt = rag.llm.calls[0]["system_prompt"]
    assert out["context"] in prompt
    assert "intermediate" in prompt


@pytest.mark.parametrize("mode", ["basic", "context", "adaptive", "rerank"])
def test_all_modes_run(make_rag, mode):
    assert make_rag().ask_with_sources("two phase locking", mode=mode)["results"]


def test_history_kept_and_trimmed_to_window(make_rag):
    rag = make_rag()
    for i in range(10):
        rag.ask(f"question {i}", mode="basic")
    assert len(rag.conversation_history) == rag.max_history_turns * 2
    assert rag.conversation_history[-2]["content"] == "question 9"
    # Previous turns are sent to the LLM
    assert len(rag.llm.calls[-1]["messages"]) == rag.max_history_turns * 2 + 1


def test_use_history_false_is_isolated(make_rag):
    rag = make_rag()
    rag.ask("first", mode="basic")
    rag.ask_with_sources("second", mode="basic", use_history=False)
    assert len(rag.llm.calls[-1]["messages"]) == 1
    assert len(rag.conversation_history) == 2  # unchanged by the isolated call


def test_llm_error_returned_not_raised_and_not_stored(make_rag):
    rag = make_rag(fail=True)
    out = rag.ask_with_sources("anything", mode="basic")
    assert out["answer"].startswith("Error generating response:")
    assert rag.conversation_history == []


def test_empty_index_gives_friendly_message(make_rag, empty_store):
    rag = make_rag()
    rag.store = empty_store
    rag.retriever = Retriever(empty_store)
    out = rag.ask_with_sources("anything", mode="basic")
    assert "couldn't find" in out["answer"]
    assert rag.llm.calls == []


def test_set_student_level_validates(make_rag):
    rag = make_rag()
    rag.set_student_level("advanced")
    rag.set_student_level("expert")
    assert rag.student_level == "advanced"


def test_rerank_refuses_out_of_scope_without_llm_call(make_rag):
    # FakeCrossEncoder gives -2 (sigmoid ~0.12) with no word overlap; raise the bar above that
    rag = make_rag()
    rag.min_rerank_score = 0.5
    out = rag.ask_with_sources("kubernetes pods", mode="rerank")
    assert out["results"] == []
    assert "couldn't find" in out["answer"]
    assert rag.llm.calls == []


def test_rerank_drops_low_scoring_chunks(make_rag):
    rag = make_rag()
    rag.min_rerank_score = 0.5
    out = rag.ask_with_sources("deadlock locking", mode="rerank")
    assert out["results"] and all(r["score"] >= 0.5 for r in out["results"])
