"""
Retriever Module
Three retrieval strategies for different needs:

1. retrieve()                -- Basic: embed query, find nearest chunks
2. retrieve_with_context()   -- Enriched: also fetch neighboring chunks for fuller context
3. retrieve_adaptive()       -- Adaptive: filter by student level/topic (project differentiator)
4. retrieve_with_reranking() -- Two-stage: bi-encoder recall, cross-encoder precision

The adaptive retriever is what makes this project stand out:
it dynamically adjusts WHAT gets retrieved based on the student's assessed level.
"""


import math

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Adaptive retrieval: how well each chunk difficulty fits each student level
# (+1 preferred, 0 neutral, -1 least preferred). "intermediate" has no preference.
LEVEL_PREFERENCE = {
    "beginner": {"basic": 1, "intermediate": 0, "advanced": -1},
    "advanced": {"basic": -1, "intermediate": 0, "advanced": 1},
}
# Cross-encoder score above which a chunk counts as clearly relevant. Level only
# reorders chunks above this line. Picked by sweeping 0.5 / 0.8 / 0.9 on the gold
# set: 0.9 kept the most nDCG while still shifting the difficulty mix.
CONFIDENT_SCORE = 0.9


class Retriever:
    def __init__(self, vector_store, reranker_model=RERANKER_MODEL):
        self.store = vector_store
        self.reranker_model = reranker_model
        self._reranker = None  # loaded on first use (~90MB, only needed for rerank mode)

    @property
    def reranker(self):
        """Lazy-load the cross-encoder so the other modes don't pay its load time."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            print(f"Loading re-ranker: {self.reranker_model}...")
            self._reranker = CrossEncoder(self.reranker_model)
        return self._reranker

    def retrieve(self, query, k=5, filters=None):
        """
        Basic retrieval: embed query, find nearest chunks.
        This is your baseline. Start here, measure quality, then upgrade.
        """
        results = self.store.search(query, k=k, filters=filters)
        return results

    def retrieve_with_context(self, query, k=3, context_window=1):
        """
        Retrieve chunks AND their neighbors for richer context.

        This is a lightweight parent-child approach:
        - Search finds the most relevant small chunks (precise retrieval)
        - Then we grab neighboring chunks (richer context for the LLM)

        Args:
            query: Search query
            k: Number of primary results
            context_window: How many neighbors to include on each side
        """
        results = self.store.search(query, k=k)

        # Primary matches go in first, so a chunk that is both a search hit and a
        # neighbour of an earlier hit keeps its real score instead of being demoted
        enriched = {r["id"]: r for r in results}

        for result in results:
            chunk_idx = result["metadata"]["chunk_index"]
            source = result["metadata"]["source"]

            # Retrieve surrounding chunks for context
            for offset in range(-context_window, context_window + 1):
                neighbor_idx = chunk_idx + offset
                neighbor_id = f"{source}_chunk_{neighbor_idx}"
                if offset == 0 or neighbor_idx < 0 or neighbor_id in enriched:
                    continue

                neighbor = self.store.get_by_id(neighbor_id)
                if neighbor:
                    enriched[neighbor_id] = {
                        "id": neighbor_id,
                        "text": neighbor["text"],
                        "score": result["score"] * 0.8,  # discount neighbor score
                        "metadata": neighbor["metadata"],
                        "is_context": True,  # flag so we know this is added context
                    }

        # Sort by source and chunk index for natural reading order
        return sorted(
            enriched.values(),
            key=lambda x: (x["metadata"]["source"], x["metadata"]["chunk_index"]),
        )

    def retrieve_adaptive(self, query, student_level="intermediate", topic=None, k=5, candidates=20):
        """
        Adaptive retrieval: adjust what gets retrieved based on student's level.

        This is the PROJECT DIFFERENTIATOR. Most RAG systems retrieve the same
        content regardless of who's asking. This one personalizes retrieval.

        How it works (two tiers, so level never beats relevance):
        1. Re-rank `candidates` chunks with the cross-encoder.
        2. Tier 1: chunks the cross-encoder is confident about (score >= CONFIDENT_SCORE),
           ordered by how well their difficulty tag fits the level, then by relevance.
        3. Tier 2: everything else, by relevance only.
        A beginner gets definitions first and an advanced student gets worked
        problems first, but only among chunks that actually answer the question.
        A hard metadata filter was tried first and dropped: it can throw away the
        only chunk that answers the question.

        Args:
            query: Student's question
            student_level: "beginner", "intermediate" (no reordering) or "advanced"
            topic: Optional topic filter (e.g., "sorting", "graphs")
            k: Number of results
            candidates: How many chunks to re-rank before reordering
        """
        results = self.retrieve_with_reranking(query, k=candidates, candidates=candidates)

        # ChromaDB metadata filters only support exact matches, not substrings
        # ({"$contains": ...} on metadata silently matches nothing), so the topic
        # filter is applied in Python. Fall back to unfiltered if it leaves too little.
        if topic:
            by_topic = self._filter_by_topic(results, topic)
            if len(by_topic) >= 2:
                results = by_topic

        preference = LEVEL_PREFERENCE.get(student_level)
        if preference:
            def tier_key(r):
                confident = r["score"] >= CONFIDENT_SCORE
                fit = preference.get(r["metadata"].get("difficulty"), 0) if confident else 0
                return (confident, fit, r["score"])

            results = sorted(results, key=tier_key, reverse=True)

        return results[:k]

    def retrieve_with_reranking(self, query, k=5, candidates=20, filters=None):
        """
        Two-stage retrieval: bi-encoder for recall, cross-encoder for precision.

        Why two stages:
        - The bi-encoder embeds query and chunk SEPARATELY, so it is fast (one vector
          lookup over the whole index) but only compares two summaries of meaning.
        - The cross-encoder reads query and chunk TOGETHER in one forward pass, so it
          sees word-level interactions ("BCNF" in the query vs "BCNF" in the chunk).
          Much more accurate, but too slow to run over every chunk.
        So: cheap bi-encoder picks `candidates`, expensive cross-encoder orders them.

        Args:
            query: Search query
            k: Number of results to return after re-ranking
            candidates: How many bi-encoder results to re-rank
            filters: Optional ChromaDB where clause for the first stage

        Returns:
            Top-k results. "score" is the cross-encoder relevance (sigmoid, 0-1);
            "bi_encoder_score" keeps the original cosine similarity.
        """
        results = self.store.search(query, k=max(candidates, k), filters=filters)
        if not results:
            return []

        logits = self.reranker.predict([(query, r["text"]) for r in results])

        reranked = []
        for r, logit in zip(results, logits):
            reranked.append({
                **r,
                "score": round(1 / (1 + math.exp(-float(logit))), 4),
                "bi_encoder_score": r["score"],
                "rerank_logit": round(float(logit), 4),
            })

        reranked.sort(key=lambda r: r["rerank_logit"], reverse=True)
        return reranked[:k]

    @staticmethod
    def _filter_by_topic(results, topic):
        """Keep results whose section or text mentions the topic (case-insensitive)."""
        if not topic:
            return results
        topic = topic.lower()
        return [r for r in results
                if topic in r["metadata"].get("section", "").lower() or topic in r["text"].lower()]

    def format_context(self, results, max_tokens_approx=3000):
        """
        Format retrieved results into a context string for the LLM prompt.
        Labels each section with its source for citation support.

        Args:
            results: List of retrieval results
            max_tokens_approx: Approximate token budget (1 token ~ 4 chars)
        """
        max_chars = max_tokens_approx * 4  # rough token-to-char conversion
        context_parts = []
        total_chars = 0

        for i, result in enumerate(results):
            source = result["metadata"].get("source", "Unknown")
            section = result["metadata"].get("section", "")
            score = result.get("score", 0)

            # Label for citation
            label = f"[Source {i + 1}: {source} | {section} | Relevance: {score:.2f}]"
            entry = f"{label}\n{result['text']}"

            # Check if adding this would exceed budget. The top chunk is always kept
            # (trimmed), otherwise one oversized chunk would leave the LLM with no context
            if total_chars + len(entry) > max_chars:
                if not context_parts:
                    context_parts.append(entry[:max_chars])
                break

            context_parts.append(entry)
            total_chars += len(entry)

        return "\n\n---\n\n".join(context_parts)
