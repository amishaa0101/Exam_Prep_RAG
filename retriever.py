"""
Retriever Module
Three retrieval strategies for different needs:

1. retrieve()             -- Basic: embed query, find nearest chunks
2. retrieve_with_context() -- Enriched: also fetch neighboring chunks for fuller context
3. retrieve_adaptive()     -- Adaptive: filter by student level/topic (project differentiator)

The adaptive retriever is what makes this project stand out:
it dynamically adjusts WHAT gets retrieved based on the student's assessed level.
"""


class Retriever:
    def __init__(self, vector_store):
        self.store = vector_store

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

        enriched = []
        seen_ids = set()

        for result in results:
            chunk_idx = result["metadata"]["chunk_index"]
            source = result["metadata"]["source"]

            # Retrieve surrounding chunks for context
            for offset in range(-context_window, context_window + 1):
                neighbor_idx = chunk_idx + offset
                if neighbor_idx < 0:
                    continue

                neighbor_id = f"{source}_chunk_{neighbor_idx}"

                if neighbor_id in seen_ids:
                    continue
                seen_ids.add(neighbor_id)

                if offset == 0:
                    # This is the primary match
                    enriched.append(result)
                else:
                    # Fetch the neighbor chunk
                    neighbor = self.store.get_by_id(neighbor_id)
                    if neighbor:
                        enriched.append(
                            {
                                "id": neighbor_id,
                                "text": neighbor["text"],
                                "score": result["score"] * 0.8,  # discount neighbor score
                                "metadata": neighbor["metadata"],
                                "is_context": True,  # flag so we know this is added context
                            }
                        )

        # Sort by source and chunk index for natural reading order
        enriched.sort(
            key=lambda x: (x["metadata"]["source"], x["metadata"]["chunk_index"])
        )

        return enriched

    def retrieve_adaptive(self, query, student_level="intermediate", topic=None, k=5):
        """
        Adaptive retrieval: adjust what gets retrieved based on student's level.

        This is the PROJECT DIFFERENTIATOR. Most RAG systems retrieve the same
        content regardless of who's asking. This one personalizes retrieval.

        Args:
            query: Student's question
            student_level: "beginner", "intermediate", or "advanced"
            topic: Optional topic filter (e.g., "sorting", "graphs")
            k: Number of results
        """
        filters = None
        filter_conditions = []

        # Build metadata filters based on student level and topic
        if student_level == "beginner":
            filter_conditions.append({"difficulty": "basic"})
        elif student_level == "advanced":
            filter_conditions.append({"difficulty": "advanced"})

        if topic:
            filter_conditions.append({"section": {"$contains": topic}})

        # Combine filters if multiple conditions
        if len(filter_conditions) == 1:
            filters = filter_conditions[0]
        elif len(filter_conditions) > 1:
            filters = {"$and": filter_conditions}

        # Try filtered search first
        results = self.store.search(query, k=k, filters=filters)

        # FALLBACK: If filters are too restrictive, relax them
        # Better to retrieve at wrong difficulty than retrieve nothing
        if len(results) < 2:
            # Try with just topic filter (drop difficulty)
            if topic:
                results = self.store.search(
                    query, k=k, filters={"section": {"$contains": topic}}
                )

            # If still not enough, go fully unfiltered
            if len(results) < 2:
                results = self.store.search(query, k=k)

        return results

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

            # Check if adding this would exceed budget
            if total_chars + len(entry) > max_chars:
                break

            context_parts.append(entry)
            total_chars += len(entry)

        return "\n\n---\n\n".join(context_parts)
