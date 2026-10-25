"""
Vector Store Module
Handles embedding generation and persistent ChromaDB storage.

Key design decisions:
- Uses sentence-transformers (free, local, no API key needed)
- Persistent storage so index survives restarts
- Batch embedding for efficiency
- Same model for indexing and querying (critical: Lesson 4)
"""

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(
        self,
        collection_name="course_material",
        persist_dir="./chroma_db",
        model_name="all-MiniLM-L6-v2",
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_dir: Directory for persistent storage
            model_name: Sentence-transformer model for embeddings
                       (384 dimensions, fast, good quality for projects)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        print(f"Connecting to ChromaDB at: {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        print(
            f"Collection '{collection_name}' ready with "
            f"{self.collection.count()} existing documents\n"
        )

    def add_chunks(self, chunks, batch_size=100):
        """
        Embed and store chunks in batches.
        Batching is important: embedding one-at-a-time is ~10x slower.
        """
        total_batches = (len(chunks) - 1) // batch_size + 1

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            texts = [c["text"] for c in batch]
            ids = [c["id"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            # Embed the entire batch at once
            embeddings = self.model.encode(texts).tolist()

            # Upsert (add or update) to ChromaDB
            self.collection.upsert(
                documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas
            )

            batch_num = i // batch_size + 1
            print(f"  Indexed batch {batch_num}/{total_batches} ({len(batch)} chunks)")

        print(f"\nTotal documents in collection: {self.collection.count()}")

    def search(self, query, k=5, filters=None):
        """
        Search for relevant chunks using semantic similarity.

        Args:
            query: The search query string
            k: Number of results to return
            filters: Optional ChromaDB where clause for metadata filtering
                    e.g., {"source": "textbook.pdf"}
                    e.g., {"$and": [{"source": "textbook.pdf"}, {"section": {"$contains": "sort"}}]}

        Returns:
            List of dicts with id, text, score (cosine similarity), metadata
        """
        # ChromaDB raises on n_results=0, which happens with an empty index
        n_results = min(k, self.collection.count())
        if n_results <= 0:
            return []

        # Embed query with the SAME model used for documents
        query_embedding = self.model.encode(query).tolist()

        search_kwargs = {"query_embeddings": [query_embedding], "n_results": n_results}

        if filters:
            search_kwargs["where"] = filters

        try:
            results = self.collection.query(**search_kwargs)
        except Exception as e:
            # If filtered search fails (e.g., invalid filter), fall back to unfiltered
            print(f"  Filter failed ({e}), falling back to unfiltered search")
            search_kwargs.pop("where", None)
            results = self.collection.query(**search_kwargs)

        # Format results into clean dicts
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # ChromaDB returns distances; convert to similarity
                # For cosine space: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance

                formatted.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "score": round(similarity, 4),
                        "metadata": results["metadatas"][0][i],
                    }
                )

        return formatted

    def get_by_id(self, chunk_id):
        """Retrieve a specific chunk by its ID."""
        try:
            result = self.collection.get(
                ids=[chunk_id], include=["documents", "metadatas"]
            )
            if result["documents"]:
                return {
                    "id": chunk_id,
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception:
            pass
        return None

    def reset(self):
        """Clear and recreate the collection. Use when re-indexing."""
        name = self.collection.name
        meta = self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata=meta
        )
        print(f"Collection '{name}' reset.")

    def stats(self):
        """Print collection statistics."""
        count = self.collection.count()
        print(f"Collection: {self.collection.name}")
        print(f"Documents: {count}")
        print(f"Embedding model: {self.model_name}")
        print(f"Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
        return count
