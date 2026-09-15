"""
Indexing Script
Run this ONCE to process your documents and build the vector index.

Usage:
    python index.py                    # use default ./data directory
    python index.py /path/to/docs      # use custom directory
    python index.py --chunk-size 300   # customize chunk size

Pipeline:
    PDF/Text Files -> Load -> Chunk -> Embed -> Store in ChromaDB
"""

import sys
import random
import argparse

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from document_loader import load_documents
from chunker import chunk_all_documents
from vector_store import VectorStore


def build_index(data_dir="./data", chunk_size=500, overlap=50, collection_name="course_material"):
    """
    Complete indexing pipeline.

    Args:
        data_dir: Directory containing your course material (PDFs, text files)
        chunk_size: Target chunk size in characters (tune this!)
        overlap: Character overlap between consecutive chunks
        collection_name: Name for the ChromaDB collection
    """

    # ---- Step 1: Load Documents ----
    print("=" * 60)
    print("STEP 1: Loading Documents")
    print("=" * 60)
    documents = load_documents(data_dir)

    if not documents:
        print(f"\nNo documents found in '{data_dir}'.")
        print("Please add PDF, TXT, or MD files to the data directory.")
        print("\nExample:")
        print("  mkdir -p data")
        print("  cp your_textbook.pdf data/")
        print("  python index.py")
        return None

    # ---- Step 2: Chunk Documents ----
    print("\n" + "=" * 60)
    print(f"STEP 2: Chunking (size={chunk_size}, overlap={overlap})")
    print("=" * 60)
    chunks = chunk_all_documents(documents, chunk_size, overlap)

    if not chunks:
        print("No chunks created. Check document content.")
        return None

    # ---- Step 3: Manual Inspection ----
    print("\n" + "=" * 60)
    print("STEP 3: Chunk Inspection (ALWAYS review before indexing!)")
    print("=" * 60)

    num_samples = min(5, len(chunks))
    samples = random.sample(chunks, num_samples)

    for i, chunk in enumerate(samples):
        print(f"\n{'- ' * 30}")
        print(f"Sample {i + 1}/{num_samples}")
        print(f"  ID:      {chunk['id']}")
        print(f"  Source:  {chunk['metadata']['source']}")
        print(f"  Section: {chunk['metadata']['section']}")
        print(f"  Length:  {chunk['metadata']['char_count']} chars")
        print(f"  Text preview:")
        preview = chunk["text"][:250].replace("\n", "\n           ")
        print(f"           {preview}")
        if len(chunk["text"]) > 250:
            print("           ...")

    print(f"\n{'- ' * 30}")
    print(f"\nTotal chunks to index: {len(chunks)}")

    # Pause for inspection (remove in production)
    try:
        response = input(
            "\nLook good? Press Enter to index, or Ctrl+C to abort: "
        )
    except KeyboardInterrupt:
        print("\n\nAborted. Adjust chunk_size/overlap and retry.")
        return None

    # ---- Step 4: Embed and Store ----
    print("\n" + "=" * 60)
    print("STEP 4: Embedding & Storing in ChromaDB")
    print("=" * 60)

    store = VectorStore(collection_name=collection_name)
    store.reset()  # fresh start (important when re-indexing with different params)
    store.add_chunks(chunks)

    # ---- Step 5: Verification ----
    print("\n" + "=" * 60)
    print("STEP 5: Verification")
    print("=" * 60)
    store.stats()

    # Quick sanity check: search for something and see if results make sense
    if chunks:
        test_query = chunks[0]["metadata"]["section"]  # use a section title as query
        print(f"\nSanity check query: '{test_query}'")
        results = store.search(test_query, k=2)
        for r in results:
            print(f"  Score: {r['score']:.3f} | {r['metadata']['section']}")

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE!")
    print(f"Collection: {collection_name}")
    print(f"Documents: {store.collection.count()}")
    print(f"Persistence: ./chroma_db/")
    print("=" * 60)
    print("\nNext step: Run 'python query.py' to test retrieval interactively.")

    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index course material for RAG retrieval")
    parser.add_argument("data_dir", nargs="?", default="./data", help="Directory containing documents (default: ./data)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk size in characters (default: 500)")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap between chunks in characters (default: 50)")
    parser.add_argument("--collection", default="course_material", help="ChromaDB collection name (default: course_material)")

    args = parser.parse_args()
    build_index(args.data_dir, args.chunk_size, args.overlap, args.collection)
