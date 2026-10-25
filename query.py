"""
Interactive Query Script
Test your retrieval pipeline interactively.

Usage:
    python query.py                    # basic retrieval
    python query.py --mode context     # with neighboring chunks
    python query.py --mode adaptive    # with student-level adaptation

Commands during interactive session:
    quit / exit    -- stop
    mode basic     -- switch to basic retrieval
    mode context   -- switch to context-enriched retrieval
    mode adaptive  -- switch to adaptive retrieval
    mode rerank    -- switch to cross-encoder re-ranking
    level beginner/intermediate/advanced  -- set student level (adaptive mode)
    stats          -- show index statistics
    help           -- show available commands
"""

import argparse
import sys

# Section names contain symbols ("⋈") that crash print() on the Windows cp1252 console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from vector_store import VectorStore
from retriever import Retriever


def print_results(results, show_full=False):
    """Pretty-print retrieval results."""
    if not results:
        print("  No results found.")
        return

    for i, result in enumerate(results):
        is_context = result.get("is_context", False)
        marker = " [context]" if is_context else ""

        print(f"\n  --- Result {i + 1}{marker} (Score: {result['score']:.3f}) ---")
        print(f"  Source:  {result['metadata']['source']}")
        print(f"  Section: {result['metadata']['section']}")

        text = result["text"]
        if show_full or len(text) <= 400:
            # Indent the text for readability
            indented = text.replace("\n", "\n  ")
            print(f"  {indented}")
        else:
            preview = text[:400].replace("\n", "\n  ")
            print(f"  {preview}")
            print(f"  ... ({len(text)} total chars)")


def interactive_query(mode="basic", k=5):
    """Run interactive retrieval session."""

    # Initialize
    store = VectorStore()
    retriever = Retriever(store)

    doc_count = store.collection.count()
    if doc_count == 0:
        print("Index is empty! Run 'python index.py' first to index your documents.")
        return

    student_level = "intermediate"
    current_mode = mode

    print(f"\nIndex has {doc_count} chunks")
    print(f"Mode: {current_mode} | Level: {student_level}")
    print("Type 'help' for commands, 'quit' to exit\n")

    while True:
        try:
            query = input("Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue

        # Handle commands
        lower = query.lower()

        if lower in ("quit", "exit"):
            print("Bye!")
            break

        elif lower == "help":
            print("\n  Commands:")
            print("    quit/exit                  -- stop")
            print("    mode basic/context/adaptive/rerank -- switch retrieval mode")
            print("    level beginner/intermediate/advanced -- set student level")
            print("    stats                      -- show index statistics")
            print("    k <number>                 -- set number of results")
            print("    full                       -- toggle full text display")
            print()
            continue

        elif lower.startswith("mode "):
            new_mode = lower.split(" ", 1)[1].strip()
            if new_mode in ("basic", "context", "adaptive", "rerank"):
                current_mode = new_mode
                print(f"  Switched to {current_mode} mode")
            else:
                print("  Modes: basic, context, adaptive, rerank")
            continue

        elif lower.startswith("level "):
            new_level = lower.split(" ", 1)[1].strip()
            if new_level in ("beginner", "intermediate", "advanced"):
                student_level = new_level
                print(f"  Student level set to: {student_level}")
            else:
                print("  Levels: beginner, intermediate, advanced")
            continue

        elif lower.startswith("k "):
            try:
                k = int(lower.split(" ", 1)[1])
                print(f"  Results count set to: {k}")
            except ValueError:
                print("  Usage: k <number>")
            continue

        elif lower == "stats":
            store.stats()
            continue

        # Execute retrieval based on current mode
        print(f"\n  [{current_mode} mode | level={student_level} | k={k}]")

        if current_mode == "basic":
            results = retriever.retrieve(query, k=k)

        elif current_mode == "context":
            results = retriever.retrieve_with_context(query, k=min(k, 3))

        elif current_mode == "adaptive":
            results = retriever.retrieve_adaptive(
                query, student_level=student_level, k=k
            )

        elif current_mode == "rerank":
            results = retriever.retrieve_with_reranking(query, k=k)

        print_results(results)

        # Also show the formatted context string (what the LLM would see)
        context = retriever.format_context(results)
        print(f"\n  Context for LLM: {len(context)} chars (~{len(context)//4} tokens)")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive retrieval testing")
    parser.add_argument(
        "--mode",
        choices=["basic", "context", "adaptive", "rerank"],
        default="basic",
        help="Retrieval mode (default: basic)",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of results (default: 5)"
    )

    args = parser.parse_args()
    interactive_query(mode=args.mode, k=args.k)
