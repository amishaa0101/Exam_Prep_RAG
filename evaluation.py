"""
Evaluation Module
Measures how well the RAG system works, in two separate stages:

1. RETRIEVAL metrics (no LLM needed, free to run)
   - Hit Rate@k:     Did at least one relevant chunk show up in the top k?
   - MRR:            How high up was the first relevant chunk? (1/rank, averaged)
   - Keyword Recall: What fraction of the expected key terms appear in the retrieved text?

2. GENERATION metrics (LLM-as-judge)
   - Faithfulness:   Is every claim in the answer supported by the retrieved context?
   - Relevancy:      Does the answer actually address the question?
   - Completeness:   Does the answer cover the key points a good answer should have?

Why evaluate the two stages separately:
If the final answer is bad, you need to know WHERE it broke. A bad answer with a
retrieval miss is a retrieval problem (chunking, embeddings). A bad answer with a
retrieval hit is a generation problem (prompt, context formatting, model).

How "relevant" is decided:
There are no hand-labelled chunk IDs (the DBMS PDF has 555 noisy chunks, and IDs
change on every re-index). Instead each test case lists topic keywords; a chunk
counts as relevant if it contains at least one of them. This is a proxy, but it
is cheap, reproducible, and survives re-indexing.

3. GOLD-LABEL retrieval metrics (eval_gold.json, hand-graded 0/1/2 per chunk)
   - nDCG@k, Recall@k, MRR, Precision@k against real relevance judgments.
   The keyword proxy saturates at 1.0; these are the numbers to compare retrievers on.

4. OUT-OF-SCOPE questions: topics the notes don't cover. A faithful system declines.

Usage:
    python evaluation.py                          # everything (retrieval + generation + refusal)
    python evaluation.py --retrieval-only         # free, no API calls
    python evaluation.py --retrieval-only --compare   # basic vs adaptive vs rerank table
    python evaluation.py --unanswerable-only --mode rerank
    python evaluation.py --limit 5                # only the first 5 test cases
"""

import json
import sys

# Section names contain symbols ("⋈") that crash print() on the Windows cp1252 console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import re
import time
from datetime import datetime

# ============================================================
# TEST CASES
# relevant_keywords: a chunk containing ANY of these counts as relevant (hit rate, MRR)
# expected_keywords: terms a good context should contain (keyword recall)
# reference_points:  what a complete answer covers (completeness judge)
# All keyword matching is case-insensitive.
# ============================================================

TEST_CASES = [
    # ---- Normalization ----
    {
        "id": "dbms_bcnf",
        "question": "What is BCNF and how is it different from 3NF?",
        "topic": "normalization",
        "relevant_keywords": ["bcnf"],
        "expected_keywords": ["bcnf", "3nf", "functional dependency", "super key"],
        "reference_points": [
            "In BCNF, for every non-trivial FD X -> Y, X must be a super key",
            "3NF additionally allows X -> Y when Y is a prime attribute",
            "BCNF is stricter than 3NF; every BCNF relation is in 3NF",
            "BCNF decomposition may not preserve dependencies",
        ],
    },
    {
        "id": "dbms_closure",
        "question": "How do you find the closure of an attribute set and use it to find candidate keys?",
        "topic": "functional dependencies",
        "relevant_keywords": ["closure", "+ ="],
        "expected_keywords": ["closure", "candidate key", "functional dependency", "attribute"],
        "reference_points": [
            "Start with the attribute set and repeatedly add attributes determined by applicable FDs",
            "If the closure contains all attributes of the relation, the set is a super key",
            "A candidate key is a minimal super key",
        ],
    },
    {
        "id": "dbms_lossless",
        "question": "What is a lossless join decomposition and how do you check it?",
        "topic": "decomposition",
        "relevant_keywords": ["lossless"],
        "expected_keywords": ["lossless", "decomposition", "common attribute", "dependency preserv"],
        "reference_points": [
            "Joining the decomposed relations gives back exactly the original relation, no spurious tuples",
            "For a binary decomposition R1, R2: the common attributes must be a super key of R1 or R2",
            "Dependency preservation is a separate property from losslessness",
        ],
    },
    # ---- Keys and constraints ----
    {
        "id": "dbms_candidate_key",
        "question": "What is the difference between a super key, candidate key and primary key?",
        "topic": "keys",
        "relevant_keywords": ["candidate key", "super key"],
        "expected_keywords": ["super key", "candidate key", "primary key", "minimal"],
        "reference_points": [
            "A super key is any attribute set that uniquely identifies a tuple",
            "A candidate key is a minimal super key",
            "The primary key is one chosen candidate key, which cannot be NULL",
        ],
    },
    {
        "id": "dbms_foreign_key",
        "question": "Explain foreign keys and referential integrity.",
        "topic": "integrity constraints",
        "relevant_keywords": ["foreign key", "referential"],
        "expected_keywords": ["foreign key", "primary key", "referential", "delete"],
        "reference_points": [
            "A foreign key in one relation refers to the primary key of another (or the same) relation",
            "Referential integrity: foreign key values must exist in the referenced relation or be NULL",
            "Deleting/updating referenced tuples can be restricted, cascaded or set to NULL",
        ],
    },
    # ---- ER model ----
    {
        "id": "dbms_weak_entity",
        "question": "What is a weak entity set in the ER model?",
        "topic": "er model",
        "relevant_keywords": ["weak entity"],
        "expected_keywords": ["weak entity", "identifying", "participation", "key"],
        "reference_points": [
            "A weak entity set has no primary key of its own",
            "It depends on an identifying (owner/strong) entity set via an identifying relationship",
            "It has total participation in the identifying relationship",
            "It is identified by the owner's key plus its own partial key (discriminator)",
        ],
    },
    # ---- SQL / query languages ----
    {
        "id": "dbms_exists",
        "question": "How does the EXISTS clause work in SQL nested queries?",
        "topic": "sql",
        "relevant_keywords": ["exists"],
        "expected_keywords": ["exists", "select", "where", "true"],
        "reference_points": [
            "EXISTS returns true if the subquery returns at least one row",
            "It is typically used with a correlated subquery that references the outer query",
            "NOT EXISTS returns true when the subquery returns no rows",
        ],
    },
    {
        "id": "dbms_trc",
        "question": "What is tuple relational calculus?",
        "topic": "relational calculus",
        "relevant_keywords": ["trc", "tuple relational calculus"],
        "expected_keywords": ["trc", "tuple", "exists", "relational algebra"],
        "reference_points": [
            "TRC is a non-procedural query language: it describes what to retrieve, not how",
            "Queries have the form {t | P(t)} where t is a tuple variable and P a predicate",
            "Uses quantifiers (there exists, for all) and logical connectives",
        ],
    },
    # ---- Transactions and concurrency ----
    {
        "id": "dbms_acid",
        "question": "What are the ACID properties of a transaction?",
        "topic": "transactions",
        "relevant_keywords": ["atomicity", "durability", "acid"],
        "expected_keywords": ["atomicity", "consistency", "isolation", "durability"],
        "reference_points": [
            "Atomicity: all or nothing",
            "Consistency: a transaction takes the database from one consistent state to another",
            "Isolation: concurrent transactions do not see each other's intermediate state",
            "Durability: committed changes survive failures",
        ],
    },
    {
        "id": "dbms_conflict_serializability",
        "question": "What is conflict serializability and how do you test a schedule for it?",
        "topic": "serializability",
        "relevant_keywords": ["conflict serializ", "precedence graph"],
        "expected_keywords": ["conflict", "precedence graph", "cycle", "schedule"],
        "reference_points": [
            "Two operations conflict if they are from different transactions, on the same item, and at least one is a write",
            "A schedule is conflict serializable if it is conflict equivalent to some serial schedule",
            "Test: build a precedence graph; the schedule is conflict serializable iff the graph has no cycle",
        ],
    },
    {
        "id": "dbms_recoverable",
        "question": "What is the difference between recoverable and cascadeless schedules?",
        "topic": "recoverability",
        "relevant_keywords": ["recoverable", "cascad"],
        "expected_keywords": ["recoverable", "cascad", "commit", "read"],
        "reference_points": [
            "Recoverable: if Tj reads data written by Ti, Ti commits before Tj commits",
            "Cascading rollback: one transaction's abort forces others that read its data to abort",
            "Cascadeless: transactions only read data written by committed transactions",
            "Every cascadeless schedule is recoverable",
        ],
    },
    {
        "id": "dbms_2pl",
        "question": "How does the two-phase locking protocol work?",
        "topic": "locking",
        "relevant_keywords": ["2pl", "two phase", "growing phase"],
        "expected_keywords": ["growing", "shrinking", "lock", "deadlock"],
        "reference_points": [
            "Growing phase: a transaction may acquire locks but not release any",
            "Shrinking phase: it may release locks but not acquire any",
            "2PL guarantees conflict serializability",
            "Basic 2PL does not prevent deadlocks or cascading rollbacks (strict/rigorous 2PL address the latter)",
        ],
    },
    {
        "id": "dbms_timestamp",
        "question": "Explain the timestamp ordering protocol and Thomas write rule.",
        "topic": "timestamp",
        "relevant_keywords": ["timestamp", "thomas", "rts", "wts"],
        "expected_keywords": ["timestamp", "rts", "wts", "thomas"],
        "reference_points": [
            "Each transaction gets a timestamp; each item keeps read (RTS) and write (WTS) timestamps",
            "Operations that would violate timestamp order cause the transaction to be rolled back",
            "Thomas write rule: an obsolete write (TS(T) < WTS(X)) is ignored instead of rolling back",
        ],
    },
    # ---- Indexing ----
    {
        "id": "dbms_bplus_tree",
        "question": "What is a B+ tree and how is it different from a B tree?",
        "topic": "indexing",
        "relevant_keywords": ["b+ tree", "b+tree"],
        "expected_keywords": ["b+ tree", "leaf", "order", "key"],
        "reference_points": [
            "In a B+ tree all record pointers / data are stored at the leaf level",
            "Internal nodes only hold keys for navigation",
            "Leaf nodes are linked, which makes range queries efficient",
            "In a B tree, keys and record pointers can appear in internal nodes too",
        ],
    },
    {
        "id": "dbms_dense_sparse",
        "question": "What is the difference between a dense index and a sparse index?",
        "topic": "indexing",
        "relevant_keywords": ["dense", "sparse"],
        "expected_keywords": ["dense", "sparse", "index", "record"],
        "reference_points": [
            "A dense index has an index entry for every search key value / record",
            "A sparse index has entries for only some records (e.g. one per block)",
            "A sparse index needs the file to be sorted on the search key",
            "Sparse indexes use less space; dense indexes locate records faster",
        ],
    },
    {
        "id": "dbms_blocking_factor",
        "question": "What is the blocking factor and how do you compute the number of blocks needed for a file?",
        "topic": "file organization",
        "relevant_keywords": ["blocking factor", "block factor"],
        "expected_keywords": ["block", "record", "factor", "size"],
        "reference_points": [
            "Blocking factor = floor(block size / record size): records that fit in one block",
            "Number of blocks = ceil(number of records / blocking factor)",
            "Unspanned organization does not split a record across blocks",
        ],
    },
    # ---- More normalization ----
    {
        "id": "dbms_2nf",
        "question": "What is second normal form and what is a partial dependency?",
        "topic": "normalization",
        "relevant_keywords": ["2nf", "partial dependency"],
        "expected_keywords": ["2nf", "partial dependency", "candidate key", "prime"],
        "reference_points": [
            "A partial dependency is when a non-prime attribute depends on a proper subset of a candidate key",
            "A relation is in 2NF if it is in 1NF and has no partial dependencies",
            "Relations whose candidate keys are single attributes are automatically in 2NF",
        ],
    },
    {
        "id": "dbms_minimal_cover",
        "question": "How do you find the minimal cover of a set of functional dependencies?",
        "topic": "functional dependencies",
        "relevant_keywords": ["minimal cover", "canonical cover", "extraneous", "redundant"],
        "expected_keywords": ["minimal cover", "functional dependency", "redundant", "closure"],
        "reference_points": [
            "Split FDs so every right-hand side has a single attribute",
            "Remove extraneous attributes from left-hand sides",
            "Remove redundant FDs that can be derived from the others",
            "Closures are used to test whether an attribute or FD is redundant",
        ],
    },
    # ---- More ER / SQL ----
    {
        "id": "dbms_participation",
        "question": "What is the difference between total and partial participation in an ER diagram?",
        "topic": "er model",
        "relevant_keywords": ["participation"],
        "expected_keywords": ["total participation", "partial", "entity", "relationship"],
        "reference_points": [
            "Total participation: every entity in the set must take part in the relationship",
            "Partial participation: some entities may not take part",
            "Total participation is drawn with a double line",
        ],
    },
    {
        "id": "dbms_joins",
        "question": "What is the difference between a natural join and an outer join?",
        "topic": "relational algebra",
        "relevant_keywords": ["natural join", "outer join"],
        "expected_keywords": ["natural join", "outer join", "common attribute", "null"],
        "reference_points": [
            "A natural join combines tuples with equal values on the common attributes and keeps one copy of them",
            "Tuples without a match are dropped by a natural (inner) join",
            "An outer join also keeps unmatched tuples, padding missing values with NULL",
            "Left, right and full outer joins keep unmatched tuples from the left, right or both relations",
        ],
    },
    {
        "id": "dbms_group_by_having",
        "question": "How do GROUP BY and HAVING work in SQL, and how is HAVING different from WHERE?",
        "topic": "sql",
        "relevant_keywords": ["group by", "having"],
        "expected_keywords": ["group by", "having", "where", "aggregate"],
        "reference_points": [
            "GROUP BY groups rows with equal values so aggregates are computed per group",
            "HAVING filters groups after grouping, usually on aggregate values",
            "WHERE filters individual rows before grouping and cannot use aggregates",
        ],
    },
    # ---- More concurrency ----
    {
        "id": "dbms_view_serializability",
        "question": "What is view serializability and how is it different from conflict serializability?",
        "topic": "serializability",
        "relevant_keywords": ["view serializ", "view equivalent"],
        "expected_keywords": ["view serializ", "conflict serializ", "blind write", "initial read"],
        "reference_points": [
            "View equivalence: same initial reads, same reads-from relationships and same final writes",
            "A schedule is view serializable if it is view equivalent to a serial schedule",
            "Every conflict serializable schedule is view serializable, but not the other way round",
            "View serializable but not conflict serializable schedules contain blind writes",
        ],
    },
    {
        "id": "dbms_strict_2pl",
        "question": "What is strict two-phase locking and why is it used instead of basic 2PL?",
        "topic": "locking",
        "relevant_keywords": ["strict 2pl", "strict two phase", "rigorous"],
        "expected_keywords": ["strict", "exclusive", "commit", "cascad"],
        "reference_points": [
            "Strict 2PL holds all exclusive locks until the transaction commits or aborts",
            "This prevents cascading rollbacks and gives strict / cascadeless schedules",
            "Rigorous 2PL holds all locks (shared and exclusive) until commit",
        ],
    },
    {
        "id": "dbms_deadlock_prevention",
        "question": "How do wait-die and wound-wait schemes prevent deadlock?",
        "topic": "deadlock",
        "relevant_keywords": ["wait-die", "wound", "wait die"],
        "expected_keywords": ["wait-die", "wound", "timestamp", "older"],
        "reference_points": [
            "Both schemes use transaction timestamps to decide who waits and who is rolled back",
            "Wait-die: an older transaction waits for a younger one; a younger requester dies (rolls back)",
            "Wound-wait: an older transaction wounds (rolls back) a younger holder; a younger requester waits",
        ],
    },
    # ---- Algorithms notes (clean markdown, acts as a control group) ----
    {
        "id": "algo_merge_vs_quick",
        "question": "Compare merge sort and quick sort in terms of time complexity and space.",
        "topic": "sorting",
        "relevant_keywords": ["merge sort", "quick sort"],
        "expected_keywords": ["merge sort", "quick sort", "o(n log n)", "o(n^2)"],
        "reference_points": [
            "Merge sort is O(n log n) in all cases and needs O(n) extra space",
            "Quick sort is O(n log n) on average but O(n^2) in the worst case",
            "Quick sort sorts in place (O(log n) stack space) and is usually faster in practice",
            "Merge sort is stable; quick sort is not",
        ],
    },
]


# ============================================================
# UNANSWERABLE QUESTIONS
# Topics the course material does not cover. A faithful RAG system should say
# the material doesn't cover them instead of answering from the model's own
# knowledge. "Extendible hashing" is a deliberate trap: it sounds like DBMS,
# but these notes have no hashing section.
# ============================================================

UNANSWERABLE_CASES = [
    {"id": "oos_hashing", "question": "How does extendible hashing handle bucket overflow?"},
    {"id": "oos_cap", "question": "Explain the CAP theorem for distributed databases."},
    {"id": "oos_kubernetes", "question": "What is a Kubernetes pod?"},
    {"id": "oos_gradient_descent", "question": "How does gradient descent update model weights?"},
    {"id": "oos_tcp", "question": "Describe the TCP three-way handshake."},
]


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def is_relevant(chunk_text, relevant_keywords):
    """A chunk is relevant if it contains any of the topic keywords."""
    text = chunk_text.lower()
    return any(kw.lower() in text for kw in relevant_keywords)


def hit_rate_at_k(relevance_flags, k):
    """1 if any of the top-k results is relevant, else 0."""
    return 1.0 if any(relevance_flags[:k]) else 0.0


def reciprocal_rank(relevance_flags):
    """1 / (rank of the first relevant result), 0 if none is relevant."""
    for rank, relevant in enumerate(relevance_flags, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def keyword_recall(retrieved_texts, expected_keywords):
    """Fraction of expected keywords that appear anywhere in the retrieved text."""
    combined = " ".join(retrieved_texts).lower()
    found = [kw for kw in expected_keywords if kw.lower() in combined]
    missing = [kw for kw in expected_keywords if kw.lower() not in combined]
    recall = len(found) / len(expected_keywords) if expected_keywords else 0.0
    return recall, found, missing


def evaluate_retrieval(retriever, test_cases, mode="basic", k=5, k_values=(1, 3, 5)):
    """
    Run every test case through the retriever and compute retrieval metrics.

    Args:
        retriever: A Retriever instance
        test_cases: List of test case dicts
        mode: "basic" (plain top-k), "adaptive" (level filtering, intermediate)
              or "rerank" (bi-encoder top-20, cross-encoder top-k)
        k: How many chunks to retrieve
        k_values: Cutoffs to report Hit Rate at

    Returns:
        Dict with "per_case" results and "summary" averages
    """
    per_case = []

    for case in test_cases:
        if mode == "adaptive":
            results = retriever.retrieve_adaptive(case["question"], student_level="intermediate", k=k)
        elif mode == "rerank":
            results = retriever.retrieve_with_reranking(case["question"], k=k)
        else:
            results = retriever.retrieve(case["question"], k=k)

        texts = [r["text"] for r in results]
        flags = [is_relevant(t, case["relevant_keywords"]) for t in texts]
        recall, found, missing = keyword_recall(texts, case["expected_keywords"])

        first_rank = next((i + 1 for i, f in enumerate(flags) if f), None)

        per_case.append({
            "id": case["id"],
            "question": case["question"],
            "hit_rate": {f"@{kk}": hit_rate_at_k(flags, kk) for kk in k_values},
            "reciprocal_rank": reciprocal_rank(flags),
            "first_relevant_rank": first_rank,
            "precision_at_k": sum(flags) / len(flags) if flags else 0.0,
            "keyword_recall": recall,
            "keywords_found": found,
            "keywords_missing": missing,
            "top_scores": [r["score"] for r in results],
            "top_sections": [r["metadata"].get("section", "") for r in results],
        })

    n = len(per_case)
    summary = {
        "mode": mode,
        "k": k,
        "num_cases": n,
        "hit_rate": {f"@{kk}": sum(c["hit_rate"][f"@{kk}"] for c in per_case) / n for kk in k_values},
        "mrr": sum(c["reciprocal_rank"] for c in per_case) / n,
        "precision_at_k": sum(c["precision_at_k"] for c in per_case) / n,
        "keyword_recall": sum(c["keyword_recall"] for c in per_case) / n,
        "avg_top1_score": sum(c["top_scores"][0] for c in per_case if c["top_scores"]) / n,
    }

    return {"per_case": per_case, "summary": summary}


# ============================================================
# GOLD-LABEL RETRIEVAL METRICS
# The keyword proxy above is lenient (any chunk mentioning "BCNF" counts), so
# Hit Rate/MRR saturate at 1.0 and can't tell retrievers apart. eval_gold.json
# holds hand-graded judgments (0/1/2) for every question, which these use.
# ============================================================

GOLD_PATH = "eval_gold.json"


def load_gold(path=GOLD_PATH, store=None):
    """
    Load graded judgments. If a store is given, warn when judged chunks no longer
    exist or their text changed (a re-index with different chunking invalidates labels).
    """
    import hashlib
    import os

    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        judgments = json.load(f)["judgments"]

    if store is not None:
        stale = 0
        for entries in judgments.values():
            for cid, e in entries.items():
                chunk = store.get_by_id(cid)
                if chunk is None or hashlib.md5(chunk["text"].encode("utf-8")).hexdigest()[:10] != e["fingerprint"]:
                    stale += 1
        if stale:
            print(f"WARNING: {stale} gold-labelled chunks are missing or changed since labelling. "
                  f"The index was rebuilt with different chunking; gold metrics are unreliable until relabelled.")

    return {qid: {cid: e["grade"] for cid, e in entries.items()} for qid, entries in judgments.items()}


def ndcg_at_k(retrieved_ids, grades, k):
    """nDCG with graded gains (2^grade - 1). 1.0 = the best possible ordering of judged chunks."""
    import math
    dcg = sum((2 ** grades.get(cid, 0) - 1) / math.log2(i + 2) for i, cid in enumerate(retrieved_ids[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate_retrieval_gold(retriever, test_cases, gold, mode="basic", k=5):
    """
    Retrieval metrics against hand-graded labels.

    - nDCG@k:      ranking quality with graded relevance (main metric)
    - Recall@k:    share of the grade-2 ("directly answers") chunks found in the top k
    - MRR:         1 / rank of the first grade-2 chunk
    - Precision@k: share of top-k chunks that are at least partially useful (grade >= 1)
    - Unjudged:    retrieved chunks with no label (treated as 0; label them if this grows)
    """
    per_case = []
    for case in test_cases:
        grades = gold.get(case["id"])
        if not grades:
            continue
        if mode == "adaptive":
            results = retriever.retrieve_adaptive(case["question"], student_level="intermediate", k=k)
        elif mode == "rerank":
            results = retriever.retrieve_with_reranking(case["question"], k=k)
        else:
            results = retriever.retrieve(case["question"], k=k)

        ids = [r["id"] for r in results]
        found = [grades.get(cid, 0) for cid in ids]
        total_g2 = sum(1 for g in grades.values() if g == 2)

        per_case.append({
            "id": case["id"],
            "ndcg": ndcg_at_k(ids, grades, k),
            "recall": sum(1 for g in found if g == 2) / total_g2 if total_g2 else 0.0,
            "reciprocal_rank": reciprocal_rank([g == 2 for g in found]),
            "precision": sum(1 for g in found if g >= 1) / len(found) if found else 0.0,
            "unjudged": sum(1 for cid in ids if cid not in grades),
            "grades": found,
        })

    n = len(per_case)
    summary = {"mode": mode, "k": k, "num_cases": n}
    for key in ("ndcg", "recall", "reciprocal_rank", "precision"):
        summary[key] = sum(c[key] for c in per_case) / n if n else 0.0
    summary["unjudged"] = sum(c["unjudged"] for c in per_case)
    return {"per_case": per_case, "summary": summary}


def print_gold_comparison(reports):
    """Side-by-side gold-label metrics for several retrieval modes."""
    modes = list(reports)
    k = reports[modes[0]]["summary"]["k"]
    print(f"\n{'=' * 70}")
    print(f"RETRIEVAL vs HAND-LABELLED GOLD SET  (k={k}, {reports[modes[0]]['summary']['num_cases']} questions)")
    print(f"{'=' * 70}")
    print(f"{'Metric':<20}" + "".join(f"{m:>12}" for m in modes))
    print("-" * (20 + 12 * len(modes)))
    for name, key in [(f"nDCG@{k}", "ndcg"), (f"Recall@{k}", "recall"), ("MRR", "reciprocal_rank"),
                      (f"Precision@{k}", "precision")]:
        print(f"{name:<20}" + "".join(f"{reports[m]['summary'][key]:>12.3f}" for m in modes))
    print(f"{'Unjudged chunks':<20}" + "".join(f"{reports[m]['summary']['unjudged']:>12d}" for m in modes))

    print(f"\n{'nDCG per question':<32}" + "".join(f"{m:>10}" for m in modes))
    for i, case in enumerate(reports[modes[0]]["per_case"]):
        print(f"{case['id']:<32}" + "".join(f"{reports[m]['per_case'][i]['ndcg']:>10.2f}" for m in modes))


def evaluate_adaptive_levels(retriever, test_cases, gold, k=5):
    """
    Does adaptive retrieval actually adapt, and what does it cost?
    For each student level: the difficulty mix of the top k, nDCG against the gold
    set, and how many questions get a different top k than plain re-ranking.
    """
    baseline = {c["id"]: [r["id"] for r in retriever.retrieve_with_reranking(c["question"], k=k)]
                for c in test_cases}
    report = {}
    for level in ("beginner", "intermediate", "advanced"):
        mix = {"basic": 0, "intermediate": 0, "advanced": 0}
        ndcg_total, changed = 0.0, 0
        for case in test_cases:
            results = retriever.retrieve_adaptive(case["question"], student_level=level, k=k)
            ids = [r["id"] for r in results]
            for r in results:
                mix[r["metadata"].get("difficulty", "intermediate")] += 1
            ndcg_total += ndcg_at_k(ids, gold.get(case["id"], {}), k)
            changed += ids != baseline[case["id"]]
        total = sum(mix.values()) or 1
        report[level] = {
            "ndcg": ndcg_total / len(test_cases),
            "share": {d: n / total for d, n in mix.items()},
            "questions_changed": changed,
        }
    return report


def print_adaptive_levels(report, num_cases):
    print(f"\n{'=' * 70}")
    print("ADAPTIVE RETRIEVAL BY STUDENT LEVEL  (difficulty mix of retrieved chunks)")
    print(f"{'=' * 70}")
    print(f"{'Level':<14}{'basic':>8}{'interm.':>9}{'advanced':>10}{'nDCG@5':>9}{'changed vs rerank':>20}")
    for level, r in report.items():
        s = r["share"]
        print(f"{level:<14}{s['basic']:>8.0%}{s['intermediate']:>9.0%}{s['advanced']:>10.0%}"
              f"{r['ndcg']:>9.3f}{r['questions_changed']:>14}/{num_cases}")
    print("\n  nDCG is measured against topic relevance only, so a small drop for beginner/advanced")
    print("  is the expected cost of preferring level-appropriate chunks over the single best match.")


def print_retrieval_comparison(reports):
    """Side-by-side summary of several retrieval modes (the ablation table)."""
    modes = list(reports)
    print(f"\n{'=' * 70}")
    print(f"RETRIEVAL COMPARISON  (k={reports[modes[0]]['summary']['k']})")
    print(f"{'=' * 70}")
    print(f"{'Metric':<20}" + "".join(f"{m:>12}" for m in modes))
    print("-" * (20 + 12 * len(modes)))
    rows = [("Hit Rate@1", lambda s: s["hit_rate"]["@1"]), ("Hit Rate@3", lambda s: s["hit_rate"]["@3"]),
            ("MRR", lambda s: s["mrr"]), ("Precision@k", lambda s: s["precision_at_k"]),
            ("Keyword Recall", lambda s: s["keyword_recall"])]
    for name, get in rows:
        print(f"{name:<20}" + "".join(f"{get(reports[m]['summary']):>12.3f}" for m in modes))

    # Per-question precision so you can see WHERE a mode helps or hurts
    print(f"\n{'Precision@k per question':<32}" + "".join(f"{m:>10}" for m in modes))
    for i, case in enumerate(reports[modes[0]]["per_case"]):
        vals = [reports[m]["per_case"][i]["precision_at_k"] for m in modes]
        print(f"{case['id']:<32}" + "".join(f"{v:>10.2f}" for v in vals))


def print_retrieval_report(report):
    """Pretty-print retrieval metrics."""
    s = report["summary"]
    print(f"\n{'=' * 70}")
    print(f"RETRIEVAL EVALUATION  (mode={s['mode']}, k={s['k']}, {s['num_cases']} questions)")
    print(f"{'=' * 70}")

    print(f"\n{'ID':<32}{'Hit@1':>6}{'Hit@5':>7}{'RR':>6}{'P@k':>6}{'KwRec':>7}{'Top1':>7}")
    print("-" * 70)
    for c in report["per_case"]:
        print(f"{c['id']:<32}"
              f"{c['hit_rate']['@1']:>6.0f}"
              f"{c['hit_rate'].get('@5', 0):>7.0f}"
              f"{c['reciprocal_rank']:>6.2f}"
              f"{c['precision_at_k']:>6.2f}"
              f"{c['keyword_recall']:>7.2f}"
              f"{c['top_scores'][0] if c['top_scores'] else 0:>7.3f}")

    print("-" * 70)
    for kk, v in s["hit_rate"].items():
        print(f"  Hit Rate{kk:<5} {v:.3f}")
    print(f"  MRR           {s['mrr']:.3f}")
    print(f"  Precision@{s['k']}   {s['precision_at_k']:.3f}")
    print(f"  Keyword Recall {s['keyword_recall']:.3f}")
    print(f"  Avg top-1 cosine similarity: {s['avg_top1_score']:.3f}")

    misses = [c for c in report["per_case"] if c["reciprocal_rank"] == 0]
    if misses:
        print("\n  Retrieval misses (no relevant chunk in top k):")
        for c in misses:
            print(f"    - {c['id']}: missing keywords {c['keywords_missing']}")


# ============================================================
# GENERATION METRICS (LLM-as-judge)
# ============================================================

JUDGE_PROMPT = """You are a strict evaluator of answers produced by a retrieval-augmented study assistant.

QUESTION:
{question}

RETRIEVED CONTEXT (the only material the assistant was allowed to use):
{context}

REFERENCE KEY POINTS (what a complete answer should cover):
{reference_points}

ASSISTANT'S ANSWER:
{answer}

Score the answer on three criteria, each an integer from 1 to 5:

1. faithfulness: Are the answer's claims supported by the RETRIEVED CONTEXT?
   5 = every factual claim is supported by the context
   3 = mostly supported, some claims not found in the context
   1 = mostly unsupported or contradicts the context
   (Judge against the context, NOT against your own knowledge. Saying the context
   lacks information is faithful.)

2. relevancy: Does the answer directly address the QUESTION?
   5 = fully on topic, 3 = partly on topic or padded, 1 = off topic

3. completeness: How many of the REFERENCE KEY POINTS does the answer cover?
   5 = all, 4 = most, 3 = about half, 2 = few, 1 = none

Respond with JSON only, in exactly this format:
{{"faithfulness": <int>, "faithfulness_reason": "<one sentence>",
  "relevancy": <int>, "relevancy_reason": "<one sentence>",
  "completeness": <int>, "completeness_reason": "<one sentence>",
  "unsupported_claims": ["<claim not found in context>", ...]}}"""


def call_with_retry(fn, max_retries=4, base_delay=15):
    """Call fn(), backing off on rate-limit errors (free tier quotas are tight)."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            rate_limited = ("429" in msg or "quota" in msg or "resource" in msg
                            or "rate" in msg or "timeout" in msg or "deadline" in msg)
            if not rate_limited or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"    Rate limited, waiting {delay}s before retrying...")
            time.sleep(delay)


def parse_judge_json(text):
    """Extract the JSON object from the judge's reply (tolerates code fences)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return JSON: {text[:200]}")
    return json.loads(match.group(0))


class LLMJudge:
    """Scores answers with an LLM. Uses temperature 0 so scores are as repeatable as possible."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def score(self, question, context, answer, reference_points):
        prompt = JUDGE_PROMPT.format(
            question=question,
            context=context,
            reference_points="\n".join(f"- {p}" for p in reference_points),
            answer=answer,
        )

        if self.llm.provider == "gemini":
            import google.generativeai as genai
            response = self.llm.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=2500,
                    response_mime_type="application/json",
                ),
                request_options={"timeout": 120},
            )
            text = response.text
        else:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content

        scores = parse_judge_json(text)
        for key in ("faithfulness", "relevancy", "completeness"):
            scores[key] = max(1, min(5, int(scores[key])))
        return scores


def evaluate_generation(rag, test_cases, mode="basic", delay_between_calls=2):
    """
    Generate an answer for every test case and score it with the LLM judge.

    Args:
        rag: A RAGPipeline instance
        test_cases: List of test case dicts
        mode: Retrieval mode passed to the pipeline
        delay_between_calls: Seconds to pause between API calls (rate limits)

    Returns:
        Dict with "per_case" results and "summary" averages
    """
    judge = LLMJudge(rag.llm)
    per_case = []

    for i, case in enumerate(test_cases, start=1):
        print(f"  [{i}/{len(test_cases)}] {case['id']}")

        try:
            out = call_with_retry(lambda: _ask_or_raise(rag, case["question"], mode))
            time.sleep(delay_between_calls)
            scores = call_with_retry(lambda: judge.score(
                case["question"], out["context"], out["answer"], case["reference_points"]
            ))
            time.sleep(delay_between_calls)
        except Exception as e:
            print(f"    FAILED: {e}")
            per_case.append({"id": case["id"], "question": case["question"], "error": str(e)})
            continue

        retrieval_hit = any(is_relevant(r["text"], case["relevant_keywords"]) for r in out["results"])

        per_case.append({
            "id": case["id"],
            "question": case["question"],
            "answer": out["answer"],
            "retrieval_hit": retrieval_hit,
            "num_chunks": len(out["results"]),
            **scores,
        })

    scored = [c for c in per_case if "error" not in c]
    n = len(scored)
    summary = {
        "mode": mode,
        "num_cases": len(per_case),
        "num_scored": n,
        "num_failed": len(per_case) - n,
    }
    for key in ("faithfulness", "relevancy", "completeness"):
        summary[key] = sum(c[key] for c in scored) / n if n else None

    return {"per_case": per_case, "summary": summary}


REFUSAL_JUDGE_PROMPT = """A study assistant must answer ONLY from the student's course notes.
The question below is NOT covered by the notes, so the correct behaviour is to say the
material doesn't cover it (optionally suggesting where to look), WITHOUT explaining the topic
from general knowledge.

QUESTION: {question}

ASSISTANT'S ANSWER:
{answer}

Did the assistant correctly decline, or did it explain the topic anyway?
Respond with JSON only: {{"refused": true/false, "reason": "<one sentence>"}}"""


def evaluate_unanswerable(rag, cases=UNANSWERABLE_CASES, mode="basic", delay_between_calls=2):
    """
    Out-of-scope questions: a faithful system declines instead of answering from the
    model's general knowledge. In rerank mode the pipeline may refuse before the LLM
    is called at all (low cross-encoder score), which costs no API calls.
    """
    per_case = []
    for i, case in enumerate(cases, start=1):
        print(f"  [{i}/{len(cases)}] {case['id']}")
        try:
            out = call_with_retry(lambda: _ask_or_raise(rag, case["question"], mode))
            if not out["results"]:
                verdict = {"refused": True, "reason": "No chunk passed the relevance threshold; LLM not called."}
                llm_called = False
            else:
                time.sleep(delay_between_calls)
                verdict = call_with_retry(lambda: _judge_refusal(rag, case["question"], out["answer"]))
                llm_called = True
            time.sleep(delay_between_calls)
        except Exception as e:
            print(f"    FAILED: {e}")
            per_case.append({"id": case["id"], "error": str(e)})
            continue
        per_case.append({"id": case["id"], "question": case["question"], "answer": out["answer"],
                         "refused": bool(verdict["refused"]), "reason": verdict.get("reason", ""),
                         "llm_called": llm_called})

    scored = [c for c in per_case if "error" not in c]
    summary = {
        "mode": mode,
        "num_cases": len(per_case),
        "refusal_rate": sum(c["refused"] for c in scored) / len(scored) if scored else None,
        "llm_calls_avoided": sum(not c["llm_called"] for c in scored),
    }
    return {"per_case": per_case, "summary": summary}


def _judge_refusal(rag, question, answer):
    prompt = REFUSAL_JUDGE_PROMPT.format(question=question, answer=answer)
    if rag.llm.provider == "gemini":
        import google.generativeai as genai
        response = rag.llm.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0, max_output_tokens=500, response_mime_type="application/json"),
            request_options={"timeout": 120},
        )
        return parse_judge_json(response.text)
    response = rag.llm.client.chat.completions.create(
        model=rag.llm.model_name, messages=[{"role": "user", "content": prompt}],
        temperature=0.0, response_format={"type": "json_object"},
    )
    return parse_judge_json(response.choices[0].message.content)


def print_unanswerable_report(report):
    s = report["summary"]
    print(f"\n{'=' * 70}")
    print(f"OUT-OF-SCOPE QUESTIONS  (mode={s['mode']}, should all be declined)")
    print(f"{'=' * 70}")
    for c in report["per_case"]:
        if "error" in c:
            print(f"  {c['id']:<24} ERROR: {c['error'][:40]}")
            continue
        status = "declined" if c["refused"] else "ANSWERED FROM OWN KNOWLEDGE"
        how = "" if c["llm_called"] else "  (blocked before LLM)"
        print(f"  {c['id']:<24} {status}{how}")
    if s["refusal_rate"] is not None:
        print(f"\n  Refusal rate: {s['refusal_rate']:.0%}   LLM calls avoided: {s['llm_calls_avoided']}")


def _ask_or_raise(rag, question, mode):
    """Ask without history; turn the pipeline's error string back into an exception so retry works."""
    out = rag.ask_with_sources(question, mode=mode, use_history=False)
    if out["answer"].startswith("Error generating response:"):
        raise RuntimeError(out["answer"])
    return out


def print_generation_report(report):
    """Pretty-print generation metrics and a diagnosis of what to fix."""
    s = report["summary"]
    print(f"\n{'=' * 70}")
    print(f"GENERATION EVALUATION  (LLM-as-judge, scores 1-5, {s['num_scored']} scored)")
    print(f"{'=' * 70}")

    print(f"\n{'ID':<32}{'Hit':>5}{'Faith':>7}{'Relev':>7}{'Compl':>7}")
    print("-" * 58)
    for c in report["per_case"]:
        if "error" in c:
            print(f"{c['id']:<32}  ERROR: {c['error'][:30]}")
            continue
        print(f"{c['id']:<32}{'yes' if c['retrieval_hit'] else 'no':>5}"
              f"{c['faithfulness']:>7}{c['relevancy']:>7}{c['completeness']:>7}")

    print("-" * 58)
    if s["num_scored"]:
        print(f"  Faithfulness  {s['faithfulness']:.2f} / 5")
        print(f"  Relevancy     {s['relevancy']:.2f} / 5")
        print(f"  Completeness  {s['completeness']:.2f} / 5")
    if s["num_failed"]:
        print(f"  ({s['num_failed']} case(s) failed and are excluded from averages)")

    # ---- Diagnosis: map each weak score to its most likely cause ----
    print(f"\n{'=' * 70}")
    print("DIAGNOSIS")
    print(f"{'=' * 70}")

    scored = [c for c in report["per_case"] if "error" not in c]
    issues = 0

    for c in scored:
        notes = []

        if c["faithfulness"] <= 3:
            notes.append(
                f"Low faithfulness ({c['faithfulness']}): answer makes claims not in the context. "
                f"Likely cause: the model is filling gaps from its own knowledge. "
                f"Fix: tighten the 'answer ONLY from context' rule or lower temperature."
            )
            for claim in c.get("unsupported_claims", [])[:3]:
                notes.append(f"    unsupported: {claim}")

        if c["completeness"] <= 3:
            if not c["retrieval_hit"]:
                notes.append(
                    f"Low completeness ({c['completeness']}) AND retrieval miss: "
                    f"this is a RETRIEVAL problem. The right chunks never reached the LLM. "
                    f"Fix: re-ranking, query expansion/HyDE, or better chunking."
                )
            else:
                notes.append(
                    f"Low completeness ({c['completeness']}) despite a retrieval hit: "
                    f"relevant chunks were found but did not contain everything. "
                    f"Likely cause: noisy PDF text or key points split across chunks. "
                    f"Fix: larger k, context-enriched mode, or re-ranking to surface better chunks."
                )

        if c["relevancy"] <= 3:
            notes.append(
                f"Low relevancy ({c['relevancy']}): answer drifts from the question. "
                f"Likely cause: off-topic chunks in context pulling the answer away. "
                f"Fix: re-ranking to drop weak chunks before generation."
            )

        if notes:
            issues += 1
            print(f"\n  {c['id']}: {c['question']}")
            for note in notes:
                print(f"    {note}")

    if issues == 0:
        print("\n  No weak spots: every case scored 4+ on all three metrics.")

    # Overall pattern
    if scored:
        hits = [c for c in scored if c["retrieval_hit"]]
        misses = [c for c in scored if not c["retrieval_hit"]]
        if hits and misses:
            avg_hit = sum(c["completeness"] for c in hits) / len(hits)
            avg_miss = sum(c["completeness"] for c in misses) / len(misses)
            print(f"\n  Completeness when retrieval hit: {avg_hit:.2f}  vs  miss: {avg_miss:.2f}")
            print("  The gap shows how much answer quality depends on retrieval quality.")

        low_faith = sum(1 for c in scored if c["faithfulness"] <= 3)
        low_comp = sum(1 for c in scored if c["completeness"] <= 3)
        print(f"\n  Summary: {low_faith}/{len(scored)} cases with faithfulness <= 3, "
              f"{low_comp}/{len(scored)} with completeness <= 3.")
        if low_comp > low_faith:
            print("  Main bottleneck is getting the RIGHT content (retrieval / source quality), "
                  "not the model inventing things.")
        elif low_faith > 0:
            print("  Main bottleneck is the model going beyond the context. Focus on the prompt.")

    print("\n  Caveat: the judge is the same model family as the generator, which can "
          "inflate scores (self-preference bias). Treat these as relative, not absolute.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip generation metrics (no API calls)")
    parser.add_argument("--mode", choices=["basic", "adaptive", "rerank"], default="basic",
                        help="Retrieval mode to evaluate (default: basic)")
    parser.add_argument("--compare", action="store_true",
                        help="Run retrieval for every mode and print a comparison table")
    parser.add_argument("--k", type=int, default=5, help="Chunks to retrieve (default: 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N test cases")
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--model", default=None,
                        help="LLM model name, e.g. gemini-flash-lite-latest (default: pipeline default)")
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--unanswerable-only", action="store_true",
                        help="Only run the out-of-scope refusal check (up to 10 API calls)")
    args = parser.parse_args()

    cases = TEST_CASES[: args.limit] if args.limit else TEST_CASES
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"mode": args.mode, "k": args.k, "num_cases": len(cases)},
    }

    if args.retrieval_only:
        from vector_store import VectorStore
        from retriever import Retriever
        retriever = Retriever(VectorStore())
        rag = None
    else:
        from rag_pipeline import RAGPipeline
        rag = RAGPipeline(provider=args.provider, model=args.model)
        rag.num_chunks = args.k
        retriever = rag.retriever
        results["config"]["llm"] = f"{rag.llm.provider}/{rag.llm.model_name}"

    # ---- Stage 1: Retrieval ----
    print("\nRunning retrieval evaluation...")
    retrieval_report = evaluate_retrieval(retriever, cases, mode=args.mode, k=args.k)
    print_retrieval_report(retrieval_report)
    results["retrieval"] = retrieval_report

    gold = load_gold(store=retriever.store)
    modes = ("basic", "adaptive", "rerank") if args.compare else (args.mode,)

    if args.compare:
        comparison = {m: evaluate_retrieval(retriever, cases, mode=m, k=args.k) for m in modes}
        print_retrieval_comparison(comparison)
        results["retrieval_comparison"] = {m: r["summary"] for m, r in comparison.items()}

    if gold:
        gold_reports = {m: evaluate_retrieval_gold(retriever, cases, gold, mode=m, k=args.k) for m in modes}
        print_gold_comparison(gold_reports)
        results["retrieval_gold"] = {m: r for m, r in gold_reports.items()}

        if args.compare:
            levels = evaluate_adaptive_levels(retriever, cases, gold, k=args.k)
            print_adaptive_levels(levels, len(cases))
            results["adaptive_levels"] = levels

    # ---- Stage 2: Generation ----
    if rag is not None:
        print(f"\nRunning out-of-scope refusal check ({len(UNANSWERABLE_CASES)} questions)...")
        unanswerable_report = evaluate_unanswerable(rag, mode=args.mode)
        print_unanswerable_report(unanswerable_report)
        results["unanswerable"] = unanswerable_report

    if rag is not None and not args.unanswerable_only:
        print(f"\nRunning generation evaluation ({len(cases)} questions, 2 API calls each)...")
        generation_report = evaluate_generation(rag, cases, mode=args.mode)
        print_generation_report(generation_report)
        results["generation"] = generation_report

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")
