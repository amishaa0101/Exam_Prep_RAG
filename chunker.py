"""
Chunker Module
Splits documents into retrieval-ready chunks with metadata preservation.

Strategies:
1. Recursive character splitting (default)
2. Structure-aware splitting (respects headers/sections)
3. Question-paper splitting (one Q+A per chunk)
4. Content quality filtering (removes PDF garbage)
"""

import re


def extract_sections(text):
    """
    Split text by headers. Handles multiple formats:
    - Markdown: # Title, ## Title 
    - Textbook: Chapter N: Title, 1.1 Title
    - GATE notes / PDF style: ALL CAPS HEADERS, Title Case Lines
    - Numbered sections: 1. Title, 1.1. Title (but NOT bare "1." or list items)
    """
    lines = text.split("\n")
    sections = []
    current_header = "Introduction"
    current_body_lines = []

    def is_header(line):
        """Detect if a line is likely a section header."""
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            return False

        # Markdown headers: # Title
        if re.match(r"^#{1,3}\s+.+", stripped):
            return True

        # Chapter headers: Chapter 1: Title
        if re.match(r"^Chapter\s+\d+", stripped, re.IGNORECASE):
            return True

        # Timestamps from the note-taking app ("6:29 PM") are never headers
        if re.match(r"^\d{1,2}:\d{2}\s*[AP]M$", stripped):
            return False

        # ALL CAPS headers (common in GATE notes): "NORMALIZATION", "ER MODEL"
        # Must be mostly letters with a real word in it, otherwise formulas like
        # "A = 1100", "W(A) - W(B)" or "[BC, BCD]" get picked up (isupper() is True for them)
        if (stripped.isupper() and len(stripped) > 4 and len(stripped) < 80
                and not stripped.startswith("-")):
            non_space = stripped.replace(" ", "")
            alpha_ratio = sum(c.isalpha() for c in non_space) / len(non_space)
            if alpha_ratio >= 0.8 and re.search(r"[A-Z]{4,}", stripped):
                return True

        # Numbered section with title: "1.1 Entity Set" or "3.1. Attribute types"
        # Requires a sub-section number so list items like "2. Repeat V-1 times:" don't count
        if re.match(r"^\d+(\.\d+)+\.?\s+[A-Z].{3,}", stripped):
            if len(stripped) < 100:
                return True

        # Title case short lines ending with common CS topic keywords
        if (len(stripped.split()) >= 2 and len(stripped) < 60
                and stripped[0].isupper()
                and stripped.endswith((":", "Model", "Set", "Keys", "Form",
                                       "Properties", "Constraints", "Operations",
                                       "Algebra", "Calculus", "Language",
                                       "Normalization", "Indexing", "Hashing"))):
            return True

        return False

    for line in lines:
        if is_header(line):
            body = "\n".join(current_body_lines).strip()
            if body and len(body) > 30:
                sections.append({"header": current_header, "body": body})
            current_header = line.strip().lstrip("#").strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)

    # Don't forget the last section
    body = "\n".join(current_body_lines).strip()
    if body and len(body) > 30:
        sections.append({"header": current_header, "body": body})

    if not sections and text.strip():
        sections.append({"header": "Content", "body": text.strip()})

    return sections


# ============================================================
# DIFFICULTY TAGGING (powers adaptive retrieval)
# Rule-based on purpose: free, instant, and every decision can be explained.
#   basic signals:    definition language ("is called", "types of", "why we use")
#   advanced signals: worked problems ("Q.", "sol.", "how many", "maximum"),
#                     formula/symbol density, digits, little plain prose
# Thresholds were set from the score distribution of the course notes:
# roughly the bottom quarter is "basic" and the top quarter "advanced".
# ============================================================

DEFINITION_CUES = re.compile(
    r"\b(is|are) (a|an|the|called|defined)\b|\bmeans\b|\bdefin|\bcalled\b|\btypes? of\b"
    r"|\bintroduction\b|\bwhy (do )?(we )?(need|use)\b|\bused to\b|\bpurpose\b|\boverview\b",
    re.IGNORECASE,
)
PROBLEM_CUES = re.compile(
    r"(^|\n)\s*(q\s*\.|q\(|que\b|sol\.|solution|practice|example|ex\s*[:.])"
    r"|\bhow many\b|\bmaximum\b|\bminimum\b|\bnumber of\b|\bcalculate\b|\bfind the\b",
    re.IGNORECASE,
)
FORMULA_SYMBOLS = set("=+[]{}()∪∩⋈σπ≤≥<>∃∀∈|÷×^")
BASIC_BELOW = 1.2
ADVANCED_ABOVE = 5.5


def difficulty_score(text):
    """Higher = more problem-solving / formula heavy, lower = more definitional."""
    body = text.split("\n\n", 1)[-1]  # ignore the prepended section header
    n = max(len(body), 1)
    symbol_density = sum(c in FORMULA_SYMBOLS for c in body) / n
    digit_density = sum(c.isdigit() for c in body) / n
    tokens = body.split()
    prose_ratio = len(re.findall(r"[A-Za-z]{3,}", body)) / max(len(tokens), 1)

    advanced = 1.5 * len(PROBLEM_CUES.findall(body)) + 25 * symbol_density + 25 * digit_density + 2 * (1 - prose_ratio)
    basic = 1.5 * len(DEFINITION_CUES.findall(body))
    return round(advanced - basic, 2)


def estimate_difficulty(text):
    """Map the score to the three levels used by adaptive retrieval."""
    score = difficulty_score(text)
    if score < BASIC_BELOW:
        return "basic", score
    if score > ADVANCED_ABOVE:
        return "advanced", score
    return "intermediate", score


def recursive_chunk(text, chunk_size=500, overlap=50):
    """
    Split text using recursive character splitting.
    Tries natural boundaries in order: paragraphs > lines > sentences > words.
    """
    if overlap >= chunk_size:
        # The hard-split fallback would never advance and loop forever
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", ", ", " "]

    for sep in separators:
        if sep not in text:
            continue

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part

            if len(candidate) > chunk_size and current:
                chunks.append(current.strip())
                if overlap > 0 and len(current) > overlap:
                    overlap_text = current[-overlap:]
                    current = overlap_text + sep + part
                else:
                    current = part
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        if chunks and all(len(c) <= chunk_size * 1.5 for c in chunks):
            return chunks

    # Fallback: hard split
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap if overlap > 0 else end

    return chunks


def chunk_document(document, chunk_size=500, overlap=50):
    """
    Chunk a single document with metadata preservation.
    Uses structure-aware splitting with content quality filtering.
    """
    text = document["text"]
    source = document["source"]

    sections = extract_sections(text)

    all_chunks = []
    chunk_id = 0

    for section in sections:
        header = section["header"]
        body = section["body"]

        if len(body.strip()) < 50:
            continue

        sub_chunks = recursive_chunk(body, chunk_size, overlap)

        for sub_chunk in sub_chunks:
            # Quality filter: skip chunks that are mostly noise
            words = sub_chunk.split()
            if len(words) < 5:
                continue  # too short to be useful

            # Check if chunk is mostly non-alphabetic (table/diagram garbage)
            alpha_chars = sum(1 for c in sub_chunk if c.isalpha())
            total_chars = len(sub_chunk.strip())
            if total_chars > 0 and alpha_chars / total_chars < 0.3:
                continue  # less than 30% alphabetic = probably garbage

            chunk_text = f"{header}\n\n{sub_chunk}"
            difficulty, score = estimate_difficulty(chunk_text)

            all_chunks.append(
                {
                    "id": f"{source}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "section": header,
                        "chunk_index": chunk_id,
                        "char_count": len(chunk_text),
                        "difficulty": difficulty,
                        "difficulty_score": score,
                    },
                }
            )
            chunk_id += 1

    return all_chunks


def chunk_question_paper(document, source_year="unknown"):
    """
    Special chunker for past year question papers.
    Each question (+ solution if present) becomes one chunk.
    """
    text = document["text"]
    source = document["source"]

    # A question starts at the beginning of a line: "Q1.", "Question 3", "4)" or "5."
    # Anchoring to line start keeps "Step 2." inside an answer from starting a new question
    question_pattern = re.compile(r"(?m)^\s*(?:Q(?:uestion)?\.?\s*(\d+)|(\d+)[\.\)])\s*[\.:\)]?\s*")
    markers = list(question_pattern.finditer(text))

    chunks = []
    chunk_id = 0

    for n, match in enumerate(markers):
        q_num = match.group(1) or match.group(2)
        end = markers[n + 1].start() if n + 1 < len(markers) else len(text)
        content = text[match.end():end].strip()

        if len(content) > 50:
            header = f"Question {q_num}"

            chunks.append(
                {
                    # Same "_chunk_" ID scheme as chunk_document so context mode can find neighbours
                    "id": f"{source}_chunk_{chunk_id}",
                    "text": f"{header}\n\n{content}",
                    "metadata": {
                        "source": source,
                        "section": header,
                        "chunk_index": chunk_id,
                        "char_count": len(content),
                        "content_type": "question_paper",
                        "year": source_year,
                        # Past-paper questions are exam problems by definition
                        "difficulty": "advanced",
                        "difficulty_score": difficulty_score(content),
                    },
                }
            )
            chunk_id += 1

    if len(chunks) < 2:
        return chunk_document(document)

    return chunks


def chunk_all_documents(documents, chunk_size=500, overlap=50):
    """Chunk all documents with appropriate strategy per document type."""
    all_chunks = []

    for doc in documents:
        source = doc["source"].lower()

        # Match whole filename words, so "latest_notes.md" isn't treated as a "test" paper
        if re.search(r"(^|[^a-z])(paper|exam|quiz|test|pyq)s?([^a-z]|$)", source):
            chunks = chunk_question_paper(doc)
            print(f"  {doc['source']}: {len(chunks)} chunks (question paper mode)")
        else:
            chunks = chunk_document(doc, chunk_size, overlap)
            print(f"  {doc['source']}: {len(chunks)} chunks (recursive mode)")

        all_chunks.extend(chunks)

    if all_chunks:
        sizes = [len(c["text"]) for c in all_chunks]
        print(f"\nTotal: {len(all_chunks)} chunks from {len(documents)} documents")
        print(f"Chunk sizes -- Min: {min(sizes)}, Max: {max(sizes)}, Avg: {sum(sizes) // len(sizes)}")
    else:
        print("\nWARNING: No chunks created. Check your data directory.")

    return all_chunks