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

        # ALL CAPS headers (common in GATE notes): "NORMALIZATION", "ER MODEL"
        if (stripped.isupper() and len(stripped) > 4 and len(stripped) < 80
                and not stripped.startswith("-")):
            return True

        # Numbered section with title: "1.1 Entity Set" or "3.1. Attribute types"
        if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z].{3,}", stripped):
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


def recursive_chunk(text, chunk_size=500, overlap=50):
    """
    Split text using recursive character splitting.
    Tries natural boundaries in order: paragraphs > lines > sentences > words.
    """
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

            all_chunks.append(
                {
                    "id": f"{source}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "section": header,
                        "chunk_index": chunk_id,
                        "char_count": len(chunk_text),
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

    question_pattern = r"(?:Q(?:uestion)?\.?\s*(\d+)|(\d+)\s*[\.\)]\s)"
    parts = re.split(question_pattern, text)

    chunks = []
    chunk_id = 0

    i = 0
    while i < len(parts):
        if parts[i] is None:
            i += 1
            continue

        q_num = None
        if i + 1 < len(parts) and parts[i + 1] and parts[i + 1].strip().isdigit():
            q_num = parts[i + 1].strip()

        content = parts[i].strip()
        if len(content) > 50:
            header = f"Question {q_num}" if q_num else f"Question {chunk_id + 1}"

            chunks.append(
                {
                    "id": f"{source}_q_{chunk_id}",
                    "text": f"{header}\n\n{content}",
                    "metadata": {
                        "source": source,
                        "section": header,
                        "chunk_index": chunk_id,
                        "char_count": len(content),
                        "content_type": "question_paper",
                        "year": source_year,
                    },
                }
            )
            chunk_id += 1
        i += 1

    if len(chunks) < 2:
        return chunk_document(document)

    return chunks


def chunk_all_documents(documents, chunk_size=500, overlap=50):
    """Chunk all documents with appropriate strategy per document type."""
    all_chunks = []

    for doc in documents:
        source = doc["source"].lower()

        if any(kw in source for kw in ["paper", "exam", "quiz", "test", "pyq"]):
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