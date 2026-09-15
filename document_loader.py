"""
Document Loader Module
Loads PDFs, text files, and markdown files from a directory.
Each document carries source metadata for traceability throughout the pipeline.
"""

import os
import re
from pathlib import Path


def clean_pdf_text(text):
    """
    Clean up noisy text extracted from PDFs (especially handwritten/scanned GATE notes).
    
    Fixes:
    - Removes page numbers ("Page 181", "Main notes Page 42")
    - Collapses excessive whitespace and newlines
    - Removes isolated single characters (diagram artifacts)
    - Removes lines that are just numbers/letters (table cell remnants)
    """
    # Remove page number patterns
    text = re.sub(r"(?i)\b(main\s+)?notes?\s*page\s*\d+\b", "", text)
    text = re.sub(r"\n\s*-?\s*\d+\s*-?\s*\n", "\n", text)  # standalone page numbers

    # Remove lines that are just isolated single characters or short gibberish
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
        elif len(stripped) <= 2 and not stripped[0].isdigit():
            continue
        elif re.match(r"^[A-Z]\(\d+\)$", stripped):
            continue  # skip "E(1)", "E(2)" etc.
        else:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse excessive spaces
    text = re.sub(r" {3,}", " ", text)

    return text.strip()


def load_text_file(filepath):
    """Load a plain text or markdown file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return {
        "text": text,
        "source": os.path.basename(filepath),
        "filepath": str(filepath),
    }


def load_pdf(filepath):
    """
    Load a PDF file using PyMuPDF (fitz).
    Extracts text from each page and combines into a single document.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(filepath)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page_number": page_num + 1})
    doc.close()

    full_text = "\n\n".join([p["text"] for p in pages])

    # Clean up noisy PDF extraction artifacts
    full_text = clean_pdf_text(full_text)

    return {
        "text": full_text,
        "source": os.path.basename(filepath),
        "filepath": str(filepath),
        "num_pages": len(pages),
    }


def load_documents(directory):
    """
    Load all supported documents from a directory (recursively).
    Supported formats: .pdf, .txt, .md
    """
    documents = []
    supported_extensions = {".pdf", ".txt", ".md"}

    for filepath in sorted(Path(directory).rglob("*")):
        if filepath.suffix.lower() in supported_extensions:
            try:
                if filepath.suffix.lower() == ".pdf":
                    doc = load_pdf(str(filepath))
                else:
                    doc = load_text_file(str(filepath))

                documents.append(doc)
                print(f"  Loaded: {filepath.name} ({len(doc['text'])} chars)")

            except Exception as e:
                print(f"  ERROR loading {filepath.name}: {e}")

    print(f"\nTotal: {len(documents)} documents loaded")
    return documents


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    docs = load_documents(data_dir)
    for doc in docs:
        print(f"\n{doc['source']}: {len(doc['text'])} chars")
        print(f"  Preview: {doc['text'][:200]}...")