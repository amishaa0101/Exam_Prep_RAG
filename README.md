# Exam Prep RAG - Retrieval Pipeline

A from-scratch RAG (Retrieval-Augmented Generation) retrieval pipeline for an Adaptive Exam Preparation Engine. Built without LangChain to understand every component.

## Architecture

```
PDF/Text Files
     |
     v
[Document Loader]  -->  raw text per document
     |
     v
[Chunker]          -->  chunks with metadata (section headers preserved)
     |
     v
[Embedder]         -->  384-dim vectors (sentence-transformers)
     |
     v
[ChromaDB Store]   -->  persistent vector index with metadata
     |
     v
[Retriever]        -->  query in, relevant chunks out
                        (basic / context-enriched / adaptive)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your course material to ./data/
#    (sample algorithms notes included)

# 3. Build the index
python index.py

# 4. Test retrieval interactively
python query.py

# 5. Try different retrieval modes
python query.py --mode adaptive
```

## Project Structure

```
exam-prep-rag/
    document_loader.py   # Load PDFs, text, markdown files
    chunker.py           # Recursive + structure-aware chunking
    vector_store.py      # Embedding + ChromaDB storage
    retriever.py         # Basic, context-enriched, adaptive retrieval
    index.py             # One-time indexing pipeline
    query.py             # Interactive retrieval testing
    data/                # Your course material goes here
    chroma_db/           # Persistent vector index (auto-created)
```

## Retrieval Modes

- **basic**: Standard semantic search. Embed query, find nearest chunks.
- **context**: Retrieves matching chunks + their neighbors for richer LLM context.
- **adaptive**: Filters by student level and topic. The project differentiator.

## Key Design Decisions

1. **No LangChain**: Built from scratch for full understanding and interview depth
2. **Header prepending**: Every chunk includes its section header for self-containment
3. **Metadata throughout**: Source, section, chunk index flow from loading to retrieval
4. **Adaptive fallback**: If filtered search returns too few results, relaxes filters gracefully
5. **Batch embedding**: 100 chunks per batch for 10x faster indexing

## Customization

```bash
# Adjust chunk size (default: 500 chars)
python index.py --chunk-size 300

# Adjust overlap (default: 50 chars)
python index.py --overlap 80

# Use a different collection name
python index.py --collection my_course
```

## What This Maps To in LangChain

| This Project              | LangChain Equivalent              |
|---------------------------|-----------------------------------|
| `document_loader.py`      | `PyMuPDFLoader`                   |
| `chunker.py`              | `RecursiveCharacterTextSplitter`  |
| `vector_store.py`         | `Chroma.from_documents()`         |
| `retriever.retrieve()`    | `vectorstore.as_retriever()`      |
| `retrieve_adaptive()`     | Custom retriever + `search_kwargs`|

## Next Steps

- [ ] Add LLM generation step (complete RAG loop)
- [ ] Build Streamlit/Gradio frontend
- [ ] Add evaluation metrics (RAGAS)
- [ ] Add re-ranking with cross-encoders
