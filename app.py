"""
Streamlit Frontend
A simple chat UI over the RAG pipeline, built for demos.

Sidebar:  upload course material, pick student level, retrieval mode and LLM model
Main:     chat with history; every answer has an expandable "Sources" section
Debug:    toggle to show retrieval scores and context size

Run with:
    streamlit run app.py
"""

import os

import streamlit as st

from chunker import chunk_all_documents
from document_loader import load_pdf, load_text_file
from rag_pipeline import LLMClient, RAGPipeline

DATA_DIR = "./data"
MODELS = ["gemini-3.6-flash", "gemini-flash-lite-latest"]
MODES = {
    "rerank": "Re-ranked (bi-encoder top-20, cross-encoder top-5)",
    "adaptive": "Adaptive (level-aware filtering)",
    "basic": "Basic (plain top-k)",
    "context": "Context-enriched (adds neighbouring chunks)",
}

st.set_page_config(page_title="Exam Prep RAG", page_icon="📚", layout="wide")


# ============================================================
# PIPELINE (cached so the embedding model loads once per model choice)
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model and index...")
def load_pipeline():
    """Build the RAG pipeline once and reuse it across reruns."""
    return RAGPipeline(provider="gemini")


@st.cache_resource(show_spinner=False)
def load_llm(model_name):
    """One LLM client per model, so switching models doesn't reload the embeddings."""
    return LLMClient(provider="gemini", model=model_name)


def index_uploaded_file(rag, uploaded_file):
    """
    Save an uploaded file into ./data and add its chunks to the index.
    Saving into ./data means a later `python index.py` rebuild keeps it.

    Returns:
        Number of chunks added (0 if the file was already indexed)
    """
    store = rag.store
    already = store.collection.get(where={"source": uploaded_file.name}, limit=1)
    if already["ids"]:
        return 0

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.lower().endswith(".pdf"):
        doc = load_pdf(path)
    else:
        doc = load_text_file(path)

    chunks = chunk_all_documents([doc])
    if chunks:
        store.add_chunks(chunks)
    return len(chunks)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📚 Exam Prep RAG")
    st.caption("Answers grounded in your own course material.")

    model_name = st.selectbox(
        "LLM model", MODELS,
        help="Switch to flash-lite if the free-tier quota for the main model runs out.",
    )
    rag = load_pipeline()
    rag.llm = load_llm(model_name)

    st.subheader("Settings")
    level = st.radio(
        "Student level", ["beginner", "intermediate", "advanced"], index=1,
        help="Changes how deep and technical the explanation is.",
    )
    mode = st.selectbox("Retrieval mode", list(MODES), format_func=MODES.get)
    verbose = st.toggle("Verbose / debug", help="Show retrieval scores under each answer")

    st.subheader("Course material")
    uploaded = st.file_uploader("Add a PDF or text file", type=["pdf", "txt", "md"])
    if uploaded is not None and st.button("Index file", use_container_width=True):
        with st.spinner(f"Chunking and embedding {uploaded.name}..."):
            added = index_uploaded_file(rag, uploaded)
        if added:
            st.success(f"Added {added} chunks from {uploaded.name}")
        else:
            st.info(f"{uploaded.name} is already indexed")

    sources = sorted({m["source"] for m in rag.store.collection.get(include=["metadatas"])["metadatas"]})
    st.caption(f"**{rag.store.collection.count()} chunks indexed** from:")
    for s in sources:
        st.caption(f"• {s}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.llm_history = []
        st.rerun()

rag.student_level = level


# ============================================================
# CHAT
# ============================================================

def render_sources(results, show_scores):
    """Expandable list of the chunks the answer was built from."""
    if not results:
        return
    with st.expander(f"Sources ({len(results)} chunks)"):
        for i, r in enumerate(results, start=1):
            meta = r["metadata"]
            header = f"**[{i}] {meta.get('source', '?')}** · {meta.get('section', '')}"
            if show_scores:
                tag = " (neighbour)" if r.get("is_context") else ""
                header += f" · score `{r['score']:.3f}`{tag}"
                if "bi_encoder_score" in r:
                    header += f" · cosine `{r['bi_encoder_score']:.3f}`"
                if meta.get("difficulty"):
                    header += f" · {meta['difficulty']}"
            st.markdown(header)
            st.text(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))


def render_debug(entry):
    """Retrieval details for the verbose toggle."""
    scores = [r["score"] for r in entry["results"]]
    cols = st.columns(4)
    cols[0].metric("Chunks", len(scores))
    cols[1].metric("Top score", f"{max(scores):.3f}" if scores else "-")
    cols[2].metric("Avg score", f"{sum(scores) / len(scores):.3f}" if scores else "-")
    cols[3].metric("Context", f"~{len(entry['context']) // 4} tok")
    st.caption(f"mode: {entry['mode']} · level: {entry['level']} · model: {entry['model']}")


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.llm_history = []

st.header("Ask about your course material")
if not st.session_state.messages:
    st.caption("Try: *What is BCNF and how is it different from 3NF?* or "
               "*How does two-phase locking work?*")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if verbose:
                render_debug(msg)
            render_sources(msg["results"], verbose)

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            # The pipeline is cached and shared by every browser session, so each
            # session keeps its own LLM history and swaps it in for this call
            rag.conversation_history = st.session_state.llm_history
            out = rag.ask_with_sources(question, mode=mode)
            st.session_state.llm_history = rag.conversation_history

        entry = {
            "role": "assistant",
            "content": out["answer"],
            "results": out["results"],
            "context": out["context"],
            "mode": mode,
            "level": level,
            "model": model_name,
        }
        st.markdown(entry["content"])
        if verbose:
            render_debug(entry)
        render_sources(entry["results"], verbose)

    st.session_state.messages.append(entry)
