import streamlit as st
from pathlib import Path
from core.config import get_settings
from core.document_loader import load_uploaded_files
from core.vector_store import add_documents, load_vector_store, get_index_stats
from core.rag_pipeline import answer_question

st.set_page_config(page_title="UniMind AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.main-header {font-size:2.25rem;font-weight:750;color:#173B63;margin-bottom:0}
.sub-header {color:#64748b;margin-top:.25rem}
div[data-testid="stMetric"] {background:#f7f9fc;padding:14px;border-radius:12px;border:1px solid #e5eaf1}
</style>
""", unsafe_allow_html=True)

settings = get_settings()
st.sidebar.title("🎓 UniMind AI")
page = st.sidebar.radio("Navigation", ["Dashboard", "Upload Documents", "Knowledge Assistant", "Knowledge Base"])

st.sidebar.divider()
st.sidebar.caption("V1 · RAG Knowledge Base")
st.sidebar.caption("PDF · DOCX · TXT · FAISS · Groq")

if page == "Dashboard":
    st.markdown('<p class="main-header">University Knowledge Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions from your uploaded university documents.</p>', unsafe_allow_html=True)
    stats = get_index_stats(settings.index_dir)
    c1, c2, c3 = st.columns(3)
    c1.metric("Indexed chunks", stats["chunks"])
    c2.metric("Indexed documents", stats["documents"])
    c3.metric("FAISS index", "Ready" if stats["ready"] else "Not created")
    st.info("Start by uploading the sample university documents, then open Knowledge Assistant to ask questions.")
    with st.expander("How this application works"):
        st.write("Documents → text extraction → chunking → HuggingFace embeddings → FAISS → retrieval → Groq answer with sources.")
    st.caption("Sample dataset is fictional. Replace it with approved university documents before real-world use.")

elif page == "Upload Documents":
    st.title("Upload University Documents")
    st.write("Upload PDF, DOCX, or TXT files. The app extracts text, creates chunks and embeddings, then saves them in the FAISS index.")
    uploads = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can select multiple files at once."
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        chunk_size = st.number_input("Chunk size", min_value=300, max_value=2000, value=settings.chunk_size, step=100)
    with col2:
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=settings.chunk_overlap, step=25)
    if st.button("Process and Index Documents", type="primary", disabled=not uploads):
        with st.spinner("Extracting text and preparing document chunks..."):
            docs, skipped = load_uploaded_files(uploads)
        if skipped:
            for message in skipped:
                st.warning(message)
        if not docs:
            st.error("No readable text was found in the selected documents.")
        else:
            with st.spinner("Generating embeddings and updating FAISS index..."):
                result = add_documents(docs, settings.index_dir, settings.embedding_model, int(chunk_size), int(chunk_overlap))
            st.success(f"Indexed {result['chunks_added']} chunks from {result['documents_added']} new document(s).")
            if result["duplicates"]:
                st.info("Skipped already indexed files: " + ", ".join(result["duplicates"]))
            if result["errors"]:
                for error in result["errors"]:
                    st.warning(error)
    st.caption(f"Index location: {settings.index_dir}")

elif page == "Knowledge Assistant":
    st.title("Ask the University Knowledge Assistant")
    stats = get_index_stats(settings.index_dir)
    if not stats["ready"]:
        st.warning("Your knowledge base is empty. Upload documents first.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
                if item.get("sources"):
                    with st.expander("Sources"):
                        for source in item["sources"]:
                            st.write(f"**{source['source']}** · Page: {source.get('page', 'N/A')}")
                            st.caption(source["excerpt"])
        question = st.chat_input("Ask about admissions, fees, courses, exams, or university rules...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                if not settings.groq_api_key:
                    st.error("Groq API key is missing. Add GROQ_API_KEY to .env or Streamlit Secrets.")
                else:
                    with st.spinner("Searching your university knowledge base..."):
                        try:
                            vector_store = load_vector_store(settings.index_dir, settings.embedding_model)
                            result = answer_question(
                                question=question,
                                vector_store=vector_store,
                                api_key=settings.groq_api_key,
                                model=settings.groq_model,
                                top_k=settings.top_k
                            )
                            st.markdown(result["answer"])
                            with st.expander("Retrieved sources"):
                                for source in result["sources"]:
                                    st.write(f"**{source['source']}** · Page: {source.get('page', 'N/A')}")
                                    st.caption(source["excerpt"])
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"]
                            })
                        except Exception as exc:
                            st.error(f"Unable to answer right now: {exc}")
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

elif page == "Knowledge Base":
    st.title("Knowledge Base Status")
    stats = get_index_stats(settings.index_dir)
    a, b = st.columns(2)
    a.metric("Documents", stats["documents"])
    b.metric("Chunks", stats["chunks"])
    if stats["ready"]:
        st.success("FAISS index found.")
        st.code(str(settings.index_dir))
        st.write("Indexed document names:")
        for name in stats["files"]:
            st.write(f"• {name}")
    else:
        st.info("No FAISS index found yet. Upload documents to create one.")
    st.warning("V1 stores the FAISS index on the app filesystem. Streamlit Cloud local storage can be reset during redeployment or restart; add persistent storage before production use.")
