import html
import streamlit as st

from core.config import get_settings
from core.document_loader import load_uploaded_files
from core.vector_store import add_documents, load_vector_store, get_index_stats
from core.rag_pipeline import answer_question

st.set_page_config(
    page_title="UniMind AI | University Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme -----------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #080d18;
        --panel: #111a2b;
        --panel-2: #162238;
        --border: #263650;
        --text: #eef4ff;
        --muted: #9aabc4;
        --accent: #62a8ff;
        --accent-2: #8b7cff;
        --good: #35d39a;
    }
    .stApp {
        background:
          radial-gradient(circle at 85% 0%, rgba(71, 111, 201, .16), transparent 28%),
          linear-gradient(135deg, #080d18 0%, #0b1220 55%, #0d1728 100%);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(8,13,24,.88); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101a2c 0%, #0b1322 100%);
        border-right: 1px solid #263650;
    }
    [data-testid="stSidebar"] * { color: #eaf1ff; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: var(--text); }
    .hero {
        padding: 30px 32px;
        border: 1px solid #2b4164;
        border-radius: 22px;
        background: linear-gradient(115deg, rgba(28,53,91,.95), rgba(26,35,66,.94) 58%, rgba(49,42,91,.88));
        box-shadow: 0 18px 45px rgba(0,0,0,.22);
        margin-bottom: 22px;
    }
    .hero-kicker {
        color: #9ecaff; font-size: .78rem; letter-spacing: .16em;
        text-transform: uppercase; font-weight: 800; margin-bottom: 10px;
    }
    .hero-title { color: #fff; font-size: 2.35rem; line-height: 1.12; font-weight: 850; margin: 0; }
    .hero-sub { color: #c1d1e9; font-size: 1rem; margin-top: 10px; margin-bottom: 0; }
    .section-label {
        color: #a9c7ef; text-transform: uppercase; letter-spacing: .12em;
        font-size: .76rem; font-weight: 800; margin: 24px 0 12px;
    }
    .metric-card {
        min-height: 142px; padding: 20px 21px; border-radius: 18px;
        background: linear-gradient(145deg, #17253b, #111b2d);
        border: 1px solid #2b405f; box-shadow: 0 12px 30px rgba(0,0,0,.18);
    }
    .metric-label { color: #a9bad3; font-size: .88rem; font-weight: 650; }
    .metric-value { color: #ffffff; font-size: 2rem; font-weight: 850; margin-top: 12px; line-height: 1.1; }
    .metric-note { color: #8fa5c4; font-size: .78rem; margin-top: 8px; }
    .info-panel {
        padding: 20px 22px; border-radius: 17px; border: 1px solid #263b59;
        background: rgba(17,26,43,.88); margin-top: 16px;
    }
    .step-pill {
        display: inline-block; padding: 7px 11px; border-radius: 999px;
        color: #cfe4ff; background: #1a3152; border: 1px solid #31527d;
        font-size: .78rem; margin: 4px 5px 4px 0;
    }
    div[data-testid="stMetric"] {
        background: #17253b; border: 1px solid #2b405f;
        border-radius: 16px; padding: 16px;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #f2f6ff !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #347fe8, #7668e8);
        color: white; border: 0; border-radius: 10px; font-weight: 750;
    }
    .stButton > button {
        border-radius: 10px; border: 1px solid #385071;
    }
    [data-testid="stFileUploader"] {
        background: #111b2d; border: 1px dashed #45648d; border-radius: 14px; padding: 12px;
    }
    [data-testid="stChatMessage"] {
        background: #121e31; border: 1px solid #263b59; border-radius: 15px;
    }
    .small-muted { color: #9aabc4; font-size: .84rem; }
    hr { border-color: #263650; }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## 🎓 UniMind AI")
    st.caption("UNIVERSITY KNOWLEDGE WORKSPACE")
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        ["Dashboard", "Upload Documents", "Knowledge Assistant", "Knowledge Base"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**V1 · Retrieval-Augmented Generation**")
    st.caption("PDF  ·  DOCX  ·  TXT")
    st.caption("HuggingFace  ·  FAISS  ·  Groq")
    st.markdown("---")
    st.caption("Use approved university documents for real deployments.")

stats = get_index_stats(settings.index_dir)

# ----------------------------- Dashboard -----------------------------
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">AI-powered campus knowledge</div>
          <div class="hero-title">Your University.<br>One Knowledge Assistant.</div>
          <p class="hero-sub">Search university policies, admissions, fees, courses and academic information from your uploaded documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Knowledge base overview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        (c1, "📚", "Indexed documents", stats["documents"], "Files successfully added to the knowledge base"),
        (c2, "🧩", "Indexed chunks", stats["chunks"], "Text segments available for semantic search"),
        (c3, "⚡", "FAISS index", "READY" if stats["ready"] else "NOT READY",
         "Vector search is available" if stats["ready"] else "Upload documents to create the index"),
    ]
    for col, icon, label, value, note in cards:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">{icon} &nbsp; {html.escape(label)}</div>
                  <div class="metric-value">{html.escape(str(value))}</div>
                  <div class="metric-note">{html.escape(note)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label">Get started</div>', unsafe_allow_html=True)
    if stats["ready"]:
        st.success("Your knowledge base is ready. Open Knowledge Assistant to ask questions.")
    else:
        st.info("Your knowledge base is currently empty. Upload PDF, DOCX or TXT documents to populate these metrics.")
    st.markdown(
        """
        <div class="info-panel">
          <b>How UniMind AI works</b><br><br>
          <span class="step-pill">1 · Upload documents</span>
          <span class="step-pill">2 · Extract text</span>
          <span class="step-pill">3 · Split into chunks</span>
          <span class="step-pill">4 · Create embeddings</span>
          <span class="step-pill">5 · Store in FAISS</span>
          <span class="step-pill">6 · Retrieve context</span>
          <span class="step-pill">7 · Generate answer with Groq</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if stats["files"]:
        st.markdown('<div class="section-label">Recently indexed files</div>', unsafe_allow_html=True)
        for filename in stats["files"][:8]:
            st.markdown(f"- 📄 {html.escape(filename)}")
    st.caption("Testing dataset is fictional. Replace it with approved university documents before real-world use.")

# ----------------------------- Upload Documents -----------------------------
elif page == "Upload Documents":
    st.markdown("## 📤 Upload University Documents")
    st.write("Add PDF, DOCX or TXT files. UniMind AI extracts text, creates embeddings and updates the FAISS index.")
    uploads = st.file_uploader(
        "Choose one or more files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can select multiple university documents at once.",
    )
    left, right = st.columns(2)
    with left:
        chunk_size = st.number_input(
            "Chunk size", min_value=300, max_value=2000,
            value=int(settings.chunk_size), step=100,
        )
    with right:
        chunk_overlap = st.number_input(
            "Chunk overlap", min_value=0, max_value=500,
            value=int(settings.chunk_overlap), step=25,
        )

    if st.button("⚡ Process and Index Documents", type="primary", disabled=not uploads):
        with st.spinner("Reading files and extracting text..."):
            docs, skipped = load_uploaded_files(uploads)
        for message in skipped:
            st.warning(message)
        if not docs:
            st.error("No readable text was found in the selected files.")
        else:
            with st.spinner("Creating embeddings and updating FAISS..."):
                result = add_documents(
                    docs, settings.index_dir, settings.embedding_model,
                    int(chunk_size), int(chunk_overlap),
                )
            st.success(
                f"Added {result['chunks_added']} chunks from "
                f"{result['documents_added']} new document(s)."
            )
            if result["duplicates"]:
                st.info("Already indexed: " + ", ".join(result["duplicates"]))
            for error in result["errors"]:
                st.warning(error)
            st.rerun()

    st.markdown("---")
    current = get_index_stats(settings.index_dir)
    a, b, c = st.columns(3)
    a.metric("Indexed documents", current["documents"])
    b.metric("Indexed chunks", current["chunks"])
    c.metric("FAISS status", "Ready" if current["ready"] else "Empty")
    st.caption("Cloud note: Streamlit Community Cloud local files may be reset on restart or redeployment. Add durable storage before production use.")

# ----------------------------- Knowledge Assistant -----------------------------
elif page == "Knowledge Assistant":
    st.markdown("## 💬 University Knowledge Assistant")
    st.caption("Ask a question and get an answer grounded in your indexed university documents.")
    stats = get_index_stats(settings.index_dir)

    if not stats["ready"]:
        st.warning("Your knowledge base is empty. Upload documents first.")
        st.page_link if False else None
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
                if item.get("sources"):
                    with st.expander("📎 Sources used"):
                        for source in item["sources"]:
                            st.write(f"**{source['source']}** · Page: {source.get('page', 'N/A')}")
                            st.caption(source["excerpt"])

        question = st.chat_input("Ask about admissions, fees, courses, exams or university rules...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                if not settings.groq_api_key:
                    st.error("Groq API key is missing. Add GROQ_API_KEY in Streamlit Cloud → Settings → Secrets.")
                else:
                    with st.spinner("Searching your university knowledge base..."):
                        try:
                            vector_store = load_vector_store(settings.index_dir, settings.embedding_model)
                            result = answer_question(
                                question=question,
                                vector_store=vector_store,
                                api_key=settings.groq_api_key,
                                model=settings.groq_model,
                                top_k=settings.top_k,
                            )
                            st.markdown(result["answer"])
                            with st.expander("📎 Retrieved sources"):
                                for source in result["sources"]:
                                    st.write(f"**{source['source']}** · Page: {source.get('page', 'N/A')}")
                                    st.caption(source["excerpt"])
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"],
                            })
                        except Exception as exc:
                            st.error(f"Unable to answer right now: {exc}")

        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

# ----------------------------- Knowledge Base -----------------------------
elif page == "Knowledge Base":
    st.markdown("## 🗂️ Knowledge Base")
    st.caption("Current FAISS index status and indexed source documents.")
    stats = get_index_stats(settings.index_dir)
    a, b, c = st.columns(3)
    a.metric("Documents", stats["documents"])
    b.metric("Chunks", stats["chunks"])
    c.metric("FAISS", "Ready" if stats["ready"] else "Not created")

    if stats["ready"]:
        st.success("FAISS index files are present.")
        st.code(str(settings.index_dir))
        st.markdown("### Indexed documents")
        for name in stats["files"]:
            st.markdown(f"- 📄 {html.escape(name)}")
    else:
        st.info("No FAISS index found yet. Upload documents to create one.")
    st.warning(
        "Streamlit Cloud local storage is not guaranteed to persist across restarts or redeployments. "
        "For a production university system, connect durable storage and implement index backup/restore."
    )
