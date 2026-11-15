    
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
