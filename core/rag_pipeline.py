from groq import Groq

SYSTEM_PROMPT = """You are UniMind AI, a university knowledge assistant.
Answer ONLY using the supplied university document context.
If the context does not contain enough information, say:
'I could not find sufficient information in the uploaded university documents to answer this question.'
Do not invent fees, dates, policies, eligibility criteria, contacts, or rules.
Be clear and concise. Mention when a value is explicitly described as illustrative or fictional.
Use source labels such as [Source 1] when useful."""

def answer_question(question: str, vector_store, api_key: str, model: str, top_k: int = 4):
    matches = vector_store.similarity_search_with_score(question, k=top_k)
    if not matches:
        return {"answer": "I could not find sufficient information in the uploaded university documents to answer this question.", "sources": []}

    context_parts, sources = [], []
    seen = set()
    for i, (doc, score) in enumerate(matches, start=1):
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page", "N/A")
        excerpt = doc.page_content[:700].strip()
        context_parts.append(f"[Source {i}] File: {source}; Page: {page}\n{doc.page_content}")
        key = (source, page, excerpt[:100])
        if key not in seen:
            sources.append({"source": source, "page": page, "excerpt": excerpt})
            seen.add(key)

    context = "\n\n".join(context_parts)
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=900,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer using only the context and cite source labels."}
        ],
    )
    answer = completion.choices[0].message.content or "No answer was generated."
    return {"answer": answer, "sources": sources}
