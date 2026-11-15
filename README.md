# UniMind AI — V1 RAG University Knowledge Assistant

A Streamlit application that accepts PDF, DOCX and TXT files, extracts text, chunks it, creates HuggingFace embeddings, stores vectors in FAISS, and answers questions using the Groq API with retrieved source excerpts.

> The bundled sample university dataset is fictional. Replace it with approved university documents before real-world use.

## Features

- Multiple PDF, DOCX and TXT uploads
- Text extraction with page metadata for PDFs
- Recursive text chunking
- HuggingFace sentence-transformer embeddings
- Local FAISS vector index with incremental document additions
- Groq-powered question answering
- Retrieved source filename, page and excerpt display
- Duplicate-file detection using SHA-256 hashes

## Project structure

```text
UniMind_AI_V1/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
└── core/
    ├── __init__.py
    ├── config.py
    ├── document_loader.py
    ├── text_chunker.py
    ├── embeddings.py
    ├── vector_store.py
    └── rag_pipeline.py
```

## Run locally

1. Install Python 3.10 or 3.11 (recommended).
2. Open a terminal in this project folder.
3. Create a virtual environment:

   `python -m venv .venv`

4. Activate it on Windows:

   `.venv\\Scripts\\activate`

5. Install packages:

   `pip install -r requirements.txt`

6. Copy `.env.example` to `.env` and add your Groq API key.
7. Start the app:

   `streamlit run app.py`

The first run may download the embedding model and can take a few minutes.

## Groq API key

Create a key at https://console.groq.com/keys and place it in `.env`:

`GROQ_API_KEY=your_actual_key`

Never commit `.env` or publish your API key. `GROQ_MODEL` can be changed to a model currently available to your Groq account.

## Streamlit Community Cloud

1. Push the project to GitHub.
2. Create a Streamlit Community Cloud app and select `app.py`.
3. In App Settings → Secrets, add:

```toml
GROQ_API_KEY = "your_actual_key"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

4. Deploy.

## Important deployment limitation

V1 writes FAISS files to the app's local filesystem. Streamlit Community Cloud local storage is not guaranteed to persist across restarts, rebuilds, or redeployments. For production, connect durable storage (for example, object storage or a database) and implement index backup/restore. Do not assume the local FAISS folder is permanent in the cloud.

## Limitations in V1

- Scanned/image-only PDFs are not OCR processed.
- DOCX/TXT metadata uses a document-level page marker because those formats do not have reliable PDF-style page metadata.
- Authentication, admin/student roles, per-user access control, and production-grade persistent storage are not included yet.
- Agent tool orchestration is planned for V2; V1 provides the foundational RAG pipeline.

## Suggested test questions

- What is the minimum eligibility for BS Software Engineering?
- What is the semester fee for BS Computer Science?
- What attendance percentage is required for final exams?
- What is the course code for Database Systems?
- When do Fall 2026 classes begin?
- Does the fee document specify hostel charges?
