from io import BytesIO
from pathlib import Path
import hashlib
from docx import Document as DocxDocument
from pypdf import PdfReader
from langchain_core.documents import Document

MAX_FILE_MB = 20

def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _extract_pdf(data: bytes, filename: str):
    reader = PdfReader(BytesIO(data))
    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": page_num, "file_hash": _file_hash(data), "file_type": "pdf"}
            ))
    return docs

def _extract_docx(data: bytes, filename: str):
    document = DocxDocument(BytesIO(data))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    for table in document.tables:
        for row in table.rows:
            text += "\n" + " | ".join(cell.text.strip() for cell in row.cells)
    if not text.strip():
        return []
    return [Document(
        page_content=text,
        metadata={"source": filename, "page": 1, "file_hash": _file_hash(data), "file_type": "docx"}
    )]

def _extract_txt(data: bytes, filename: str):
    text = data.decode("utf-8-sig", errors="replace")
    if not text.strip():
        return []
    return [Document(
        page_content=text,
        metadata={"source": filename, "page": 1, "file_hash": _file_hash(data), "file_type": "txt"}
    )]

def load_uploaded_files(uploaded_files):
    all_docs, skipped = [], []
    for uploaded in uploaded_files:
        data = uploaded.getvalue()
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            skipped.append(f"{uploaded.name}: file exceeds {MAX_FILE_MB} MB limit.")
            continue
        suffix = Path(uploaded.name).suffix.lower()
        try:
            if suffix == ".pdf":
                docs = _extract_pdf(data, uploaded.name)
            elif suffix == ".docx":
                docs = _extract_docx(data, uploaded.name)
            elif suffix == ".txt":
                docs = _extract_txt(data, uploaded.name)
            else:
                skipped.append(f"{uploaded.name}: unsupported file type.")
                continue
            if not docs:
                skipped.append(f"{uploaded.name}: no readable text found. Scanned PDFs need OCR, which is not included in V1.")
                continue
            all_docs.extend(docs)
        except Exception as exc:
            skipped.append(f"{uploaded.name}: could not read file ({exc}).")
    return all_docs, skipped
