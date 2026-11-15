import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from core.embeddings import get_embeddings
from core.text_chunker import split_documents

def _manifest_path(index_dir: Path) -> Path:
    return index_dir / "manifest.json"

def _read_manifest(index_dir: Path):
    path = _manifest_path(index_dir)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"files": {}}
    return {"files": {}}

def _write_manifest(index_dir: Path, manifest):
    index_dir.mkdir(parents=True, exist_ok=True)
    _manifest_path(index_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def add_documents(documents, index_dir: Path, model_name: str, chunk_size: int, chunk_overlap: int):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(index_dir)
    existing_hashes = set(manifest.get("files", {}).keys())

    grouped = {}
    for doc in documents:
        file_hash = doc.metadata.get("file_hash", "")
        grouped.setdefault(file_hash, []).append(doc)

    new_docs, duplicates, new_files = [], [], {}
    for file_hash, pages in grouped.items():
        filename = pages[0].metadata.get("source", "unknown")
        if file_hash in existing_hashes:
            duplicates.append(filename)
            continue
        new_docs.extend(pages)
        new_files[file_hash] = {"filename": filename, "type": pages[0].metadata.get("file_type", "unknown")}

    if not new_docs:
        return {"chunks_added": 0, "documents_added": 0, "duplicates": duplicates, "errors": []}

    chunks = split_documents(new_docs, chunk_size, chunk_overlap)
    embeddings = get_embeddings(model_name)
    index_file = index_dir / "index.faiss"
    if index_file.exists():
        store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        store.add_documents(chunks)
    else:
        store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(index_dir))

    manifest.setdefault("files", {}).update(new_files)
    manifest["chunk_count"] = int(manifest.get("chunk_count", 0)) + len(chunks)
    _write_manifest(index_dir, manifest)
    return {"chunks_added": len(chunks), "documents_added": len(new_files), "duplicates": duplicates, "errors": []}

def load_vector_store(index_dir: Path, model_name: str):
    index_dir = Path(index_dir)
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError("FAISS index does not exist. Upload documents first.")
    embeddings = get_embeddings(model_name)
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)

def get_index_stats(index_dir: Path):
    index_dir = Path(index_dir)
    manifest = _read_manifest(index_dir)
    files = [item.get("filename", "unknown") for item in manifest.get("files", {}).values()]
    ready = (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()
    return {
        "ready": ready,
        "documents": len(files),
        "chunks": int(manifest.get("chunk_count", 0)),
        "files": files,
    }
