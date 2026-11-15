import os
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    groq_api_key: str
    groq_model: str
    embedding_model: str
    index_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int

def _secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)

def get_settings() -> Settings:
    return Settings(
        groq_api_key=_secret("GROQ_API_KEY"),
        groq_model=_secret("GROQ_MODEL", "openai/gpt-oss-120b"),
        embedding_model=_secret("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        index_dir=Path(_secret("FAISS_INDEX_DIR", "knowledge_base/faiss_index")),
        chunk_size=int(_secret("CHUNK_SIZE", "800")),
        chunk_overlap=int(_secret("CHUNK_OVERLAP", "150")),
        top_k=int(_secret("TOP_K", "4")),
    )
