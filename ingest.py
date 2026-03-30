import os
import re
from pathlib import Path
from typing import List, Dict, Any

from PyPDF2 import PdfReader
import docx2txt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    DATASET_COURSES, DATASET_PROGRAM_GUIDE, DATASET_COLLEGE_POLICY,
    VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
)

def _extract_course_name(text: str) -> str:
    for line in text.split("\n")[:15]:
        line = line.strip()
        
        m = re.search(r"[Cc]ourse\s*[Nn]ame\s*[:]\s*(.+)", line)
        if m:
            return m.group(1).strip()
        
        m = re.search(r"[Cc]ourse\s*[Tt]itle\s*[:]\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return ""


def _extract_course_code(text: str) -> str:
    """Try to pull a course code from the first few lines."""
    for line in text.split("\n")[:15]:
        m = re.search(r"[Cc]ourse\s*[Cc]ode\s*[:]\s*([A-Z]{2,4}\d{3,5})", line)
        if m:
            return m.group(1).strip()
    return ""

def load_pdf(filepath: str, doc_category: str = "course") -> List[Dict[str, Any]]:
    reader = PdfReader(filepath)
    filename = os.path.basename(filepath)
    pages = []
    full_first_page = ""

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if i == 0:
            full_first_page = text
        pages.append({
            "text": text,
            "metadata": {
                "source": filename,
                "page": i + 1,  
                "source_type": "pdf",
                "category": doc_category,
            },
        })

    course_name = _extract_course_name(full_first_page)
    course_code = _extract_course_code(full_first_page)
    for p in pages:
        p["metadata"]["course_name"] = course_name
        p["metadata"]["course_code"] = course_code

    return pages


def load_docx(filepath: str, doc_category: str = "course") -> List[Dict[str, Any]]:
    """
    Load a DOCX file.  Since DOCX has no page concept, each chunk
    (after splitting) gets a simulated page = chunk_id.
    Returns raw text with metadata stub — splitting happens later,
    but we mark source_type = "docx" so the chunker can assign page IDs.
    """
    text = docx2txt.process(filepath)
    filename = os.path.basename(filepath)
    course_name = _extract_course_name(text)
    course_code = _extract_course_code(text)

    return [{
        "text": text,
        "metadata": {
            "source": filename,
            "page": 0,  
            "source_type": "docx",
            "category": doc_category,
            "course_name": course_name,
            "course_code": course_code,
        },
    }]

def load_all_documents() -> List[Dict[str, Any]]:
    all_docs: List[Dict[str, Any]] = []

    folder_map = {
        DATASET_COURSES: "course",
        DATASET_PROGRAM_GUIDE: "program_guide",
        DATASET_COLLEGE_POLICY: "college_policy",
    }

    for folder, category in folder_map.items():
        if not folder.exists():
            print(f"  Folder not found: {folder}")
            continue
        for f in sorted(folder.iterdir()):
            ext = f.suffix.lower()
            try:
                if ext == ".pdf":
                    all_docs.extend(load_pdf(str(f), doc_category=category))
                elif ext in (".docx", ".doc"):
                    all_docs.extend(load_docx(str(f), doc_category=category))
                else:
                    print(f"⏭️  Skipping unsupported file: {f.name}")
            except Exception as e:
                print(f"❌ Error loading {f.name}: {e}")

    print(f" Loaded {len(all_docs)} raw page/doc segments from {len(folder_map)} folders.")
    return all_docs

def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: List[Dict[str, Any]] = []

    for doc in docs:
        text = doc["text"]
        meta = doc["metadata"].copy()

        if not text.strip():
            continue

        splits = splitter.split_text(text)
        for j, chunk_text in enumerate(splits):
            chunk_meta = meta.copy()
            if meta["source_type"] == "docx":
                chunk_meta["page"] = j + 1
            chunks.append({"text": chunk_text, "metadata": chunk_meta})

    print(f" Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks

def build_vector_store(chunks: List[Dict[str, Any]], persist: bool = True) -> FAISS:
    print(" Loading embedding model")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"Building FAISS index with {len(texts)} chunks")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    if persist:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(VECTOR_STORE_DIR))
        print(f" Vector store saved to {VECTOR_STORE_DIR}")

    return vectorstore


def load_vector_store() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
def run_ingestion():
    docs = load_all_documents()
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
    print("🎉 Ingestion complete!")


if __name__ == "__main__":
    run_ingestion()
