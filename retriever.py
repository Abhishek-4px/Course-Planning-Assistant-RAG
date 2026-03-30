
from typing import List, Dict, Tuple, Any

from langchain_community.vectorstores import FAISS

from config import TOP_K, EVIDENCE_KEYWORDS


def retrieve(vectorstore: FAISS, query: str, top_k: int = TOP_K) -> List[Tuple[str, Dict[str, Any]]]:
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [(doc.page_content, doc.metadata) for doc, _score in results]


def evidence_filter(
    chunks: List[Tuple[str, Dict[str, Any]]],
) -> List[Tuple[str, Dict[str, Any]]]:
    priority = []
    for text, meta in chunks:
        text_lower = text.lower()
        if any(kw in text_lower for kw in EVIDENCE_KEYWORDS):
            priority.append((text, meta))

    if priority:
        return priority
    
    return chunks


def aggregate_by_page(
    chunks: List[Tuple[str, Dict[str, Any]]],
) -> str:
    
    groups: Dict[Tuple[str, int], List[str]] = {}
    for text, meta in chunks:
        key = (meta.get("source", "Unknown"), meta.get("page", 0))
        groups.setdefault(key, []).append(text.strip())

    blocks: List[str] = []
    for (source, page), texts in groups.items():
        header = f"[{source} - Page {page}]"
        body = "\n".join(f"- {t}" for t in texts)
        blocks.append(f"{header}\n{body}")

    return "\n\n".join(blocks)


def get_citations(chunks: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    
    seen = set()
    citations: List[str] = []
    for _, meta in chunks:
        cite = f"{meta.get('source', 'Unknown')} - Page {meta.get('page', '?')}"
        if cite not in seen:
            seen.add(cite)
            citations.append(cite)
    return citations


def retrieve_and_prepare(
    vectorstore: FAISS, query: str, top_k: int = TOP_K
) -> Dict[str, Any]:
    
    raw_chunks = retrieve(vectorstore, query, top_k)
    filtered = evidence_filter(raw_chunks)
    context = aggregate_by_page(filtered)
    citations = get_citations(filtered)

    return {
        "raw_chunks": raw_chunks,
        "filtered_chunks": filtered,
        "context": context,
        "citations": citations,
    }
