from __future__ import annotations
from typing import Any, Callable
from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document

class EmbeddingStore:
    def __init__(self, collection_name: str = "documents", embedding_fn: Callable[[str], list[float]] | None = None) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": self._embedding_fn(doc.content)
        }

    def add_documents(self, docs: list[Document]) -> None:
        for doc in docs:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vec = self._embedding_fn(query)
        results = []
        for record in self._store:
            score = compute_similarity(query_vec, record["embedding"])
            results.append({**record, "score": score})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_collection_size(self) -> int:
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        filtered_records = self._store
        if metadata_filter:
            filtered_records = [
                r for r in self._store 
                if all(r['metadata'].get(k) == v for k, v in metadata_filter.items())
            ]
        
        query_vec = self._embedding_fn(query)
        results = []
        for record in filtered_records:
            score = compute_similarity(query_vec, record["embedding"])
            results.append({**record, "score": score})
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        initial_count = len(self._store)
        self._store = [r for r in self._store if r["id"] != doc_id]
        return len(self._store) < initial_count